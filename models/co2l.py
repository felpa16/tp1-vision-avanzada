"""Contrastive Continual Learning (Co2L) for Seq-CIFAR-10.

Cha et al., ICCV 2021 — "Co2L: Contrastive Continual Learning"
arXiv: 2106.14413

The method combines two objectives trained jointly on each task:

    L = L_asym_SupCon  +  lambda_co2l * L_IRD

L_asym_SupCon : Asymmetric Supervised Contrastive loss (Eq. 10).
                Only current-task samples act as anchors; replay buffer
                samples contribute to the denominator and as positives,
                but are never anchors. This prevents overfitting to the
                small replay set.

L_IRD         : Instance-wise Relation Distillation loss (Eq. 13).
                Preserves the relational structure of the representation
                space by minimising the cross-entropy between the old
                model's and current model's normalised pairwise-similarity
                distributions over the full mixed batch.

After each task:
  1. The replay buffer is updated with reservoir sampling.
  2. A Task-IL head is trained (frozen backbone) for Task-IL evaluation.
  3. The joint expandable head is trained (frozen backbone) for Class-IL.

Usage
-----
    from models.co2l import train_co2l

    results = train_co2l(backbone_weights='backbone.pth')
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from data.prepare_cifar10 import task_splits
from models.train_backbone import (
    BackboneModel, build_backbone, EMBEDDING_DIM, TEMPERATURE,
)
from models.replay_buffer import ReplayBuffer
from models.evaluate import evaluate_task_il, train_task_head

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLASSES_PER_TASK = 2


# ── Two-view augmentation helpers ─────────────────────────────────────────────

class _TwoViewSubset(torch.utils.data.Dataset):
    """Wraps a Subset and returns (view1, view2, label).

    Because CIFAR-10's train_transform applies RandomCrop and
    RandomHorizontalFlip stochastically in __getitem__, calling the
    same index twice yields two independently augmented views —
    exactly the multiviewed batch that SupCon requires.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, label = self.dataset[idx]
        img2, _     = self.dataset[idx]   # different random aug
        return img1, img2, label


def _augment_tensor(img: torch.Tensor) -> torch.Tensor:
    """Apply RandomHorizontalFlip + RandomCrop(32, padding=4) to a stored tensor.

    Used to generate a second augmented view of replay-buffer images, which
    are stored as already-normalised tensors (cannot be re-fed through PIL
    transforms).
    """
    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        img = img.flip(-1)
    # Simulate RandomCrop(32, padding=4): reflect-pad to 40×40 then crop
    img = F.pad(img.unsqueeze(0), [4, 4, 4, 4], mode='reflect').squeeze(0)
    top  = torch.randint(0, 9, (1,)).item()   # offset ∈ [0, 8]
    left = torch.randint(0, 9, (1,)).item()
    return img[:, top:top + 32, left:left + 32]


def _asym_supcon_loss(
    projections: torch.Tensor,
    labels: torch.Tensor,
    n_current: int,
    tau: float,
) -> torch.Tensor:
    """Asymmetric Supervised Contrastive Loss (Co2L paper, Eq. 10).

    Only the first *n_current* samples (current-task batch) act as anchors.
    All samples (current + replay) participate in the denominator and as
    potential positives. This prevents overfitting to the small replay set.

    Parameters
    ----------
    projections : (N, D) L2-normalised projection vectors
    labels      : (N,)  class labels (current task first, then replay)
    n_current   : number of current-task samples at the start of the batch
    tau         : SupCon temperature τ
    """
    device = projections.device
    N = projections.size(0)

    # Positive mask: same label, excluding self
    lbl = labels.unsqueeze(1)
    pos_mask = (lbl == lbl.T).float()
    diag_mask = 1.0 - torch.eye(N, device=device)
    pos_mask = pos_mask * diag_mask

    # Log-prob with numerical stability
    sim = projections @ projections.T / tau
    max_sim = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - max_sim) * diag_mask
    log_prob = (sim - max_sim) - torch.log(exp_sim.sum(dim=1, keepdim=True))

    # Only current-task samples (rows 0..n_current-1) are anchors
    cur_log_prob = log_prob[:n_current]   # (n_current, N)
    cur_pos_mask = pos_mask[:n_current]   # (n_current, N)

    pos_counts = cur_pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (cur_pos_mask[valid] * cur_log_prob[valid]).sum(dim=1) / pos_counts[valid]
    return -mean_log_prob_pos.mean()


def _ird_loss(
    current_model: nn.Module,
    old_model: nn.Module,
    images: torch.Tensor,
    kappa: float = 0.07,
) -> torch.Tensor:
    """Instance-wise Relation Distillation (IRD) loss (Co2L paper, Eq. 13).

    Preserves the *relational structure* of the representation space by
    minimising the cross-entropy between the old model's normalised
    pairwise-similarity distribution and the current model's distribution:

        L_IRD = -1/N * sum_i  p(x_i; ψ_old) · log p(x_i; ψ_current)

    where p(x_i; ψ, κ)[j] = softmax over similarities to all other samples.

    Gradients flow only through the current model (old model is frozen).

    Parameters
    ----------
    current_model : BackboneModel being trained
    old_model     : frozen snapshot from end of previous task
    images        : (N, C, H, W) full mixed batch (current + replay)
    kappa         : temperature for both models (fixed throughout training)
    """
    with torch.no_grad():
        _, z_old = old_model(images)
        z_old = F.normalize(z_old, dim=1)

    _, z_new = current_model(images)
    z_new = F.normalize(z_new, dim=1)

    N = images.size(0)
    off_diag = ~torch.eye(N, dtype=torch.bool, device=images.device)

    # Old model: target similarity distribution (no grad)
    sim_old = (z_old @ z_old.T / kappa).masked_fill(~off_diag, float('-inf'))
    p_old = F.softmax(sim_old, dim=1)          # (N, N)

    # Current model: log-similarity distribution
    sim_new = (z_new @ z_new.T / kappa).masked_fill(~off_diag, float('-inf'))
    log_p_new = F.log_softmax(sim_new, dim=1)  # (N, N)

    return -(p_old * log_p_new).sum(dim=1).mean()


def _subset_to_tensors(subset, batch_size: int = 256) -> TensorDataset:
    """Convert a Subset (which may return int labels) into a TensorDataset."""
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    all_images, all_labels = [], []
    for images, labels in loader:
        all_images.append(images)
        all_labels.append(
            labels if isinstance(labels, torch.Tensor)
            else torch.tensor(labels, dtype=torch.long)
        )
    return TensorDataset(torch.cat(all_images), torch.cat(all_labels))


def _train_fresh_classifier(
    backbone_model: nn.Module,
    task_splits: list,
    n_tasks_seen: int,
    replay_buffer: ReplayBuffer,
    embedding_dim: int,
    num_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> nn.Module:
    """Train a fresh linear classifier for Class-IL evaluation (Co2L paper §5).

    Per the paper: after each task a new linear classifier is trained on the
    frozen representation using current-task samples + replay buffer, with
    class-balanced sampling. This decouples classifier quality from backbone
    quality so that Class-IL truly measures representation transferability.

    Parameters
    ----------
    backbone_model : frozen during classifier training
    task_splits    : full task split list
    n_tasks_seen   : how many tasks have been trained so far
    replay_buffer  : contains samples from all previous tasks
    embedding_dim  : dimension of backbone output (e.g. 512)
    num_epochs     : classifier training epochs (paper uses 100)
    lr             : learning rate
    batch_size     : mini-batch size for classifier training

    Returns
    -------
    nn.Linear  – trained linear classifier (EMBEDDING_DIM → 2*n_tasks_seen)
    """
    # Current task data + replay buffer
    current_task = task_splits[n_tasks_seen - 1]
    datasets = [_subset_to_tensors(current_task['train'], batch_size=batch_size)]

    if len(replay_buffer) > 0:
        r_images, r_labels = replay_buffer.get_tensors()
        datasets.append(TensorDataset(r_images, r_labels))

    combined = ConcatDataset(datasets)
    loader   = DataLoader(combined, batch_size=batch_size, shuffle=True)

    n_classes = N_CLASSES_PER_TASK * n_tasks_seen
    head      = nn.Linear(embedding_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for param in backbone_model.parameters():
        param.requires_grad = False
    backbone_model.train(False)

    for _ in range(num_epochs):
        head.train(True)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                embeddings, _ = backbone_model(images)
            loss = criterion(head(embeddings), labels)
            loss.backward()
            optimizer.step()

    for param in backbone_model.parameters():
        param.requires_grad = True
    backbone_model.train(True)

    return head


def _eval_with_fresh_head(
    backbone_model: nn.Module,
    head: nn.Module,
    task_splits: list,
    n_tasks_seen: int,
    batch_size: int = 256,
) -> float:
    """Evaluate Class-IL accuracy using a pre-trained linear head."""
    backbone_model.train(False)
    head.train(False)
    correct = total = 0
    with torch.no_grad():
        for t in range(n_tasks_seen):
            for images, labels in DataLoader(task_splits[t]['test'],
                                             batch_size=batch_size, shuffle=False):
                images, labels = images.to(device), labels.to(device)
                embeddings, _ = backbone_model(images)
                preds = head(embeddings).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
    backbone_model.train(True)
    return 100.0 * correct / total if total > 0 else 0.0


def train_co2l(
    backbone_weights: str = 'backbone.pth',
    lambda_co2l: float = 1.0,
    temperature: float = TEMPERATURE,
    buffer_size: int = 200,
    num_epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 64,
    head_epochs: int = 5,
    head_lr: float = 8e-4,
    joint_head_epochs: int = 100,
    joint_head_lr: float = 1e-3,
    verbose: bool = True,
) -> dict:
    """Train Co2L sequentially over all 5 tasks.

    Parameters
    ----------
    backbone_weights  : str   – path to pre-trained backbone
    lambda_co2l       : float – weight on IRD distillation loss weight λ
    temperature       : float – SupCon / IRD temperature τ (= κ)
    buffer_size       : int   – total replay buffer capacity
    num_epochs        : int   – backbone training epochs per task
    lr                : float – backbone learning rate
    batch_size        : int
    head_epochs       : int   – Task-IL head training epochs
    head_lr           : float – Task-IL head LR
    joint_head_epochs : int   – fresh Class-IL linear classifier epochs (paper: 100)
    joint_head_lr     : float – fresh Class-IL linear classifier LR
    verbose           : bool

    Returns
    -------
    dict with 'class_il' and 'task_il' lists (one entry per task)
    """
    backbone_model = BackboneModel(
        backbone=build_backbone(),
        embedding_dim=EMBEDDING_DIM,
        intermediate_dim=256,
        projection_dim=128,
    ).to(device)
    backbone_model.load_state_dict(
        torch.load(backbone_weights, map_location=device)
    )

    replay_buffer = ReplayBuffer(capacity=buffer_size)
    task_heads    = []

    class_il_accs = []
    task_il_accs  = []

    old_model = None  # frozen snapshot of model before current task

    for task_idx in range(len(task_splits)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Co2L  |  Task {task_idx + 1} / {len(task_splits)}")
            print(f"{'='*60}")

        task      = task_splits[task_idx]
        has_replay = task_idx > 0 and len(replay_buffer) > 0

        # ── Snapshot old model before training on new task ────────────────────
        if task_idx > 0:
            old_model = copy.deepcopy(backbone_model).to(device)
            old_model.train(False)
            for param in old_model.parameters():
                param.requires_grad = False

        # ── Two-view DataLoader for current task ──────────────────────────────
        # Each __getitem__ call applies independent random augmentations, so
        # _TwoViewSubset returns (view1, view2, label) for the 2N batch.
        two_view_loader = DataLoader(
            _TwoViewSubset(task['train']),
            batch_size=batch_size,
            shuffle=True,
        )

        # ── Backbone training with asymmetric SupCon + IRD ───────────────────
        optimizer = torch.optim.Adam(backbone_model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            backbone_model.train(True)
            total_loss = total_sc = total_dist = 0.0

            for view1, view2, labels in two_view_loader:
                view1  = view1.to(device)
                view2  = view2.to(device)
                labels = labels.to(device)
                n_curr = view1.size(0)          # current-task samples (one view)
                optimizer.zero_grad()

                if has_replay:
                    # Sample replay images
                    r_imgs, r_lbls = replay_buffer.get_tensors()
                    n_rep = min(batch_size, len(r_imgs))
                    ridx  = torch.randperm(len(r_imgs))[:n_rep]
                    r_v1  = r_imgs[ridx].to(device)
                    # Second view via tensor augmentation (flip + crop)
                    r_v2  = torch.stack([_augment_tensor(r_imgs[i]) for i in ridx]).to(device)
                    r_lbl = r_lbls[ridx].to(device)

                    # 2N batch layout: [curr_v1, curr_v2, replay_v1, replay_v2]
                    # Labels:          [curr,    curr,    replay,    replay   ]
                    mixed_images = torch.cat([view1, view2, r_v1, r_v2], dim=0)
                    mixed_labels = torch.cat([labels, labels, r_lbl, r_lbl], dim=0)
                    # Anchors = both views of current-task samples (first 2*n_curr rows)
                    n_anchors = 2 * n_curr
                else:
                    mixed_images = torch.cat([view1, view2], dim=0)
                    mixed_labels = torch.cat([labels, labels], dim=0)
                    n_anchors = 2 * n_curr   # all samples are anchors when no replay

                # Asymmetric SupCon on the 2N batch
                _, projections = backbone_model(mixed_images)
                projections = F.normalize(projections, dim=1)
                sc_loss = _asym_supcon_loss(projections, mixed_labels, n_anchors, tau=temperature)

                # IRD on the full 2N batch
                if has_replay:
                    dist_loss = _ird_loss(
                        backbone_model, old_model, mixed_images, kappa=temperature
                    )
                else:
                    dist_loss = torch.tensor(0.0, device=device)

                loss = sc_loss + lambda_co2l * dist_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_sc   += sc_loss.item()
                total_dist += dist_loss.item()

            if verbose:
                n = len(two_view_loader)
                print(f"  Epoch [{epoch + 1}/{num_epochs}]  "
                      f"Loss: {total_loss/n:.4f}  "
                      f"SupCon: {total_sc/n:.4f}  "
                      f"IRD: {total_dist/n:.4f}")

        # ── Update replay buffer (use original single-view loader) ────────────
        if verbose:
            print(f"  Updating replay buffer (capacity={buffer_size})…")
        single_loader = DataLoader(task['train'], batch_size=batch_size, shuffle=False)
        for images, labels in single_loader:
            replay_buffer.update(images, labels)
        if verbose:
            print(f"  {replay_buffer}")

        # ── Train Task-IL head (frozen backbone) ──────────────────────────────
        head = train_task_head(
            backbone_model=backbone_model,
            task_split=task,
            embedding_dim=EMBEDDING_DIM,
            n_classes=N_CLASSES_PER_TASK,
            num_epochs=head_epochs,
            lr=head_lr,
            batch_size=batch_size,
        )
        task_heads.append(head)

        # ── Train fresh linear classifier for Class-IL (paper §5) ─────────────
        # A fresh nn.Linear is trained on current task + replay buffer with the
        # backbone frozen, matching the paper's evaluation protocol.
        if verbose:
            print(f"  Training fresh Class-IL classifier ({joint_head_epochs} epochs)…")
        fresh_head = _train_fresh_classifier(
            backbone_model=backbone_model,
            task_splits=task_splits,
            n_tasks_seen=task_idx + 1,
            replay_buffer=replay_buffer,
            embedding_dim=EMBEDDING_DIM,
            num_epochs=joint_head_epochs,
            lr=joint_head_lr,
            batch_size=batch_size,
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        class_il = _eval_with_fresh_head(backbone_model, fresh_head, task_splits, task_idx + 1)
        task_il  = evaluate_task_il(backbone_model, task_heads, task_splits, task_idx + 1)
        class_il_accs.append(class_il)
        task_il_accs.append(task_il)

        if verbose:
            print(f"  >> Class-IL: {class_il:.2f}%  |  Task-IL: {task_il:.2f}%")

    return {'class_il': class_il_accs, 'task_il': task_il_accs}


if __name__ == '__main__':
    results = train_co2l()
    print("\nFinal results:")
    for i, (cil, til) in enumerate(zip(results['class_il'], results['task_il'])):
        print(f"  After task {i + 1}: Class-IL={cil:.2f}%  Task-IL={til:.2f}%")
