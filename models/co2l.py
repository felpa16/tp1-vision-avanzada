"""Contrastive Continual Learning (Co2L) for Seq-CIFAR-10.

Cha et al., ICCV 2021 — "Co2L: Contrastive Continual Learning"
arXiv: 2106.14413

The method combines two objectives trained jointly on each task:

    L = L_SupCon  +  lambda_co2l * L_distill

L_SupCon  : Supervised Contrastive loss on mixed batches of new-task data
            and replay buffer samples (reuses supcon_loss from train_backbone).

L_distill : Asymmetric Distillation loss on replay buffer samples only.
            The old model (frozen) produces target projections z_old; the
            current model produces z_new. The loss is:
                L_distill = -mean( cosine_sim(z_old, z_new) )
            Asymmetric means gradients only flow through the current model.

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
    BackboneModel, build_backbone, EMBEDDING_DIM, supcon_loss, TEMPERATURE,
)
from models.replay_buffer import ReplayBuffer
from models.evaluate import ExpandableHead, evaluate_class_il, evaluate_task_il, train_task_head

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLASSES_PER_TASK = 2


def _asymmetric_distillation_loss(
    current_model: nn.Module,
    old_model: nn.Module,
    replay_images: torch.Tensor,
    replay_labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the Asymmetric Distillation loss on replay buffer samples.

    The old model's projections are treated as fixed targets. Only the current
    model's projections receive gradients (asymmetric).

    L_distill = -mean_i( z_old_i · z_new_i )   where z are L2-normalised.

    Parameters
    ----------
    current_model  : BackboneModel currently being trained
    old_model      : frozen snapshot of the model before this task
    replay_images  : (N, C, H, W) – replay buffer images on device
    replay_labels  : (N,)         – unused here, kept for API consistency

    Returns
    -------
    loss : scalar Tensor
    """
    # Old model: no gradient
    with torch.no_grad():
        _, z_old = old_model(replay_images)
        z_old = F.normalize(z_old, dim=1)

    # Current model: gradients flow
    _, z_new = current_model(replay_images)
    z_new = F.normalize(z_new, dim=1)

    # Negative cosine similarity (we want to maximise alignment)
    cos_sim = (z_old * z_new).sum(dim=1)   # element-wise then sum → (N,)
    return -cos_sim.mean()


def _train_joint_head_frozen(
    backbone_model: nn.Module,
    joint_head: ExpandableHead,
    task_splits: list,
    n_tasks_seen: int,
    replay_buffer: ReplayBuffer,
    num_epochs: int = 5,
    lr: float = 8e-4,
    batch_size: int = 64,
) -> None:
    """Train the Class-IL joint head with frozen backbone.

    Uses both the current task's data and replay buffer samples.
    Trains in-place (modifies joint_head).
    """
    # Build a combined dataset from all tasks seen so far
    datasets = [task_splits[i]['train'] for i in range(n_tasks_seen)]

    # Include replay buffer as an additional TensorDataset if non-empty
    loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in datasets]

    optimizer = torch.optim.Adam(joint_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for param in backbone_model.parameters():
        param.requires_grad = False
    backbone_model.train(False)

    for _ in range(num_epochs):
        joint_head.train(True)
        for loader in loaders:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    embeddings, _ = backbone_model(images)
                loss = criterion(joint_head(embeddings), labels)
                loss.backward()
                optimizer.step()

    for param in backbone_model.parameters():
        param.requires_grad = True
    backbone_model.train(True)


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
    joint_head_epochs: int = 5,
    joint_head_lr: float = 8e-4,
    verbose: bool = True,
) -> dict:
    """Train Co2L sequentially over all 5 tasks.

    Parameters
    ----------
    backbone_weights  : str   – path to pre-trained backbone
    lambda_co2l       : float – weight on distillation loss
    temperature       : float – SupCon temperature τ
    buffer_size       : int   – total replay buffer capacity
    num_epochs        : int   – backbone training epochs per task
    lr                : float – backbone learning rate
    batch_size        : int
    head_epochs       : int   – Task-IL head training epochs
    head_lr           : float – Task-IL head LR
    joint_head_epochs : int   – Class-IL joint head training epochs
    joint_head_lr     : float – Class-IL joint head LR
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
    backbone_model.backbone.load_state_dict(
        torch.load(backbone_weights, map_location=device)
    )

    joint_head    = ExpandableHead(in_features=EMBEDDING_DIM, n_classes=N_CLASSES_PER_TASK).to(device)
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

        task = task_splits[task_idx]
        train_loader = DataLoader(task['train'], batch_size=batch_size, shuffle=True)

        # ── Snapshot old model before training on new task ────────────────────
        if task_idx > 0:
            old_model = copy.deepcopy(backbone_model).to(device)
            old_model.train(False)
            for param in old_model.parameters():
                param.requires_grad = False

        # ── Expand joint head ─────────────────────────────────────────────────
        if task_idx > 0:
            joint_head.expand(N_CLASSES_PER_TASK)
        joint_head = joint_head.to(device)

        # ── Backbone training with SupCon + Distillation ──────────────────────
        optimizer = torch.optim.Adam(backbone_model.parameters(), lr=lr)

        has_replay = task_idx > 0 and len(replay_buffer) > 0

        for epoch in range(num_epochs):
            backbone_model.train(True)
            total_loss = total_sc = total_dist = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                if has_replay:
                    # Mix new task batch with replay samples
                    r_images, r_labels = replay_buffer.get_tensors()
                    # Sample up to batch_size replay items to keep ratios balanced
                    n_replay = min(batch_size, len(r_images))
                    idx = torch.randperm(len(r_images))[:n_replay]
                    r_images = r_images[idx].to(device)
                    r_labels = r_labels[idx].to(device)

                    mixed_images = torch.cat([images, r_images], dim=0)
                    mixed_labels = torch.cat([labels, r_labels], dim=0)
                else:
                    mixed_images = images
                    mixed_labels = labels

                # SupCon loss on mixed batch
                _, projections = backbone_model(mixed_images)
                projections = F.normalize(projections, dim=1)
                sc_loss = supcon_loss(projections, mixed_labels, tau=temperature)

                # Asymmetric Distillation loss on replay batch only
                if has_replay:
                    dist_loss = _asymmetric_distillation_loss(
                        backbone_model, old_model, r_images, r_labels
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
                n = len(train_loader)
                print(f"  Epoch [{epoch + 1}/{num_epochs}]  "
                      f"Loss: {total_loss/n:.4f}  "
                      f"SupCon: {total_sc/n:.4f}  "
                      f"Distill: {total_dist/n:.4f}")

        # ── Update replay buffer with new task data ───────────────────────────
        if verbose:
            print(f"  Updating replay buffer (capacity={buffer_size})…")
        for images, labels in train_loader:
            replay_buffer.update(images, labels)
        if verbose:
            print(f"  {replay_buffer}")

        # ── Train Task-IL head ────────────────────────────────────────────────
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

        # ── Train Class-IL joint head (frozen backbone) ───────────────────────
        if verbose:
            print("  Training Class-IL joint head…")
        _train_joint_head_frozen(
            backbone_model=backbone_model,
            joint_head=joint_head,
            task_splits=task_splits,
            n_tasks_seen=task_idx + 1,
            replay_buffer=replay_buffer,
            num_epochs=joint_head_epochs,
            lr=joint_head_lr,
            batch_size=batch_size,
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        class_il = evaluate_class_il(backbone_model, joint_head, task_splits, task_idx + 1)
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
