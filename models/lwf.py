"""Learning without Forgetting (LwF) for Continual Learning on Seq-CIFAR-10.

Li & Hoiem, 2018 — "Learning without Forgetting"

Before training on task t, the current model (old model) runs on the new
task's data to produce soft targets. During training the loss combines:

    L = CE(new classes) + lambda_lwf * KD(old soft targets, current output)

The distillation term uses only the logits for old classes (i.e. the neurons
added for tasks 0 … t-1); the new task neurons are excluded from KD.

Usage
-----
    from models.lwf import train_lwf

    results = train_lwf(backbone_weights='backbone.pth')
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data.prepare_cifar10 import task_splits
from models.train_backbone import BackboneModel, build_backbone, EMBEDDING_DIM
from models.evaluate import ExpandableHead, evaluate_class_il, evaluate_task_il, train_task_head

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLASSES_PER_TASK = 2


def _collect_soft_targets(
    backbone_model: nn.Module,
    joint_head: ExpandableHead,
    loader: DataLoader,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the old model on *loader* and return soft-targets for old classes.

    Returns
    -------
    all_images       : Tensor (N, C, H, W) – CPU
    soft_targets     : Tensor (N, n_old_classes) – soft probabilities at temperature T
    """
    backbone_model.train(False)
    joint_head.train(False)

    images_list, targets_list = [], []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            embeddings, _ = backbone_model(images)
            logits = joint_head(embeddings)                 # (B, n_old_classes)
            soft = F.softmax(logits / temperature, dim=1)  # soften
            images_list.append(images.cpu())
            targets_list.append(soft.cpu())

    backbone_model.train(True)
    joint_head.train(True)

    return torch.cat(images_list), torch.cat(targets_list)


def _kd_loss(
    current_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    n_old_classes: int,
    temperature: float,
) -> torch.Tensor:
    """KL-divergence distillation loss over old class logits only."""
    old_logits = current_logits[:, :n_old_classes]
    log_probs  = F.log_softmax(old_logits / temperature, dim=1)
    # KL( soft_targets || current ) = sum( T * log(T/S) ) — here we use NLL form
    return F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)


def train_lwf(
    backbone_weights: str = 'backbone.pth',
    lambda_lwf: float = 1.0,
    temperature: float = 2.0,
    num_epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 64,
    head_epochs: int = 5,
    head_lr: float = 8e-4,
    verbose: bool = True,
) -> dict:
    """Train LwF sequentially over all 5 tasks.

    Parameters
    ----------
    backbone_weights : str   – path to pre-trained backbone
    lambda_lwf       : float – weight on the distillation term
    temperature      : float – softmax temperature for soft targets
    num_epochs       : int   – epochs per task
    lr               : float – learning rate
    batch_size       : int
    head_epochs      : int   – epochs for Task-IL head
    head_lr          : float
    verbose          : bool

    Returns
    -------
    dict with 'class_il' and 'task_il' lists
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

    joint_head    = ExpandableHead(in_features=EMBEDDING_DIM, n_classes=N_CLASSES_PER_TASK).to(device)
    task_heads    = []
    criterion_ce  = nn.CrossEntropyLoss()

    class_il_accs = []
    task_il_accs  = []

    for task_idx in range(len(task_splits)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"LwF  |  Task {task_idx + 1} / {len(task_splits)}")
            print(f"{'='*60}")

        task = task_splits[task_idx]
        train_loader = DataLoader(task['train'], batch_size=batch_size, shuffle=True)

        n_old_classes = joint_head.out_features  # classes BEFORE expansion

        # ── Collect soft targets from old model (skip for first task) ─────────
        if task_idx > 0:
            if verbose:
                print("  Collecting soft targets from old model…")
            all_images, soft_targets = _collect_soft_targets(
                backbone_model, joint_head, train_loader, temperature
            )
            kd_dataset  = TensorDataset(all_images, soft_targets)
            kd_loader   = DataLoader(kd_dataset, batch_size=batch_size, shuffle=True)

        # ── Expand joint head for new task ────────────────────────────────────
        if task_idx > 0:
            joint_head.expand(N_CLASSES_PER_TASK)
        joint_head = joint_head.to(device)

        optimizer = torch.optim.Adam(
            list(backbone_model.parameters()) + list(joint_head.parameters()),
            lr=lr,
        )

        # Build a CE loader that also yields soft-target batches when needed
        # We iterate both loaders in lock-step (zip stops at the shorter one)
        for epoch in range(num_epochs):
            backbone_model.train(True)
            joint_head.train(True)
            total_loss = 0.0

            if task_idx == 0:
                # First task: standard CE only
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    embeddings, _ = backbone_model(images)
                    loss = criterion_ce(joint_head(embeddings), labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            else:
                # Joint iteration: one batch of new-task labels + one of soft targets
                for (imgs_ce, labels_ce), (imgs_kd, soft_tgt) in zip(train_loader, kd_loader):
                    imgs_ce, labels_ce = imgs_ce.to(device), labels_ce.to(device)
                    imgs_kd, soft_tgt  = imgs_kd.to(device), soft_tgt.to(device)

                    optimizer.zero_grad()

                    # CE term on new task
                    emb_ce, _ = backbone_model(imgs_ce)
                    ce_loss   = criterion_ce(joint_head(emb_ce), labels_ce)

                    # KD term on same samples (using images from soft-target set)
                    emb_kd, _ = backbone_model(imgs_kd)
                    cur_logits = joint_head(emb_kd)
                    kd = _kd_loss(cur_logits, soft_tgt, n_old_classes, temperature)

                    loss = ce_loss + lambda_lwf * kd
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            if verbose:
                batches = len(train_loader)
                print(f"  Epoch [{epoch + 1}/{num_epochs}]  Loss: {total_loss / batches:.4f}")

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

        # ── Evaluate ──────────────────────────────────────────────────────────
        class_il = evaluate_class_il(backbone_model, joint_head, task_splits, task_idx + 1)
        task_il  = evaluate_task_il(backbone_model, task_heads, task_splits, task_idx + 1)
        class_il_accs.append(class_il)
        task_il_accs.append(task_il)

        if verbose:
            print(f"  >> Class-IL: {class_il:.2f}%  |  Task-IL: {task_il:.2f}%")

    return {'class_il': class_il_accs, 'task_il': task_il_accs}


if __name__ == '__main__':
    results = train_lwf()
    print("\nFinal results:")
    for i, (cil, til) in enumerate(zip(results['class_il'], results['task_il'])):
        print(f"  After task {i + 1}: Class-IL={cil:.2f}%  Task-IL={til:.2f}%")
