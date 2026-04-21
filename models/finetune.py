"""Naive Fine-Tuning baseline for Continual Learning on Seq-CIFAR-10.

The model is fine-tuned end-to-end with cross-entropy on each new task.
No mechanism is used to prevent catastrophic forgetting. This serves as
the lower-bound reference.

Usage
-----
    from models.finetune import train_finetune

    results = train_finetune(backbone_weights='backbone.pth')
    # results['class_il'] -> list of Class-IL accuracy after each task
    # results['task_il']  -> list of Task-IL accuracy after each task
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.prepare_cifar10 import task_splits, TASKS
from models.train_backbone import BackboneModel, build_backbone, EMBEDDING_DIM
from models.evaluate import ExpandableHead, evaluate_class_il, evaluate_task_il, train_task_head

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLASSES_PER_TASK = 2


def train_finetune(
    backbone_weights: str = 'backbone.pth',
    num_epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 64,
    head_epochs: int = 5,
    head_lr: float = 8e-4,
    verbose: bool = True,
) -> dict:
    """Train the naive fine-tuning baseline sequentially over all 5 tasks.

    Parameters
    ----------
    backbone_weights : str   – path to the pre-trained backbone checkpoint
    num_epochs       : int   – epochs per task for joint backbone+head training
    lr               : float – learning rate for backbone + joint head
    batch_size       : int
    head_epochs      : int   – epochs for the Task-IL head (frozen backbone)
    head_lr          : float – learning rate for Task-IL head
    verbose          : bool  – print per-epoch metrics

    Returns
    -------
    dict with keys:
        'class_il' : list[float]  – Class-IL accuracy after each task
        'task_il'  : list[float]  – Task-IL accuracy after each task
    """
    # ── Load pre-trained backbone ─────────────────────────────────────────────
    backbone = build_backbone().to(device)
    backbone.load_state_dict("backbone.pth")
    backbone.load_state_dict(torch.load(backbone_weights, map_location=device))

    # ── Shared Class-IL head (grows with each task) ───────────────────────────
    joint_head = ExpandableHead(in_features=EMBEDDING_DIM, n_classes=N_CLASSES_PER_TASK).to(device)

    task_heads = []   # per-task 2-class heads for Task-IL
    class_il_accs = []
    task_il_accs  = []

    criterion = nn.CrossEntropyLoss()

    for task_idx in range(len(task_splits)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fine-Tune  |  Task {task_idx + 1} / {len(task_splits)}")
            print(f"{'='*60}")

        task = task_splits[task_idx]
        train_loader = DataLoader(task['train'], batch_size=batch_size, shuffle=True)

        # Expand joint head for the 2 new classes of this task
        if task_idx > 0:
            joint_head.expand(N_CLASSES_PER_TASK)
        joint_head = joint_head.to(device)

        # ── Fine-tune backbone + joint head end-to-end ────────────────────────
        optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(joint_head.parameters()),
            lr=lr,
        )

        for epoch in range(num_epochs):
            backbone.train(True)
            joint_head.train(True)
            total_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                embeddings = backbone(images)
                logits = joint_head(embeddings)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose:
                avg = total_loss / len(train_loader)
                print(f"  Epoch [{epoch + 1}/{num_epochs}]  Loss: {avg:.4f}")

        # ── Train Task-IL head (frozen backbone) ──────────────────────────────
        if verbose:
            print(f"  Training Task-IL head for task {task_idx + 1}…")
        head = train_task_head(
            backbone_model=backbone,
            task_split=task,
            embedding_dim=EMBEDDING_DIM,
            n_classes=N_CLASSES_PER_TASK,
            num_epochs=head_epochs,
            lr=head_lr,
            batch_size=batch_size,
        )
        task_heads.append(head)

        # ── Evaluate ──────────────────────────────────────────────────────────
        class_il = evaluate_class_il(backbone, joint_head, task_splits, task_idx + 1)
        task_il  = evaluate_task_il(backbone, task_heads, task_splits, task_idx + 1)
        class_il_accs.append(class_il)
        task_il_accs.append(task_il)

        if verbose:
            print(f"  >> Class-IL: {class_il:.2f}%  |  Task-IL: {task_il:.2f}%")

    return {'class_il': class_il_accs, 'task_il': task_il_accs}


if __name__ == '__main__':
    results = train_finetune()
    print("\nFinal results:")
    for i, (cil, til) in enumerate(zip(results['class_il'], results['task_il'])):
        print(f"  After task {i + 1}: Class-IL={cil:.2f}%  Task-IL={til:.2f}%")
