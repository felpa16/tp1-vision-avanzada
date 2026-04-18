"""Elastic Weight Consolidation (EWC) for Continual Learning on Seq-CIFAR-10.

Kirkpatrick et al., 2017 — "Overcoming catastrophic forgetting in neural networks"

After each task the diagonal Fisher Information Matrix is computed and the
current parameter values are snapshotted. In subsequent tasks a quadratic
penalty is added to the loss:

    L = CE(new_task) + lambda_ewc * sum_i  F_i * (theta_i - theta*_i)^2

Fisher diagonals are accumulated (summed) across tasks (online EWC variant).

Usage
-----
    from models.ewc import train_ewc

    results = train_ewc(backbone_weights='backbone.pth')
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.prepare_cifar10 import task_splits
from models.train_backbone import BackboneModel, build_backbone, EMBEDDING_DIM
from models.evaluate import ExpandableHead, evaluate_class_il, evaluate_task_il, train_task_head

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLASSES_PER_TASK = 2


class EWCState:
    """Holds the accumulated Fisher diagonal and parameter snapshot.

    Attributes
    ----------
    means  : dict[str, Tensor]  – θ* (parameter values after each task)
    fisher : dict[str, Tensor]  – accumulated diagonal Fisher across all tasks
    """

    def __init__(self):
        self.means:  dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}

    def update(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        n_samples: int = 200,
    ) -> None:
        """Compute Fisher diagonal on *loader* and accumulate into self.fisher.

        Uses the empirical Fisher (square of the gradient of the log-likelihood
        averaged over a subset of training samples).

        Parameters
        ----------
        model     : the current model (backbone + joint head must be joined
                    by the caller into a single forward pass)
        loader    : DataLoader for the just-trained task
        criterion : CrossEntropyLoss
        n_samples : how many samples to use (subsample for speed)
        """
        model.train(False)
        new_fisher: dict[str, torch.Tensor] = {}

        # Exclude the joint_head: its weights grow with each task (expand()),
        # so a snapshot taken at task t would mismatch in size at task t+1.
        # EWC regularisation is applied only to the backbone parameters.
        def _is_backbone(name: str) -> bool:
            return not name.startswith('joint_head')

        for name, param in model.named_parameters():
            if param.requires_grad and _is_backbone(name):
                new_fisher[name] = torch.zeros_like(param.data)

        samples_seen = 0
        for images, labels in loader:
            if samples_seen >= n_samples:
                break
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and _is_backbone(name):
                    new_fisher[name] += param.grad.data.pow(2)

            samples_seen += images.size(0)

        # Average and accumulate
        for name in new_fisher:
            new_fisher[name] /= max(samples_seen, 1)
            if name in self.fisher:
                self.fisher[name] += new_fisher[name]
            else:
                self.fisher[name] = new_fisher[name]

        # Snapshot current backbone parameters
        self.means = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad and _is_backbone(name)
        }

        model.train(True)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Return the EWC regularisation term  sum_i F_i * (θ_i - θ*_i)^2."""
        loss = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                diff = param - self.means[name].to(device)
                loss = loss + (self.fisher[name].to(device) * diff.pow(2)).sum()
        return loss


# ── Thin wrapper that combines backbone + head for Fisher computation ─────────

class _JointModel(nn.Module):
    """Wraps backbone + joint head for use during Fisher computation."""

    def __init__(self, backbone_model, joint_head):
        super().__init__()
        self.backbone_model = backbone_model
        self.joint_head = joint_head

    def forward(self, x):
        embeddings, _ = self.backbone_model(x)
        return self.joint_head(embeddings)


# ── Main training function ────────────────────────────────────────────────────

def train_ewc(
    backbone_weights: str = 'backbone.pth',
    lambda_ewc: float = 100.0,
    num_epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 64,
    fisher_samples: int = 200,
    head_epochs: int = 5,
    head_lr: float = 8e-4,
    verbose: bool = True,
) -> dict:
    """Train EWC sequentially over all 5 tasks.

    Parameters
    ----------
    backbone_weights : str   – path to pre-trained backbone
    lambda_ewc       : float – EWC regularisation strength
    num_epochs       : int   – epochs per task
    lr               : float – learning rate
    batch_size       : int
    fisher_samples   : int   – samples used to estimate Fisher diagonal
    head_epochs      : int   – epochs for Task-IL head training
    head_lr          : float – LR for Task-IL head
    verbose          : bool

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

    joint_head = ExpandableHead(in_features=EMBEDDING_DIM, n_classes=N_CLASSES_PER_TASK).to(device)
    ewc_state  = EWCState()
    task_heads = []
    criterion  = nn.CrossEntropyLoss()

    class_il_accs = []
    task_il_accs  = []

    for task_idx in range(len(task_splits)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"EWC  |  Task {task_idx + 1} / {len(task_splits)}")
            print(f"{'='*60}")

        task = task_splits[task_idx]
        train_loader = DataLoader(task['train'], batch_size=batch_size, shuffle=True)

        if task_idx > 0:
            joint_head.expand(N_CLASSES_PER_TASK)
        joint_head = joint_head.to(device)

        optimizer = torch.optim.Adam(
            list(backbone_model.parameters()) + list(joint_head.parameters()),
            lr=lr,
        )

        for epoch in range(num_epochs):
            backbone_model.train(True)
            joint_head.train(True)
            total_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                embeddings, _ = backbone_model(images)
                logits = joint_head(embeddings)
                ce_loss = criterion(logits, labels)

                # EWC penalty (zero on first task since ewc_state is empty)
                joint_model = _JointModel(backbone_model, joint_head)
                ewc_loss = ewc_state.penalty(joint_model)

                loss = ce_loss + lambda_ewc * ewc_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose:
                avg = total_loss / len(train_loader)
                print(f"  Epoch [{epoch + 1}/{num_epochs}]  Loss: {avg:.4f}")

        # ── Compute Fisher and snapshot parameters ────────────────────────────
        if verbose:
            print("  Computing Fisher Information Matrix…")
        joint_model = _JointModel(backbone_model, joint_head)
        ewc_state.update(joint_model, train_loader, criterion, n_samples=fisher_samples)

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
    results = train_ewc()
    print("\nFinal results:")
    for i, (cil, til) in enumerate(zip(results['class_il'], results['task_il'])):
        print(f"  After task {i + 1}: Class-IL={cil:.2f}%  Task-IL={til:.2f}%")
