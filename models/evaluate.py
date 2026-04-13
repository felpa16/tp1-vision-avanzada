import torch
import torch.nn as nn
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Expandable classification head ───────────────────────────────────────────

class ExpandableHead(nn.Module):
    """Linear head whose output dimension grows by *step* each time expand() is called.

    Old weights are preserved when expanding; new class neurons are initialised
    with Kaiming uniform (same default as nn.Linear).

    Parameters
    ----------
    in_features : int   – backbone embedding dimension (e.g. 512 for ResNet-18)
    n_classes   : int   – initial number of output classes
    """

    def __init__(self, in_features: int, n_classes: int = 2):
        super().__init__()
        self.in_features = in_features
        self.fc = nn.Linear(in_features, n_classes)

    @property
    def out_features(self) -> int:
        return self.fc.out_features

    def expand(self, n_new_classes: int) -> None:
        """Add *n_new_classes* output neurons, preserving existing weights."""
        old_out = self.fc.out_features
        new_out = old_out + n_new_classes

        new_fc = nn.Linear(self.in_features, new_out)
        with torch.no_grad():
            new_fc.weight[:old_out] = self.fc.weight
            new_fc.bias[:old_out]   = self.fc.bias
        self.fc = new_fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ── Shared evaluation functions ───────────────────────────────────────────────

def evaluate_class_il(
    backbone_model,
    joint_head: ExpandableHead,
    task_splits: list,
    n_tasks_seen: int,
    batch_size: int = 256,
) -> float:
    """Class-Incremental Learning accuracy.

    Classifies among all classes seen so far (2 * n_tasks_seen) without any
    task identity hint.
    """
    backbone_model.train(False)
    joint_head.train(False)

    correct = total = 0
    with torch.no_grad():
        for task_idx in range(n_tasks_seen):
            loader = DataLoader(
                task_splits[task_idx]['test'],
                batch_size=batch_size,
                shuffle=False,
            )
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                embeddings, _ = backbone_model(images)
                logits = joint_head(embeddings)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

    backbone_model.train(True)
    joint_head.train(True)
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_task_il(
    backbone_model,
    task_heads: list,
    task_splits: list,
    n_tasks_seen: int,
    batch_size: int = 256,
) -> float:
    """Task-Incremental Learning accuracy.

    For each seen task the corresponding 2-class head is used; task identity
    is provided at inference time.
    """
    backbone_model.train(False)

    task_accuracies = []
    with torch.no_grad():
        for task_idx in range(n_tasks_seen):
            head = task_heads[task_idx]
            head.train(False)
            loader = DataLoader(
                task_splits[task_idx]['test'],
                batch_size=batch_size,
                shuffle=False,
            )
            correct = total = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                embeddings, _ = backbone_model(images)
                preds = head(embeddings).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
            task_accuracies.append(100.0 * correct / total)
            head.train(True)

    backbone_model.train(True)
    return sum(task_accuracies) / len(task_accuracies) if task_accuracies else 0.0


# ── Convenience: train a single Task-IL head (frozen backbone) ────────────────

def train_task_head(
    backbone_model,
    task_split: dict,
    embedding_dim: int = 512,
    n_classes: int = 2,
    num_epochs: int = 5,
    lr: float = 8e-4,
    batch_size: int = 64,
) -> nn.Module:
    """Train and return a 2-class linear head for Task-IL with a frozen backbone.

    Replicates the logic of train_head.py as a reusable function so every CL
    method can call it without code duplication.
    """
    from models.train_head import ClassificationHead

    train_loader = DataLoader(task_split['train'], batch_size=batch_size, shuffle=True)

    head = ClassificationHead(embedding_dim=embedding_dim, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Freeze backbone
    for param in backbone_model.parameters():
        param.requires_grad = False
    backbone_model.train(False)

    for _ in range(num_epochs):
        head.train(True)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                embeddings, _ = backbone_model(images)
            loss = criterion(head(embeddings), labels)
            loss.backward()
            optimizer.step()

    # Restore backbone gradients for subsequent CL training
    for param in backbone_model.parameters():
        param.requires_grad = True
    backbone_model.train(True)

    return head
