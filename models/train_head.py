import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.train_backbone import EMBEDDING_DIM, BackboneModel, build_backbone
from data.prepare_cifar10 import task_splits
import matplotlib.pyplot as plt

LEARNING_RATE = 0.0008
NUM_EPOCHS = 5
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Classification head ───────────────────────────────────────────────────────
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 8),
            nn.ReLU(),
            nn.Linear(embedding_dim // 8, n_classes),
        )

    def forward(self, x):
        return self.model(x)


def load_backbone(weights_path='backbone.pth'):
    """Load the frozen backbone from a saved checkpoint."""
    backbone_model = BackboneModel(
        backbone=build_backbone(),
        embedding_dim=EMBEDDING_DIM,
        intermediate_dim=256,
        projection_dim=128,
    ).to(device)
    backbone_model.backbone.load_state_dict(torch.load(weights_path, map_location=device))
    backbone_model.eval()
    for param in backbone_model.parameters():
        param.requires_grad = False
    return backbone_model


def evaluate(head, backbone_model, loader):
    """Return the accuracy (%) of *head* over *loader* using the frozen backbone."""
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            embeddings, _ = backbone_model(images)
            preds = head(embeddings).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total


def train_classifier(task_index, backbone_model, n_classes=2,
                     num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    """
    Train a ClassificationHead on the given task and return the trained head
    along with its per-epoch training losses and test accuracies.

    Parameters
    ----------
    task_index      : int   – index into task_splits (0–4)
    backbone_model  : nn.Module – frozen backbone, already on device
    n_classes       : int   – number of output classes for this task
    num_epochs      : int   – number of training epochs
    lr              : float – learning rate for Adam
    batch_size      : int   – DataLoader batch size

    Returns
    -------
    head            : trained ClassificationHead
    train_losses    : list[float]
    test_accuracies : list[float]
    """
    task = task_splits[task_index]
    train_loader = DataLoader(task['train'], batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(task['test'],  batch_size=batch_size, shuffle=False)

    head = ClassificationHead(embedding_dim=EMBEDDING_DIM, n_classes=n_classes).to(device)
    optimizer = Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses    = []
    test_accuracies = []

    print(f"\n── Task {task_index} ──────────────────────────────────")
    for epoch in range(num_epochs):
        head.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                embeddings, _ = backbone_model(images)
            loss = criterion(head(embeddings), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(head, backbone_model, test_loader)
        train_losses.append(avg_loss)
        test_accuracies.append(acc)
        print(f"  Epoch [{epoch + 1}/{num_epochs}]  Loss: {avg_loss:.4f}  Test Acc: {acc:.2f}%")

    print(f"Final Test Accuracy: {acc:.2f}%")
    return head, train_losses, test_accuracies


def plot_metrics(train_losses, test_accuracies, task_index, save_path=None):
    """Plot the loss and accuracy curves for a single task side-by-side."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'Task {task_index} – Classification Head Metrics', fontweight='bold')

    ax1.plot(epochs, train_losses, marker='o', linewidth=2, color='tab:blue')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True)

    ax2.plot(epochs, test_accuracies, marker='o', linewidth=2, color='tab:green')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Test Accuracy")
    ax2.set_ylim(0, 100)
    ax2.axhline(y=test_accuracies[-1], color='gray', linestyle='--', linewidth=1,
                label=f'Final: {test_accuracies[-1]:.2f}%')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = save_path or f'classifier_metrics_task{task_index}.png'
    plt.savefig(path, dpi=150)
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    backbone_model = load_backbone('backbone.pth')

    # Train on a single task
    # head, losses, accuracies = train_classifier(task_index=0, backbone_model=backbone_model)
    # plot_metrics(losses, accuracies, task_index=0)

    # Or loop over all 5 tasks in sequence:
    for task_index in range(len(task_splits)):
        head, losses, accuracies = train_classifier(task_index, backbone_model)
        test_loader  = DataLoader(task_splits[task_index]['test'], batch_size=BATCH_SIZE, shuffle=False)
        evaluate = evaluate(head, backbone_model, test_loader)
        print(f'Classifier accuracy for task {task_index + 1}: {evaluate}')