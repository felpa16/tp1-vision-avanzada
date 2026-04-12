import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from data.prepare_cifar10 import task_splits, TASKS
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 25
LEARNING_RATE = 5e-3
TEMPERATURE = 0.07
EMBEDDING_DIM = 512
INTERMEDIATE_DIM = 256
PROJECTION_DIM = 128
SNAPSHOT_EPOCHS = {0: 'before training', 12: 'mid-training', NUM_EPOCHS: 'after training'}


# ── Model definition ──────────────────────────────────────────────────────────
def build_backbone():
    backbone = resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    return backbone


class BackboneModel(nn.Module):
    def __init__(self, backbone, embedding_dim, intermediate_dim, projection_dim):
        super().__init__()
        self.backbone = backbone
        self.loss = nn.CrossEntropyLoss()
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, projection_dim)
        )

    def forward(self, x):
        embedding = self.backbone(x)
        projection = self.projection_head(embedding)
        return embedding, projection


# ── Loss ──────────────────────────────────────────────────────────────────────
def supcon_loss(features, labels, tau):
    device = features.device
    labels = labels.unsqueeze(1)
    mask = (labels == labels.T).float()

    sim = features @ features.T / tau
    logits_mask = torch.ones_like(mask) - torch.eye(len(mask), device=device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

    pos_counts = mask.sum(1)
    if (pos_counts == 0).any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    return -mean_log_prob_pos.mean()


# ── Visualisation helpers ─────────────────────────────────────────────────────
def collect_embeddings(model, loader):
    """Run inference over the full loader and return (embeddings, labels) as numpy arrays."""
    model.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            embeddings, _ = model(images.to(device))
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def plot_embeddings(ax, embeddings, labels, class_names, title):
    """Reduce embeddings to 2-D with t-SNE and scatter-plot them on *ax*."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    reduced = tsne.fit_transform(embeddings)

    colors = ['tab:blue', 'tab:orange']
    for i, cls in enumerate(class_names):
        mask = labels == cls
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=colors[i], label=f'Class {cls}',
                   alpha=0.6, s=15, edgecolors='none')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(loc='best', markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_snapshots(snapshots, class_indices):
    """Render the three t-SNE snapshots side-by-side and save to disk."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Latent-space representations (t-SNE) – CIFAR-10 classes '
                 + str(class_indices), fontsize=15, fontweight='bold', y=1.02)

    for ax, (epoch_key, label) in zip(axes, sorted(SNAPSHOT_EPOCHS.items())):
        emb, lbl = snapshots[epoch_key]
        plot_embeddings(ax, emb, lbl, class_indices,
                        title=f'Epoch {epoch_key} — {label}')

    plt.tight_layout()
    plt.savefig('latent_space_snapshots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure saved to latent_space_snapshots.png")


# ── Training ──────────────────────────────────────────────────────────────────
def train_backbone(task_number=0, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                   temperature=TEMPERATURE, batch_size=64, save_path='backbone.pth'):
    """
    Train the backbone with supervised contrastive loss on one task and save
    the weights. Snapshots of the embeddings are collected at epochs 0, 12,
    and *num_epochs* for t-SNE visualisation.

    Parameters
    ----------
    task_number : int   – index into task_splits (0–4)
    num_epochs  : int   – total training epochs
    lr          : float – Adam learning rate
    temperature : float – SupCon temperature τ
    batch_size  : int   – DataLoader batch size
    save_path   : str   – where to write the backbone state-dict

    Returns
    -------
    model       : trained BackboneModel
    """
    task = task_splits[task_number]
    train_loader = DataLoader(task['train'], batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(task['test'],  batch_size=batch_size, shuffle=False)
    class_indices = TASKS[task_number]

    model = BackboneModel(
        backbone=build_backbone(),
        embedding_dim=EMBEDDING_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        projection_dim=PROJECTION_DIM,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    snapshots = {}

    # Epoch-0 snapshot (before any training)
    emb, lbl = collect_embeddings(model, test_loader)
    snapshots[0] = (emb, lbl)
    np.save('snapshot_epoch_0.npy', {'embeddings': emb, 'labels': lbl}, allow_pickle=True)
    print("Snapshot collected: epoch 0 (before training)")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _, projections = model(images)
            projections = nn.functional.normalize(projections, dim=1)
            loss = supcon_loss(projections, labels, tau=temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        completed_epoch = epoch + 1
        print(f"  Epoch [{completed_epoch}/{num_epochs}]  Loss: {avg_loss:.4f}")

        if completed_epoch in SNAPSHOT_EPOCHS and completed_epoch != 0:
            emb, lbl = collect_embeddings(model, test_loader)
            snapshots[completed_epoch] = (emb, lbl)
            np.save(f'snapshot_epoch_{completed_epoch}.npy',
                    {'embeddings': emb, 'labels': lbl}, allow_pickle=True)
            print(f"Snapshot collected: epoch {completed_epoch} "
                  f"({SNAPSHOT_EPOCHS[completed_epoch]})")

    torch.save(model.backbone.state_dict(), save_path)
    print(f"\nBackbone saved to {save_path}")

    plot_snapshots(snapshots, class_indices)
    return model


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_backbone(task_number=0)