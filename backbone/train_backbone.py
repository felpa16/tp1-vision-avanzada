import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from data.prepare_cifar10 import task_splits, TASKS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

backbone = resnet18()
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
backbone.maxpool = nn.Identity()
embedding_dim = backbone.fc.in_features
backbone.fc = nn.Identity()


class BackboneModel(nn.Module):
    def __init__(self, backbone, embedding_dim, intermediate_dim, projection_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, projection_dim)
        )

    def forward(self, x):
        embedding = self.backbone(x)
        projection = self.projection_head(embedding)
        return embedding, projection


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

    if(pos_counts == 0).any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = -mean_log_prob_pos.mean()
    return loss

NUM_EPOCHS = 25
LEARNING_RATE = 5e-3
TEMPERATURE = 0.07

model = BackboneModel(
    backbone=backbone,
    embedding_dim=512,
    intermediate_dim=256,
    projection_dim=128
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


task_number = 0
task = task_splits[task_number]
train_loader = DataLoader(task['train'], batch_size=64, shuffle=True)
test_loader = DataLoader(task['test'], batch_size=64, shuffle=False)
class_indices = TASKS[task_number]

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        _, projections = model(images)
        projections = nn.functional.normalize(projections, dim=1)

        loss = supcon_loss(projections, labels, tau=TEMPERATURE)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}")

torch.save(model.backbone.state_dict(), 'backbone.pth')
print("\nBackbone saved to backbone.pth")