import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from data.prepare_cifar10 import task_splits
from data.prepare_cifar10 import TASKS

backbone = resnet18()
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
backbone.maxpool = nn.Identity()
embedding_dim = backbone.fc.in_features
backbone.fc = nn.Identity()

class BackboneModel:
    def __init__(self, backbone, embedding_dim, intermediate_dim, projection_dim):
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

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = -mean_log_prob_pos.mean()

    return loss
        


for task_number, task in task_splits:
    train_loader = DataLoader(task['train'], batch_size=64, shuffle=True)
    test_loader = DataLoader(task['test'], batch_size=64, shuffle=False)

    class_indices = TASKS[task_number]
    label_map = {cls : i   for i, cls in enumerate(class_indices)}

    for images, labels in train_loader:
        labels = torch.tensor([label_map[l.item()] for l in labels])