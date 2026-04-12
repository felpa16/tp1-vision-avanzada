import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10('./data', train=True, download=False, transform=transform)
test_set = datasets.CIFAR10('./data', train=False, download=False, transform=transform)

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

TASKS = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]

def get_task_subset(dataset, class_indices):
    """Return a Subset containing only samples from the given classes."""
    targets = torch.tensor(dataset.targets)
    mask = torch.zeros(len(targets), dtype=torch.bool)
    for cls in class_indices:
        mask |= (targets == cls)
    indices = mask.nonzero(as_tuple=True)[0].tolist()
    return Subset(dataset, indices)


task_splits = []
for i, class_indices in enumerate(TASKS):
    train_subset = get_task_subset(train_set, class_indices)
    test_subset  = get_task_subset(test_set,  class_indices)
    task_splits.append({'train': train_subset, 'test': test_subset})
    class_names = [CIFAR10_CLASSES[c] for c in class_indices]
    # print(f"Task {i+1} {class_names}: {len(train_subset)} train, {len(test_subset)} test samples")