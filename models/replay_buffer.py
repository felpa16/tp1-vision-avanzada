import random
import torch
from torch.utils.data import DataLoader, TensorDataset


class ReplayBuffer:
    """Fixed-capacity episodic memory with reservoir sampling.

    Reservoir sampling guarantees that every sample seen so far has an equal
    probability of being retained, regardless of the order in which tasks arrive.

    Parameters
    ----------
    capacity : int
        Maximum total number of (image, label) pairs stored across all tasks.
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._images: list[torch.Tensor] = []
        self._labels: list[int] = []
        self._n_seen = 0  # total samples ever offered to the buffer

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """Add a batch of samples using reservoir sampling.

        Parameters
        ----------
        images : Tensor  shape (N, C, H, W) — CPU tensors
        labels : Tensor  shape (N,)
        """
        images = images.cpu()
        labels = labels.cpu()

        for img, lbl in zip(images, labels):
            self._n_seen += 1
            if len(self._images) < self.capacity:
                # Buffer not yet full — always add
                self._images.append(img)
                self._labels.append(lbl.item())
            else:
                # Reservoir sampling: replace slot j with prob capacity/n_seen
                j = random.randint(0, self._n_seen - 1)
                if j < self.capacity:
                    self._images[j] = img
                    self._labels[j] = lbl.item()

    def get_loader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        """Return a DataLoader over all currently stored samples."""
        if len(self._images) == 0:
            raise RuntimeError("ReplayBuffer is empty — call update() first.")
        images = torch.stack(self._images)          # (N, C, H, W)
        labels = torch.tensor(self._labels, dtype=torch.long)
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all stored samples as a pair of tensors (images, labels)."""
        if len(self._images) == 0:
            raise RuntimeError("ReplayBuffer is empty — call update() first.")
        return torch.stack(self._images), torch.tensor(self._labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._images)

    def __repr__(self) -> str:
        return (f"ReplayBuffer(capacity={self.capacity}, "
                f"stored={len(self)}, seen={self._n_seen})")
