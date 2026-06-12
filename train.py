import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross entropy loss for sequence prediction.

    logits:  (B, T, vocab_size)
    targets: (B, T)

    PyTorch cross_entropy expects:
        input:  (N, C)
        target: (N,)

    Therefore, we flatten batch and time dimensions:
        (B, T, vocab_size) -> (B*T, vocab_size)
        (B, T)             -> (B*T,)
    """

    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(B * T)
    return F.cross_entropy(logits, targets)


def train_one_epoch(model, dataloader: DataLoader, optimizer, device: str, max_steps=None):
    """
    Train the model for one epoch.

    The core training loop follows the standard PyTorch pattern:

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    This is the same training pattern emphasized in class.
    """

    model.train()
    total_loss = 0.0
    steps = 0

    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_loss(model, dataloader: DataLoader, device: str, max_steps=20):
    """
    Evaluate average loss without gradient updates.

    model.eval() disables dropout behavior during evaluation.
    torch.no_grad() prevents unnecessary gradient computation.
    """

    model.eval()
    total_loss = 0.0
    steps = 0

    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)

        total_loss += loss.item()
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    return total_loss / max(steps, 1)
