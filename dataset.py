from pathlib import Path
from typing import Dict, Tuple

import requests
import torch
from torch.utils.data import Dataset


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(data_dir: str = "data") -> Path:
    """
    Download tiny Shakespeare text if it does not already exist.

    The project uses a character-level language modeling task.
    Each character is treated as one token.
    """

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    input_path = data_path / "input.txt"
    if input_path.exists():
        return input_path

    response = requests.get(TINY_SHAKESPEARE_URL, timeout=20)
    response.raise_for_status()
    input_path.write_text(response.text, encoding="utf-8")
    return input_path


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build character vocabulary.

    stoi: string-to-index dictionary
    itos: index-to-string dictionary
    """

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    """
    Convert text into a tensor of integer token ids.
    """

    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(indices, itos: Dict[int, str]) -> str:
    """
    Convert token ids back into text.
    """

    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    return "".join(itos[int(i)] for i in indices)


class NextTokenDataset(Dataset):
    """
    GPT-style next-token prediction dataset.

    The key idea from the class notebooks is that the input and target are
    shifted by one position.

    Example:
        text:  "I am a student"
        x:     "I am a studen"
        y:     " am a student"

    For each position, the model sees previous tokens and predicts the next token.
    This matches the block_size idea from the earlier Dataset Class exercise.
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def train_val_split(data: torch.Tensor, train_ratio: float = 0.9):
    n = int(train_ratio * len(data))
    return data[:n], data[n:]
