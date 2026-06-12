import random

import torch
from torch.utils.data import DataLoader

from dataset import (
    NextTokenDataset,
    build_vocab,
    download_tiny_shakespeare,
    encode,
    load_text,
    train_val_split,
)
from generate import sample_gpt
from model import TinyGPT
from train import evaluate_loss, train_one_epoch


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    # Small default settings so that the project can run quickly on CPU.
    block_size = 64
    batch_size = 64
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.2
    learning_rate = 3e-4
    epochs = 2
    max_train_steps_per_epoch = 100
    max_eval_steps = 20
    max_new_tokens = 300

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    input_path = download_tiny_shakespeare()
    text = load_text(input_path)

    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)

    data = encode(text, stoi)
    train_data, val_data = train_val_split(data)

    train_dataset = NextTokenDataset(train_data, block_size)
    val_dataset = NextTokenDataset(val_data, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"vocab_size: {vocab_size}")
    print(f"block_size: {block_size}")
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Check shape flow once.
    xb, yb = next(iter(train_loader))
    xb = xb.to(device)
    yb = yb.to(device)

    logits = model(xb)
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"target shape: {tuple(yb.shape)}")

    initial_val_loss = evaluate_loss(model, val_loader, device, max_steps=max_eval_steps)
    print(f"initial val loss: {initial_val_loss:.4f}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_steps=max_train_steps_per_epoch,
        )
        val_loss = evaluate_loss(model, val_loader, device, max_steps=max_eval_steps)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f}"
        )

    print("\nGenerated text:")
    print("-" * 60)
    generated = sample_gpt(
        model,
        start_text="\n",
        stoi=stoi,
        itos=itos,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    print(generated)
    print("-" * 60)


if __name__ == "__main__":
    main()
