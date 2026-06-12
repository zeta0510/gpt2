import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """
    One head of masked self-attention.

    Input shape:
        x: (B, T, C)

    B = batch size
    T = block size, sequence length
    C = embedding dimension

    Each token is projected into query, key, and value vectors.

    - Query: what this token is looking for
    - Key: what each token contains
    - Value: information carried by each token

    Attention compares Query and Key, then uses the resulting weights
    to take a weighted sum of Value vectors.

    The lower-triangular mask prevents each position from seeing future tokens.
    This is the central difference between ordinary self-attention and
    GPT-style masked self-attention.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # tril[i, j] = 1 only when j <= i.
        # Therefore token i can attend only to itself and previous tokens.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)

        # Scaled dot-product attention.
        # q @ k.transpose(-2, -1) gives attention scores between all token pairs.
        # Division by sqrt(d_k) prevents dot products from becoming too large.
        wei = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)  # (B, T, T)

        # Mask future tokens.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Normalize attention scores into probabilities.
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)    # (B, T, head_size)

        # Weighted sum of values.
        out = wei @ v        # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple masked self-attention heads in parallel.

    A single head can learn one type of relation between tokens.
    Multi-head attention allows the model to learn several relations at once.
    """

    def __init__(self, n_embd: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate outputs from all heads along the channel dimension.
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Position-wise feedforward network.

    After attention mixes information across time positions,
    the feedforward network transforms each token representation independently.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    Transformer decoder block.

    GPT is a decoder-only Transformer.
    Each block contains:

    1. masked self-attention
    2. feedforward network
    3. residual connections
    4. layer normalization

    This implementation uses the common pre-norm form:
        x = x + attention(layer_norm(x))
        x = x + feedforward(layer_norm(x))
    """

    def __init__(self, n_embd: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, num_heads, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    Character-level Tiny GPT model.

    Input:
        idx: (B, T), integer token ids

    Forward flow:
        token ids
        -> token embedding
        -> positional embedding
        -> stacked Transformer blocks
        -> final LayerNorm
        -> linear language modeling head
        -> logits over vocabulary

    Output:
        logits: (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} is larger than block_size {self.block_size}."
            )

        token_emb = self.token_embedding_table(idx)  # (B, T, C)

        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)  # (T, C)

        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)       # (B, T, C)
        x = self.ln_f(x)         # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressively generate new tokens.

        At each step, the model:
        1. keeps only the latest block_size tokens,
        2. computes logits,
        3. uses the last time step's logits,
        4. applies softmax,
        5. samples the next token,
        6. appends it to the sequence.
        """

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)

            # Use logits from the last position only.
            logits = logits[:, -1, :]  # (B, vocab_size)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
