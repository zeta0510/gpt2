import torch

from dataset import decode


@torch.no_grad()
def sample_gpt(model, start_text: str, stoi, itos, device: str, max_new_tokens: int = 300):
    """
    Generate text from a trained TinyGPT model.

    Generation process:
    1. encode the starting text into token ids,
    2. feed it into the model,
    3. get logits for the next token,
    4. apply softmax inside model.generate,
    5. sample and append one token at a time.

    This corresponds to the prediction step discussed in class:
    logits -> softmax -> choose/sample next token -> append.
    """

    model.eval()

    if not start_text:
        start_text = "\n"

    context = torch.tensor(
        [[stoi[ch] for ch in start_text if ch in stoi]],
        dtype=torch.long,
        device=device,
    )

    if context.numel() == 0:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=max_new_tokens)
    return decode(generated[0], itos)
