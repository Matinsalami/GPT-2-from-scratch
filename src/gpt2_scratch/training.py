from pathlib import Path

import torch
import torch.nn.functional as F

from .generation import generate, text_to_token_ids, token_ids_to_text


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    return F.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
    )


@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    model.eval()
    total_loss = 0.0

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(
            input_batch, target_batch, model, device
        )
        total_loss += loss.item()

    return total_loss / num_batches


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq=20,
    eval_iter=5,
    checkpoint_path=None,
):
    train_losses = []
    val_losses = []
    tokens_seen = 0
    global_step = 0

    model.to(device)

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(
                    train_loader, model, device, eval_iter
                )
                val_loss = calc_loss_loader(
                    val_loader, model, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"tokens_seen={tokens_seen}"
                )

            global_step += 1

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": model.cfg,
            },
            path,
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen,
    }


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt, max_new_tokens=30):
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    out = generate(
        model,
        encoded,
        max_new_tokens=max_new_tokens,
        context_size=model.cfg["context_length"],
        temperature=0.0,
    )
    return token_ids_to_text(out.cpu(), tokenizer)
