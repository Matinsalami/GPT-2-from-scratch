import argparse
from pathlib import Path
import sys

import torch
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt2_scratch.config import SMALL_DEMO_CONFIG
from gpt2_scratch.data import create_dataloader
from gpt2_scratch.model import GPTModel
from gpt2_scratch.training import (
    calc_loss_loader,
    generate_sample,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    text = Path(args.data).read_text(encoding="utf-8")
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    cfg = dict(SMALL_DEMO_CONFIG)

    train_loader = create_dataloader(
        train_text,
        batch_size=args.batch_size,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_text,
        batch_size=args.batch_size,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        shuffle=False,
        drop_last=False,
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError(
            "Dataset is too small for the selected context length. "
            "Use a larger corpus or reduce context_length."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"device={device}")
    print(
        f"initial_train_loss="
        f"{calc_loss_loader(train_loader, model.to(device), device, 2):.4f}"
    )
    print(
        f"initial_val_loss="
        f"{calc_loss_loader(val_loader, model, device, 2):.4f}"
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=10,
        eval_iter=2,
        checkpoint_path=args.checkpoint,
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    print(
        generate_sample(
            model,
            tokenizer,
            device,
            prompt="Every effort moves you",
            max_new_tokens=30,
        )
    )


if __name__ == "__main__":
    main()
