import argparse
from pathlib import Path
import sys

import torch
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt2_scratch.generation import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)
from gpt2_scratch.model import GPTModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = GPTModel(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    idx = text_to_token_ids(args.prompt, tokenizer).to(device)

    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        context_size=checkpoint["config"]["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(token_ids_to_text(out.cpu(), tokenizer))


if __name__ == "__main__":
    main()
