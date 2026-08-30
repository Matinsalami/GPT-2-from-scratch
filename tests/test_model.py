import pytest
import torch

from gpt2_scratch.generation import generate
from gpt2_scratch.model import GPTModel, LayerNorm, MultiHeadAttention


TEST_CONFIG = {
    "vocab_size": 100,
    "context_length": 8,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def test_gpt_output_shape():
    model = GPTModel(TEST_CONFIG)
    x = torch.randint(0, TEST_CONFIG["vocab_size"], (2, 5))
    logits = model(x)
    assert logits.shape == (2, 5, TEST_CONFIG["vocab_size"])


def test_layer_norm_statistics():
    torch.manual_seed(123)
    x = torch.randn(4, 7, 32)
    layer = LayerNorm(32)
    y = layer(x)

    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_context_length_guard():
    model = GPTModel(TEST_CONFIG)
    x = torch.randint(
        0,
        TEST_CONFIG["vocab_size"],
        (1, TEST_CONFIG["context_length"] + 1),
    )

    with pytest.raises(ValueError):
        model(x)


def test_causal_attention_future_tokens_do_not_affect_past_outputs():
    torch.manual_seed(123)

    attn = MultiHeadAttention(
        d_in=16,
        d_out=16,
        context_length=6,
        dropout=0.0,
        num_heads=4,
    )
    attn.eval()

    x1 = torch.randn(1, 6, 16)
    x2 = x1.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :]) * 100

    y1 = attn(x1)
    y2 = attn(x2)

    assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-5)


def test_generation_adds_requested_tokens():
    torch.manual_seed(123)
    model = GPTModel(TEST_CONFIG)
    prompt = torch.randint(0, TEST_CONFIG["vocab_size"], (1, 3))

    out = generate(
        model,
        prompt,
        max_new_tokens=4,
        context_size=TEST_CONFIG["context_length"],
        temperature=0.0,
    )

    assert out.shape == (1, 7)
