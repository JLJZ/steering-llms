"""
src/models/loader.py

Model and tokenizer loading with architecture detection.
Supports Qwen2.5, SmolLM2, and Phi-3 families.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "n_layers": 24,
        "hidden_size": 896,
        "depth_41pct": 9,
    },
    "Qwen/Qwen2.5-1B-Instruct": {
        "n_layers": 28,
        "hidden_size": 1536,
        "depth_41pct": 11,
    },
    "HuggingFaceTB/SmolLM2-360M-Instruct": {
        "n_layers": 32,
        "hidden_size": 960,
        "depth_41pct": 13,
    },
}


@dataclass
class ModelConfig:
    model_id: str
    n_layers: int
    hidden_size: int
    hypothesis_layer: int  # ~41% depth — predicted refusal-sensitive layer
    layer_module_path: str  # e.g. "model.layers" for Qwen2.5


def load_model(
    model_id: str,
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    Load model and tokenizer. Returns (model, tokenizer, config).

    Notes
    -----
    - Always loads in bfloat16 to fit sub-1B models on single GPU/CPU
    - Sets model to eval mode and disables gradient computation
    - Detects architecture to set correct layer path for hook registration
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {model_id} on {device} ({torch_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Detect architecture
    if model_id in SUPPORTED_MODELS:
        info = SUPPORTED_MODELS[model_id]
        config = ModelConfig(
            model_id=model_id,
            n_layers=info["n_layers"],
            hidden_size=info["hidden_size"],
            hypothesis_layer=info["depth_41pct"],
            layer_module_path="model.layers",
        )
    else:
        # Auto-detect from model config
        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        config = ModelConfig(
            model_id=model_id,
            n_layers=n_layers,
            hidden_size=hidden_size,
            hypothesis_layer=int(0.41 * n_layers),
            layer_module_path="model.layers",
        )
        logger.warning(
            f"Unknown model {model_id}. Auto-detected: {n_layers} layers, "
            f"hidden_size={hidden_size}. Always verify with empirical sweep."
        )

    # Validate tokenizer has refusal tokens
    _validate_refusal_tokens(tokenizer, model_id)

    logger.info(
        f"Loaded {model_id}: {config.n_layers} layers, "
        f"hypothesis target layer: L{config.hypothesis_layer}"
    )
    return model, tokenizer, config


def get_layer(model, layer_idx: int):
    """Return the transformer layer module at layer_idx."""
    return model.model.layers[layer_idx]


def _validate_refusal_tokens(tokenizer, model_id: str):
    """Assert at least one refusal-proxy token exists in vocabulary."""
    vocab = tokenizer.get_vocab()
    candidates = ["Sorry", "▁Sorry", "I", "▁I", "Sure", "▁Sure"]
    found = [t for t in candidates if t in vocab]
    if len(found) < 2:
        raise RuntimeError(
            f"Could not find refusal proxy tokens in {model_id} tokenizer. "
            f"Found: {found}. Check tokenizer vocabulary."
        )
    logger.debug(f"Refusal tokens found: {found}")


def collect_activation(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    device: str = "cuda",
    token_position: int = -1,
) -> torch.Tensor:
    """
    Run a single forward pass and collect the hidden state at layer_idx.

    Returns
    -------
    torch.Tensor of shape (hidden_size,), float32
    """
    activation: dict = {}

    def hook(module, input, output):
        hidden = output[0]  # (batch, seq_len, hidden_size) or (seq_len, hidden_size)
        if hidden.dim() == 2:
            activation["hidden"] = hidden[token_position, :].detach().float()
        else:
            activation["hidden"] = hidden[:, token_position, :].detach().float()

    handle = get_layer(model, layer_idx).register_forward_hook(hook)
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            model(**inputs)
    finally:
        handle.remove()

    return activation["hidden"].squeeze(0)  # (hidden_size,)
