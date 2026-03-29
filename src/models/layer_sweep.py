"""
src/models/layer_sweep.py

Per-layer selectivity sweep to identify the refusal-sensitive layer.
See agents/LAYER_SWEEP.md for the full specification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.loader import collect_activation, get_layer

logger = logging.getLogger(__name__)


def refusal_composite(logits: torch.Tensor, tokenizer) -> float:
    """
    Logit-level proxy for refusal intent at the last token position.
    Positive = leans toward refusal. Negative = leans toward compliance.

    refusal_composite = (P(Sorry) + P(I) - P(Sure)) / 3
    """
    vocab = tokenizer.get_vocab()

    def _get_id(candidates):
        for c in candidates:
            if c in vocab:
                return vocab[c]
        return None

    sorry_id = _get_id(["▁Sorry", "Sorry"])
    i_id = _get_id(["▁I", "I"])
    sure_id = _get_id(["▁Sure", "Sure"])

    probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
    p_sorry = probs[sorry_id].item() if sorry_id is not None else 0.0
    p_i = probs[i_id].item() if i_id is not None else 0.0
    p_sure = probs[sure_id].item() if sure_id is not None else 0.0

    return (p_sorry + p_i - p_sure) / 3.0


def _get_composite_with_hook(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    vector: torch.Tensor,
    alpha: float,
    device: str,
) -> float:
    """Run forward pass with a steering hook and return refusal_composite."""
    logits_store: dict = {}

    def hook(module, input, output):
        hidden = output[0]
        vec = alpha * vector.to(hidden.device)
        if hidden.dim() == 2:
            hidden[-1, :] = hidden[-1, :] + vec
        else:
            hidden[:, -1, :] = hidden[:, -1, :] + vec
        return (hidden,) + output[1:]

    def logit_hook(module, input, output):
        logits_store["logits"] = output.logits.detach()

    h1 = get_layer(model, layer_idx).register_forward_hook(hook)
    h2 = model.register_forward_hook(logit_hook)
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            model(**inputs)
    finally:
        h1.remove()
        h2.remove()

    return refusal_composite(logits_store["logits"], tokenizer)


def run_layer_sweep(
    model,
    tokenizer,
    harmful_prompts: list[str],
    neutral_prompts: list[str],
    pilot_vector: torch.Tensor,
    n_layers: int,
    alpha: float = 0.01,
    device: str = "cuda",
    results_dir: Optional[str] = None,
) -> dict:
    """
    Sweep all layers. For each layer, compute H-N gap = mean(refusal_composite|harmful)
    - mean(refusal_composite|neutral) under steering with pilot_vector at that layer.

    Parameters
    ----------
    pilot_vector : torch.Tensor
        A preliminary steering vector (shape: hidden_size). Used to probe each layer.
        Can be extracted from any middle layer as a first pass. Will be refined
        after sweep identifies the true target layer.

    Returns
    -------
    dict with keys: gaps, peak_layer, peak_gap, selectivity_slopes, anti_selective_layers
    """
    logger.info(
        f"Starting layer sweep across {n_layers} layers "
        f"({len(harmful_prompts)} harmful, {len(neutral_prompts)} neutral prompts)"
    )

    gaps: dict[int, float] = {}

    for layer_idx in range(n_layers):
        h_composites = []
        n_composites = []

        for prompt in harmful_prompts:
            try:
                c = _get_composite_with_hook(
                    model, tokenizer, prompt, layer_idx, pilot_vector, alpha, device
                )
                h_composites.append(c)
            except Exception as e:
                logger.warning(f"Layer {layer_idx}, harmful prompt failed: {e}")

        for prompt in neutral_prompts:
            try:
                c = _get_composite_with_hook(
                    model, tokenizer, prompt, layer_idx, pilot_vector, alpha, device
                )
                n_composites.append(c)
            except Exception as e:
                logger.warning(f"Layer {layer_idx}, neutral prompt failed: {e}")

        if h_composites and n_composites:
            gaps[layer_idx] = np.mean(h_composites) - np.mean(n_composites)
        else:
            gaps[layer_idx] = 0.0

        logger.debug(f"  L{layer_idx:02d}: H-N gap = {gaps[layer_idx]:.5f}")

    peak_layer = max(gaps, key=lambda l: gaps[l])
    peak_gap = gaps[peak_layer]
    anti_selective = [l for l, g in gaps.items() if g < 0]

    # Validate results
    gap_range = max(gaps.values()) - min(gaps.values())
    if gap_range < 0.005:
        logger.warning(
            f"Gap range {gap_range:.5f} < 0.005. Pilot vector may be noise. "
            "Consider extracting a better contrastive vector first."
        )

    hypothesis_layer = int(0.41 * n_layers)
    logger.info(
        f"Sweep complete | Peak: L{peak_layer} (gap={peak_gap:.5f}) | "
        f"Hypothesis: L{hypothesis_layer} | "
        f"Anti-selective layers: {anti_selective}"
    )

    result = {
        "gaps": gaps,
        "peak_layer": peak_layer,
        "peak_gap": peak_gap,
        "hypothesis_layer": hypothesis_layer,
        "hypothesis_matched": abs(peak_layer - hypothesis_layer) <= 2,
        "anti_selective_layers": anti_selective,
        "anti_selective_fraction": len(anti_selective) / n_layers,
        "n_harmful": len(harmful_prompts),
        "n_neutral": len(neutral_prompts),
        "alpha_used": alpha,
        "gap_range": gap_range,
    }

    if len(anti_selective) / n_layers > 0.30:
        logger.warning(
            "More than 30% of layers are anti-selective. "
            "Improve contrastive pairs before proceeding."
        )

    if results_dir:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "layer_sweep.json", "w") as f:
            # gaps keys must be strings for JSON
            json_result = {**result, "gaps": {str(k): v for k, v in gaps.items()}}
            json.dump(json_result, f, indent=2)
        logger.info(f"Sweep results saved to {out / 'layer_sweep.json'}")

    return result
