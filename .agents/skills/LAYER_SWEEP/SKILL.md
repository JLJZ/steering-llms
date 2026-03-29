# Agent Skill: Layer Selectivity Sweep

**Purpose:** Identify the refusal-sensitive layer in a model by measuring the
harmful-vs-neutral gap in refusal composite at every layer, with and without steering.

**Read first:** `docs/CONTEXT.md` (especially "Finding 1") and `docs/LIMITATIONS.md`
(especially sections 2, 7).

---

## What This Skill Does

Runs a per-layer logit-probe sweep to find the layer where steering the refusal
signal produces the largest harmful/neutral separation (H-N gap).

This is the prerequisite for all other skills. Never extract a steering vector or
apply steering until this sweep has confirmed a target layer.

---

## Entry Point

```
src/models/layer_sweep.py
```

Primary function:
```python
def run_layer_sweep(
    model,
    tokenizer,
    harmful_prompts: list[str],
    neutral_prompts: list[str],
    steering_vector: torch.Tensor,   # extracted at any mid-layer as pilot
    alpha: float = 0.01,
    device: str = "cuda",
) -> dict[int, float]                # layer_idx -> H-N gap
```

---

## Algorithm

```
For each layer L in range(model.num_layers):
    1. Register a forward hook on layer L's residual stream output
    2. For each harmful prompt p_h:
        a. Run forward pass with hook active (steering_vector * alpha added)
        b. Record refusal_composite = (P(Sorry) + P(I) - P(Sure)) / 3
    3. For each neutral prompt p_n:
        a. Same as above
    4. H_mean = mean(refusal_composite | harmful)
       N_mean = mean(refusal_composite | neutral)
       gap[L] = H_mean - N_mean
    5. Remove hook
Return gap dict
```

---

## Refusal Composite Calculation

```python
def refusal_composite(logits: torch.Tensor, tokenizer) -> float:
    """
    Logit-level proxy for refusal intent at first generated token.
    Positive = model leans toward refusal. Negative = compliance.
    """
    vocab = tokenizer.get_vocab()
    sorry_id = vocab.get("Sorry", vocab.get("▁Sorry", -1))
    i_id     = vocab.get("I",     vocab.get("▁I",     -1))
    sure_id  = vocab.get("Sure",  vocab.get("▁Sure",  -1))

    probs = torch.softmax(logits[0, -1, :], dim=-1)
    p_sorry = probs[sorry_id].item() if sorry_id >= 0 else 0.0
    p_i     = probs[i_id].item()     if i_id >= 0 else 0.0
    p_sure  = probs[sure_id].item()  if sure_id >= 0 else 0.0

    return (p_sorry + p_i - p_sure) / 3.0
```

**IMPORTANT — Tokenizer differences:** Qwen2.5 uses SentencePiece with `▁` prefix.
Always check both `"Sorry"` and `"▁Sorry"`. If neither exists, fall back to
the token IDs of the first token of `" Sorry"` (with leading space) in the model's
tokenizer. Add a startup assertion that at least one refusal token was found.

---

## Hook Registration Pattern

```python
import torch
from typing import Callable

def make_add_hook(vector: torch.Tensor, alpha: float, token_idx: int = -1) -> Callable:
    """
    Returns a forward hook that adds alpha * vector to the residual stream
    at position token_idx (default: last token only).
    """
    def hook(module, input, output):
        # output is typically a tuple; residual stream is output[0]
        hidden = output[0]
        hidden[:, token_idx, :] = hidden[:, token_idx, :] + alpha * vector.to(hidden.device)
        return (hidden,) + output[1:]
    return hook
```

**Architecture notes:**
- Qwen2.5: hook target is `model.model.layers[L]` (the full decoder layer)
- SmolLM2: same pattern
- Always hook the *output* of the full layer (post-attention + post-MLP), not
  just the attention or MLP sub-component, during the sweep phase.
  Sub-component targeting comes after the sweep confirms the target layer.

---

## Output Format

```python
{
    "gaps": {0: 0.002, 1: 0.005, ..., 23: -0.003},  # layer -> H-N gap
    "peak_layer": 9,
    "peak_gap": 0.047,
    "selectivity_slopes": {9: 0.031, 10: 0.028, ...},  # for dose-response
    "anti_selective_layers": [15, 16, 17],              # negative slopes
    "n_harmful": 25,
    "n_neutral": 25,
    "alpha_used": 0.01,
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
}
```

Save to: `results/{model_slug}/layer_sweep.json`

---

## Validation Checks (Run Before Returning)

1. **Refusal tokens found:** Assert at least one of Sorry/I was resolved to a valid token ID.
2. **Non-trivial variance:** Assert `max(gaps.values()) - min(gaps.values()) > 0.005`.
   If variance is < 0.005, the steering vector used for the pilot is likely noise.
3. **Prediction vs. actual:** Log whether the peak layer is near 41% depth.
   Print: `f"Hypothesis: L{int(0.41 * n_layers)} | Empirical: L{peak_layer}"`
4. **Anti-selective count:** Warn if more than 30% of layers are anti-selective.
   This suggests the contrastive pairs need improvement.

---

## Common Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `IndexError: index out of range` in hook | Hook returning wrong tuple shape | Check model architecture; Qwen returns `(hidden_states, past_key_values)` |
| All gaps near zero | Wrong layer type hooked | Ensure hooking full decoder layer, not attention sub-layer |
| `KeyError` on token ID | Tokenizer prefix mismatch | Check both `"Sorry"` and `"▁Sorry"` |
| CUDA OOM | Batch size too large | Run prompts one at a time (batch_size=1) during sweep |
