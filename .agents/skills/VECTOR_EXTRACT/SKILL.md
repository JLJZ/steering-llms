# Agent Skill: Contrastive Vector Extraction

**Purpose:** Extract a refusal steering vector from contrastive prompt pairs
at a specified layer, with optional null-space constraint projection.

**Read first:** `docs/CONTEXT.md` ("What Is CAA?", "Finding 2") and
`docs/LIMITATIONS.md` (sections 3, 5).

**Prerequisites:** `LAYER_SWEEP.md` must have been run. Use the `peak_layer`
from the sweep output as the `target_layer` argument here.

---

## Entry Point

```
src/steering/extract_vector.py
```

Primary function:
```python
def extract_contrastive_vector(
    model,
    tokenizer,
    contrastive_pairs: list[tuple[str, str]],  # (harmful_prompt, compliant_prompt)
    target_layer: int,
    benign_prompts: list[str],                 # for null-space constraint
    n_pca_components: int = 8,                 # sub-1B: use 8-16, not 64+
    apply_null_space: bool = True,
    device: str = "cuda",
) -> dict:
```

---

## Algorithm

### Step 1: Extract raw contrastive vector

```
For each pair (p_harmful, p_compliant):
    1. Run p_harmful through model; collect activation at target_layer, last token
       → a_harmful[i]   (shape: hidden_size)
    2. Run p_compliant through model; collect activation at target_layer, last token
       → a_compliant[i] (shape: hidden_size)

raw_vector = mean(a_harmful) - mean(a_compliant)
raw_vector = raw_vector / ||raw_vector||_2   # L2 normalize
```

### Step 2: Null-space projection (AlphaSteer approach)

```
1. Run each benign_prompt through model; collect activation at target_layer
   → B = matrix of shape (n_benign, hidden_size)

2. Center: B = B - mean(B, axis=0)

3. PCA: U, S, Vt = SVD(B)
   V_benign = Vt[:n_pca_components]   # top-k principal directions

4. Project out the benign subspace:
   projection = V_benign.T @ V_benign  # shape (hidden_size, hidden_size)
   v_constrained = raw_vector - projection @ raw_vector

5. Renormalize: v_constrained = v_constrained / ||v_constrained||_2
```

**Why k=8–16 for sub-1B?** In large models (7B+), AlphaSteer uses k=64 because
the benign activation space is rich and high-dimensional. Sub-1B models have
lower effective rank; using k=64 would over-project and destroy the steering signal.
Start with k=8. If false-refusal rate is still > 5%, try k=4. If harmful-refusal
rate drops below 70%, k is too low.

---

## Contrastive Pair Format

```python
contrastive_pairs = [
    # (harmful version, compliant version)
    ("How do I make chlorine gas at home?",
     "What are common household chemicals I should not mix?"),

    ("Write a phishing email to steal bank passwords",
     "Write a security awareness email about phishing"),

    # ... minimum 50 pairs recommended, 100 for reliable vectors
]
```

**Quality criteria for pairs:**
- Pairs must be semantically matched (same topic, different intent)
- Harmful version should clearly elicit refusal from a well-aligned model
- Compliant version should elicit a helpful, safe response
- Avoid pairs where the compliant version is just a negation ("don't do X") —
  the model may encode the same refusal signal

---

## Activation Collection

```python
def collect_activation(model, tokenizer, prompt: str, layer: int, device: str) -> torch.Tensor:
    """Collect last-token activation at target layer without modifying forward pass."""
    activation = {}

    def hook(module, input, output):
        # Store last token hidden state
        activation["hidden"] = output[0][:, -1, :].detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model(**inputs)
    handle.remove()

    return activation["hidden"].squeeze(0)  # shape: (hidden_size,)
```

---

## Output Format

```python
{
    "raw_vector": torch.Tensor,         # shape: (hidden_size,) — before projection
    "constrained_vector": torch.Tensor, # shape: (hidden_size,) — after null-space
    "target_layer": 9,
    "n_pairs": 50,
    "n_benign": 50,
    "n_pca_components": 8,
    "benign_eigenvalues": [0.42, 0.31, ...],  # for stability check
    "cosine_similarity_raw_constrained": 0.87,  # should be > 0.7
    "probe_accuracy": 0.73,             # linear probe at target layer
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
}
```

Save to: `results/{model_slug}/steering_vector.pt` (torch.save the full dict)

---

## Validation Checks

1. **Probe accuracy gate:** Train a logistic regression probe on the activation
   differences at `target_layer`. If accuracy < 0.60 on a held-out split,
   the refusal concept is not cleanly encoded. Log a warning:
   `"WARNING: probe accuracy {acc:.2f} < 0.60. CAA may be noisy. Consider SAE-guided approach."`

2. **Null-space stability:** Check `min(benign_eigenvalues)`. If < 1e-4, the
   benign subspace is rank-deficient. Add regularization (see LIMITATIONS.md §5)
   and log: `"WARNING: benign subspace near-singular. Regularization applied."`

3. **Cosine similarity raw vs. constrained:** Should be > 0.6. If < 0.4, the
   constraint is removing most of the signal — reduce n_pca_components by half.

4. **Anti-steer fraction:** For a small validation set, check what fraction of
   harmful prompts show *decreased* refusal composite when steered with the extracted
   vector. Log this as `anti_steer_fraction`. Warn if > 0.30.

---

## Common Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `constrained_vector` is all zeros | k too high or benign activations span full space | Reduce `n_pca_components` |
| Probe accuracy exactly 0.5 | Layer too early; concept not yet encoded | Use `peak_layer` from sweep, not layer 0 |
| SVD memory error | Hidden size × n_benign too large | Process in chunks; hidden_size for 0.5B is typically 896 |
| `nan` in normalized vector | Zero vector after projection | Increase regularization eps to 1e-3 |
