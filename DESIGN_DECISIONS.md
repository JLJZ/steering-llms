# Design Decisions

*Why each architectural and algorithmic choice was made.*

---

## Why PyTorch Hooks Instead of a Steering Library?

We register `register_forward_hook` on specific transformer layers to intercept and
modify residual stream activations at runtime. This is lower-level than libraries
like `transformer_lens` or `baukit` but gives us exact control over:
- Which layer and which sub-component (attention output vs. MLP output)
- Whether to steer the full residual stream or just the delta from a sub-layer
- Applying the decaying coefficient as a function of token position

`transformer_lens` is excellent for interpretability research on large models but
adds overhead and may not support all sub-1B model architectures cleanly.

---

## Why the Logit-Proxy Metric for Sweeps?

```
refusal_composite = (P(Sorry) + P(I) - P(Sure)) / 3
```

Full generation + LLM judge takes ~2s per prompt. A 34-layer × 13-alpha sweep
over 50 prompts = 22,100 evaluations. At 2s each that is 12 hours.

The logit proxy takes ~50ms per prompt (no generation needed), making the full
sweep tractable in ~18 minutes.

The tradeoff: logit proxy does not catch theatrical compliance. We accept this
for sweep stages and always judge-score the finalist configurations.

---

## Why BH-FDR Correction?

A layer sweep across 24–32 layers runs 24–32 hypothesis tests. Without correction,
we expect 1–2 spurious "significant" results by chance alone at p < 0.05.
Benjamini-Hochberg controls false discovery rate without being as conservative
as Bonferroni (which would require p < 0.002 to declare significance at 24 tests).

---

## Why Qwen2.5-0.5B as Primary Model?

1. **Best reasoning/size ratio at sub-1B:** Qwen2.5-0.5B-Instruct achieves ~38%
   on MMLU despite having only 500M parameters, making it a better test of
   capability preservation than models with lower baselines.

2. **Well-documented architecture:** 24 layers, standard transformer, no exotic
   attention patterns that would complicate hook registration.

3. **Instruction-tuned:** The model already has some refusal behaviour, meaning
   the "refusal direction" likely exists in its representations. CAA cannot create
   capability that doesn't exist; it can only amplify existing signal.

4. **Community support:** Active HuggingFace community; if we hit architecture
   edge cases, solutions are findable.

---

## Why Null-Space Constraint as Default (Not CAST)?

CAST requires a well-calibrated harm-condition vector. At sub-1B, this vector
is noisy and has a known failure mode on math/code prompts (AlphaSteer paper).

The null-space constraint is unconditional: it mathematically guarantees the
steering vector has near-zero effect on inputs whose activations lie in the
benign subspace. It doesn't require a separate calibration step.

CAST is implemented as an optional additional layer on top of null-space steering,
not as a replacement for it.

---

## Why Decaying Coefficient by Default?

Small models have ~12–24 layers. If we apply a constant alpha throughout
generation, the steering perturbation compounds across every layer in the
forward pass at every token. By token 20, the accumulated deviation from the
original model is large.

The decay `alpha_t = alpha_0 * exp(-0.4 * t)` means:
- Token 0: full alpha (targeting the critical first token that sets refusal direction)
- Token 5: ~14% alpha
- Token 10: ~2% alpha (effectively zero)

This matches the finding that "faster decaying of the SAE steering vector is
preferred as it provides better control" (SAE-RSV paper, 2025).

---

## Why Matplotlib Instead of a Web Dashboard?

The project is designed to run on a compute cluster without a browser. Matplotlib
produces publication-ready figures that can be saved as PDF/PNG for inclusion in
reports. The visualization module is stateless — pass data, get figure — making
it composable and easy to test.
