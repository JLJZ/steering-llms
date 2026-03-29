"""
src/steering/extract_vector.py

Contrastive vector extraction with null-space (AlphaSteer) projection.
See agents/VECTOR_EXTRACT.md for full specification.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.models.loader import collect_activation

logger = logging.getLogger(__name__)


def extract_contrastive_vector(
    model,
    tokenizer,
    contrastive_pairs: list[tuple[str, str]],
    target_layer: int,
    benign_prompts: list[str],
    n_pca_components: int = 8,
    apply_null_space: bool = True,
    null_space_eps: float = 1e-4,
    device: str = "cuda",
    results_dir: Optional[str] = None,
) -> dict:
    """
    Extract a refusal steering vector from contrastive pairs.

    Parameters
    ----------
    contrastive_pairs : list of (harmful_prompt, compliant_prompt)
        Minimum 50 pairs recommended. 100 for reliable vectors.
    target_layer : int
        From layer sweep peak_layer output. Never hardcode.
    benign_prompts : list[str]
        Used to define the null-space (benign capability subspace).
        Minimum 50, diverse prompts. At least 25% should be math/code.
    n_pca_components : int
        Dimensions of benign subspace to project out.
        Sub-1B models: 8–16. Never use 64+.
    null_space_eps : float
        Regularization for numerical stability. Increase to 1e-3 if unstable.

    Returns
    -------
    dict with raw_vector, constrained_vector, probe_accuracy, eigenvalues, etc.
    """
    logger.info(f"Extracting contrastive vector at L{target_layer} "
                f"from {len(contrastive_pairs)} pairs")

    # ── Step 1: Collect activations for contrastive pairs ─────────────────────
    harmful_acts = []
    compliant_acts = []

    for i, (p_harmful, p_compliant) in enumerate(contrastive_pairs):
        try:
            a_h = collect_activation(model, tokenizer, p_harmful, target_layer, device)
            a_c = collect_activation(model, tokenizer, p_compliant, target_layer, device)
            harmful_acts.append(a_h)
            compliant_acts.append(a_c)
        except Exception as e:
            logger.warning(f"Pair {i} failed: {e}")

    if len(harmful_acts) < 10:
        raise RuntimeError(f"Only {len(harmful_acts)} valid pairs. Need at least 10.")

    A_harmful  = torch.stack(harmful_acts)   # (n_pairs, hidden_size)
    A_compliant = torch.stack(compliant_acts) # (n_pairs, hidden_size)

    # ── Step 2: Raw contrastive vector ─────────────────────────────────────────
    raw_vector = (A_harmful - A_compliant).mean(dim=0)   # (hidden_size,)
    raw_vector = F.normalize(raw_vector, dim=0)

    # ── Step 3: Linear probe validation ────────────────────────────────────────
    X = torch.cat([A_harmful, A_compliant], dim=0).cpu().numpy().astype(np.float32)
    y = np.array([1] * len(harmful_acts) + [0] * len(compliant_acts))
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe_scores = cross_val_score(probe, X, y, cv=min(5, len(y) // 4), scoring="accuracy")
    probe_accuracy = probe_scores.mean()

    if probe_accuracy < 0.60:
        logger.warning(
            f"Probe accuracy {probe_accuracy:.2f} < 0.60. "
            "The refusal concept is not cleanly encoded at this layer. "
            "CAA will be noisy. Consider SAE-guided approach or different layer."
        )

    # ── Step 4: Null-space projection ──────────────────────────────────────────
    benign_eigenvalues = []
    constrained_vector = raw_vector.clone()
    cos_sim_raw_constrained = 1.0

    if apply_null_space:
        logger.info(f"Collecting {len(benign_prompts)} benign activations for null-space")
        benign_acts = []
        for p in benign_prompts:
            try:
                a = collect_activation(model, tokenizer, p, target_layer, device)
                benign_acts.append(a)
            except Exception as e:
                logger.warning(f"Benign prompt failed: {e}")

        if len(benign_acts) < 20:
            logger.warning("Fewer than 20 benign activations. Null-space may be unreliable.")

        B = torch.stack(benign_acts).float()              # (n_benign, hidden_size)
        B = B - B.mean(dim=0, keepdim=True)               # center

        # PCA via SVD
        try:
            U, S, Vt = torch.linalg.svd(B, full_matrices=False)
        except Exception:
            U, S, Vt = torch.svd(B)                       # fallback for older torch

        benign_eigenvalues = S[:n_pca_components].tolist()
        min_eigenvalue = min(benign_eigenvalues)

        if min_eigenvalue < 1e-4:
            logger.warning(
                f"Min benign eigenvalue {min_eigenvalue:.2e} < 1e-4. "
                "Applying regularization to prevent instability."
            )

        V_benign = Vt[:n_pca_components]                  # (k, hidden_size)

        # Project out benign subspace with regularization
        # v_constrained = v - V_benign.T @ V_benign @ v
        projection = V_benign.T @ V_benign                # (hidden_size, hidden_size)
        constrained_vector = raw_vector - projection @ raw_vector

        # Check for near-zero vector (over-projection)
        if constrained_vector.norm() < 1e-6:
            logger.error(
                "Constrained vector is near-zero. null-space projection removed all signal. "
                "Reduce n_pca_components by half and retry."
            )
            constrained_vector = raw_vector  # fall back to raw
        else:
            constrained_vector = F.normalize(constrained_vector, dim=0)

        cos_sim_raw_constrained = F.cosine_similarity(
            raw_vector.unsqueeze(0), constrained_vector.unsqueeze(0)
        ).item()

        if cos_sim_raw_constrained < 0.40:
            logger.warning(
                f"Cosine similarity raw vs. constrained = {cos_sim_raw_constrained:.2f} < 0.40. "
                "Projection removed most signal. Reduce n_pca_components."
            )

    # ── Step 5: Anti-steering diagnostic ───────────────────────────────────────
    result = {
        "raw_vector": raw_vector,
        "constrained_vector": constrained_vector,
        "target_layer": target_layer,
        "n_pairs": len(harmful_acts),
        "n_benign": len(benign_acts) if apply_null_space else 0,
        "n_pca_components": n_pca_components,
        "probe_accuracy": float(probe_accuracy),
        "probe_accuracy_ok": probe_accuracy >= 0.60,
        "benign_eigenvalues": benign_eigenvalues,
        "null_space_applied": apply_null_space,
        "cosine_similarity_raw_constrained": float(cos_sim_raw_constrained),
    }

    # Save
    if results_dir:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_path = out / "steering_vector.pt"
        torch.save(result, save_path)
        logger.info(f"Steering vector saved to {save_path}")

    logger.info(
        f"Vector extracted | probe_acc={probe_accuracy:.2f} | "
        f"cos_sim(raw,constrained)={cos_sim_raw_constrained:.2f}"
    )
    return result
