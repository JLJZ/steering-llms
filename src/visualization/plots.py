"""
src/visualization/plots.py

Matplotlib plotting functions for steering project results.
See agents/VISUALIZATION.md for full specification.
All functions are stateless: pass data, receive Figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "baseline": "#7F7F7F",
    "targeted":  "#2196F3",
    "narrow":    "#FF9800",
    "dense":     "#F44336",
    "harmful":   "#D32F2F",
    "neutral":   "#388E3C",
    "positive":  "#1976D2",
    "negative":  "#C62828",
    "zero_line": "#9E9E9E",
}

VARIANT_COLOR_MAP = {
    "BASELINE":   COLORS["baseline"],
    "TARGETED":   COLORS["targeted"],
    "NARROW":     COLORS["narrow"],
    "DENSE":      COLORS["dense"],
}


def setup_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })


def _variant_color(name: str) -> str:
    for key, color in VARIANT_COLOR_MAP.items():
        if key in name.upper():
            return color
    return COLORS["baseline"]


# ── Plot 1: Layer Selectivity Sweep ───────────────────────────────────────────

def plot_layer_sweep(
    gaps: dict,                # {layer_idx (int or str): float}
    n_layers: int,
    peak_layer: int,
    hypothesis_layer: int,
    model_name: str,
    alpha_used: float,
) -> plt.Figure:
    setup_style()
    layers = list(range(n_layers))
    gap_values = [gaps.get(i, gaps.get(str(i), 0.0)) for i in layers]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [COLORS["positive"] if g >= 0 else COLORS["negative"] for g in gap_values]
    bars = ax.bar(layers, gap_values, color=colors, alpha=0.8, edgecolor="none", width=0.75)

    ax.axhline(0, color=COLORS["zero_line"], linewidth=1.2, zorder=0)
    ax.axvline(peak_layer, color=COLORS["targeted"], linewidth=2, linestyle="--",
               label=f"Empirical peak: L{peak_layer}")
    ax.axvline(hypothesis_layer, color=COLORS["baseline"], linewidth=1.5, linestyle=":",
               label=f"Hypothesis (~41%): L{hypothesis_layer}")

    # Annotate peak
    peak_gap = gaps.get(peak_layer, gaps.get(str(peak_layer), 0.0))
    ax.annotate(
        f"L{peak_layer}\ngap={peak_gap:.4f}",
        xy=(peak_layer, peak_gap),
        xytext=(peak_layer + 1.5, peak_gap * 1.2 + 0.002),
        fontsize=9,
        color=COLORS["targeted"],
        arrowprops=dict(arrowstyle="->", color=COLORS["targeted"], lw=1.2),
    )

    ax.set_xlabel("Layer index")
    ax.set_ylabel("H–N gap (refusal composite)")
    ax.set_title(f"Layer Selectivity Sweep — {model_name}  (α={alpha_used})")
    ax.set_xticks(layers)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return fig


# ── Plot 2: Dose-Response Curves ──────────────────────────────────────────────

def plot_dose_response(
    dose_response_by_variant: dict[str, dict],
    model_name: str,
) -> plt.Figure:
    """
    dose_response_by_variant: {variant_name: {alpha: {harmful_refusal_rate, false_refusal_rate}}}
    """
    setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax_harm, ax_false, ax_gap = axes

    for variant, data in dose_response_by_variant.items():
        alphas  = sorted(data.keys())
        h_rates = [data[a]["harmful_refusal_rate"] for a in alphas]
        f_rates = [data[a]["false_refusal_rate"]   for a in alphas]
        color   = _variant_color(variant)

        ax_harm.plot(alphas, h_rates, marker="o", color=color, label=variant)
        ax_false.plot(alphas, f_rates, marker="s", color=color, label=variant)

    # Targets
    ax_harm.axhline(0.80, color="black", linewidth=1, linestyle="--", alpha=0.5,
                    label="Target ≥ 0.80")
    ax_false.axhspan(0.05, 1.0, alpha=0.08, color="red", label="False refusal > 5%")
    ax_false.axhline(0.05, color="red", linewidth=1, linestyle="--", alpha=0.6)

    ax_harm.axvline(0, color=COLORS["zero_line"], linewidth=1, linestyle=":")
    ax_false.axvline(0, color=COLORS["zero_line"], linewidth=1, linestyle=":")

    ax_harm.set_ylabel("Harmful refusal rate")
    ax_harm.set_title(f"Dose-Response Curves — {model_name}")
    ax_harm.legend(loc="lower right", framealpha=0.9)
    ax_harm.set_ylim(-0.05, 1.05)

    ax_false.set_ylabel("False refusal rate (neutral)")
    ax_false.legend(loc="upper left", framealpha=0.9)
    ax_false.set_ylim(-0.05, 1.05)

    ax_gap.set_xlabel("Steering strength (α)")
    ax_gap.set_ylabel("H–N selectivity gap")
    ax_gap.axhline(0, color=COLORS["negative"], linewidth=1.5, linestyle="-", alpha=0.5)
    ax_gap.axhspan(-1.0, 0, alpha=0.06, color="red", label="Anti-selective zone")
    ax_gap.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return fig


# ── Plot 3: Cross-Effect Heatmap ──────────────────────────────────────────────

def plot_cross_effect_matrix(
    matrix: np.ndarray,
    steer_axes: list[str],
    eval_axes: list[str],
    title: str = "Cross-Effect Matrix",
) -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=text_color, fontsize=9, fontweight=weight)
            # Bold border on diagonal
            if i == j:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      fill=False, edgecolor="black", linewidth=2)
                ax.add_patch(rect)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Score change per unit α", fontsize=10)

    ax.set_xticks(range(len(eval_axes)))
    ax.set_yticks(range(len(steer_axes)))
    ax.set_xticklabels(eval_axes, rotation=45, ha="right")
    ax.set_yticklabels(steer_axes)
    ax.set_xlabel("Eval axis →")
    ax.set_ylabel("← Steer axis")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ── Plot 4: Evaluation Summary (2×2) ─────────────────────────────────────────

def plot_evaluation_summary(results: list[dict], model_name: str) -> plt.Figure:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    (ax_harm, ax_false), (ax_ground, ax_ppl) = axes

    names  = [r["variant"] for r in results]
    colors = [_variant_color(n) for n in names]
    y_pos  = np.arange(len(names))

    def _hbar(ax, values, errs=None, xlabel="", title="", threshold=None,
              threshold_label="", threshold_color="red", threshold_side="above"):
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor="white")
        if errs:
            lo = [v - e[0] for v, e in zip(values, errs)]
            hi = [e[1] - v for v, e in zip(values, errs)]
            ax.errorbar(values, y_pos, xerr=[lo, hi], fmt="none",
                        color="black", capsize=3, linewidth=1.2)
        # Annotate bars
        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va="center", fontsize=9)
        if threshold is not None:
            ax.axvline(threshold, color=threshold_color, linewidth=1.5,
                       linestyle="--", label=threshold_label, alpha=0.8)
            if threshold_side == "above":
                ax.axvspan(threshold, ax.get_xlim()[1] if ax.get_xlim()[1] > threshold
                           else threshold + 0.2, alpha=0.08, color=threshold_color)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(fontsize=9)

    h_vals = [r["harmful_refusal_rate"] for r in results]
    h_cis  = [r.get("harmful_refusal_ci", (r["harmful_refusal_rate"], r["harmful_refusal_rate"]))
               for r in results]
    _hbar(ax_harm, h_vals, h_cis, "Rate", "Harmful Refusal Rate",
          threshold=0.80, threshold_label="Target ≥ 0.80", threshold_color="green",
          threshold_side="below")

    f_vals = [r["false_refusal_rate"] for r in results]
    _hbar(ax_false, f_vals, None, "Rate", "False Refusal Rate (Neutral Prompts)",
          threshold=0.05, threshold_label="Threshold = 0.05", threshold_color="red")

    g_vals = [r.get("grounding_delta") or 0.0 for r in results]
    g_colors = [COLORS["positive"] if v >= -0.2 else COLORS["negative"] for v in g_vals]
    ax_ground.barh(y_pos, g_vals, color=g_colors, alpha=0.85, edgecolor="white")
    ax_ground.axvline(0, color="black", linewidth=1)
    ax_ground.axvline(-0.5, color="red", linewidth=1, linestyle="--", label="Heuristic threshold")
    ax_ground.set_yticks(y_pos); ax_ground.set_yticklabels(names)
    ax_ground.set_xlabel("Δ Grounding from baseline (1–4 scale)")
    ax_ground.set_title("Grounding Degradation")
    ax_ground.legend(fontsize=9)

    ppl_vals = [r.get("perplexity_ratio", 1.0) for r in results]
    p_colors = [COLORS["positive"] if v < 1.5 else COLORS["negative"] for v in ppl_vals]
    ax_ppl.barh(y_pos, ppl_vals, color=p_colors, alpha=0.85, edgecolor="white")
    ax_ppl.axvline(1.0, color="black", linewidth=1, label="Baseline (1.0)")
    ax_ppl.axvline(1.5, color="red", linewidth=1.5, linestyle="--", label="Collapse threshold")
    ax_ppl.set_yticks(y_pos); ax_ppl.set_yticklabels(names)
    ax_ppl.set_xlabel("Perplexity ratio (steered / baseline)")
    ax_ppl.set_title("Capability: Perplexity Ratio")
    ax_ppl.legend(fontsize=9)

    fig.suptitle(f"Evaluation Summary — {model_name}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── Plot 5: Decay Coefficient Analysis ────────────────────────────────────────

def plot_decay_analysis(
    alpha_0: float,
    lambdas: list[float] = None,
    n_tokens: int = 30,
) -> plt.Figure:
    if lambdas is None:
        lambdas = [0.2, 0.4, 0.6, 1.0]
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    tokens = np.arange(n_tokens)

    for lam in lambdas:
        alphas = alpha_0 * np.exp(-lam * tokens)
        ax.plot(tokens, alphas, marker=None, label=f"λ={lam}")

    ax.axvspan(0, 5, alpha=0.10, color=COLORS["targeted"], label="Critical window (t=0–5)")
    ax.axhline(0.001, color=COLORS["zero_line"], linewidth=1, linestyle=":", label="α≈0 threshold")
    ax.set_xlabel("Token position")
    ax.set_ylabel(f"Effective α  (α₀={alpha_0})")
    ax.set_title("Decaying Steering Coefficient Over Generation")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return fig


# ── Save Helper ────────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, name: str, results_dir: str,
                formats: tuple = ("png", "pdf")):
    out = Path(results_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out / f"{name}.{fmt}", bbox_inches="tight", dpi=150)
    plt.close(fig)
