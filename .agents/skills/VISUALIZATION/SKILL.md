# Agent Skill: Visualization

**Purpose:** Generate all standard plots for the steering project using Matplotlib.
All functions are stateless: pass data, get a `Figure` object back. Save externally.

**Read first:** `docs/DESIGN_DECISIONS.md` ("Why Matplotlib").

---

## Entry Point

```
src/visualization/plots.py
```

---

## Style Setup (Call Once Per Session)

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def setup_style():
    """Apply consistent research-quality style across all plots."""
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

# Color palette (colorblind-safe)
COLORS = {
    "baseline":  "#7F7F7F",   # grey
    "targeted":  "#2196F3",   # blue — primary / hero variant
    "narrow":    "#FF9800",   # orange
    "dense":     "#F44336",   # red — expected to fail
    "harmful":   "#D32F2F",
    "neutral":   "#388E3C",
    "positive":  "#1976D2",
    "negative":  "#C62828",
    "zero_line": "#9E9E9E",
}
```

---

## Plot 1: Layer Selectivity Sweep

```python
def plot_layer_sweep(
    gaps: dict[int, float],           # layer -> H-N gap
    n_layers: int,
    peak_layer: int,
    hypothesis_layer: int,            # int(0.41 * n_layers)
    model_name: str,
    alpha_used: float,
) -> plt.Figure:
```

**Layout:**
- x-axis: layer index (0 to n_layers-1)
- y-axis: H-N gap (refusal composite)
- Bar chart: bars colored blue for positive (selective), red for negative (anti-selective)
- Vertical dashed line: `peak_layer` (blue, "Empirical peak: L{peak_layer}")
- Vertical dotted line: `hypothesis_layer` (grey, "Hypothesis (41%): L{hypothesis_layer}")
- Horizontal line at y=0
- Annotation box: "Peak selectivity: L{peak_layer}, gap={gaps[peak_layer]:.4f}"
- Title: `f"Layer Selectivity Sweep — {model_name} (α={alpha_used})"`

---

## Plot 2: Dose-Response Curves

```python
def plot_dose_response(
    dose_response_data: dict,   # alpha -> {harmful_refusal_rate, false_refusal_rate, h_n_gap}
    variants: list[str],        # e.g. ["TARGETED_L9", "NARROW_L8_L10", "DENSE_L6_L12"]
    model_name: str,
) -> plt.Figure:
```

**Layout:** 3 subplots, stacked vertically, shared x-axis

- **Top:** Harmful refusal rate vs. alpha — target ≥ 0.80 marked with horizontal dashed line
- **Middle:** False refusal rate vs. alpha — target ≤ 0.05 marked with red zone (shaded above 0.05)
- **Bottom:** H-N selectivity gap vs. alpha — zero line emphasized; anti-selective region shaded red

Each variant gets its own line colored per `COLORS` dict. Add markers at each alpha point.
Add a vertical dashed line at alpha=0 (grey, "baseline").

---

## Plot 3: Cross-Effect Heatmap

```python
def plot_cross_effect_matrix(
    matrix: np.ndarray,          # shape (n_steer_axes, n_eval_axes)
    steer_axes: list[str],
    eval_axes: list[str],
    title: str = "Cross-Effect Matrix",
) -> plt.Figure:
```

**Layout:**
- Heatmap with diverging colormap (`RdBu_r`, center at 0)
- Cell annotations: rounded values, white text for dark cells
- Diagonal cells (self-effects): bold border
- Colorbar label: "Score change per unit alpha"
- Row/column labels at 45° for readability

```python
cmap = plt.cm.RdBu_r
norm = mpl.colors.TwoSlopeNorm(vmin=matrix.min(), vcenter=0, vmax=matrix.max())
```

---

## Plot 4: Evaluation Summary Bar Chart

```python
def plot_evaluation_summary(
    results: list[dict],         # list of evaluate_variant outputs
    model_name: str,
) -> plt.Figure:
```

**Layout:** 4 subplots in a 2×2 grid

- Top-left: Harmful refusal rate per variant (horizontal bars, error bars for 95% CI)
- Top-right: False refusal rate per variant (horizontal bars, red fill > 0.05)
- Bottom-left: Grounding delta per variant (bar, blue = good, red = bad)
- Bottom-right: Perplexity ratio per variant (bar, red fill > 1.5)

Color each bar by variant using `COLORS` dict. Add value annotations on bars.
Add target threshold lines (dashed).

---

## Plot 5: Anti-Steering Diagnostic

```python
def plot_anti_steering(
    per_prompt_deltas: dict[str, list[float]],  # variant -> list of (steered - baseline)
    variant_names: list[str],
) -> plt.Figure:
```

**Layout:**
- Violin plot or KDE plot of delta distribution per variant
- x-axis: steering delta (steered refusal_composite - baseline)
- Vertical line at x=0: "Zero = no effect"
- Shaded region x < 0: "Anti-steered (harmful)" with red background
- Annotation: f"Anti-steer: {fraction*100:.0f}% of inputs" per variant

---

## Plot 6: Decay Coefficient Analysis

```python
def plot_decay_analysis(
    token_positions: list[int],
    alpha_0: float,
    lambdas: list[float] = [0.2, 0.4, 0.6, 1.0],
) -> plt.Figure:
```

**Layout:**
- x-axis: token position (0 to 30)
- y-axis: effective alpha
- One line per lambda value
- Shaded region: tokens 0–5 ("Critical refusal setup window")
- Annotation: point where alpha < 0.001 for each lambda

---

## Saving Figures

```python
def save_figure(fig: plt.Figure, name: str, results_dir: str, formats=("png", "pdf")):
    """Save a figure in multiple formats."""
    from pathlib import Path
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out / f"{name}.{fmt}", bbox_inches="tight", dpi=150)
    plt.close(fig)
```

All figures saved to: `results/{model_slug}/figures/`

---

## Standard Figure Sizes

| Plot | figsize |
|------|---------|
| Layer sweep | (12, 4) |
| Dose-response (3 panels) | (10, 10) |
| Cross-effect heatmap | (8, 6) |
| Evaluation summary (2×2) | (12, 10) |
| Anti-steering violin | (10, 5) |
| Decay analysis | (8, 4) |
