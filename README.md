# Refusal Steering for Sub-1B Language Models

A research project applying **Contrastive Activation Addition (CAA)** with causal interpretability tools to steer small language models (≤1B parameters) toward refusal behaviour, while preserving general performance.

## Research Goals

1. Identify refusal-sensitive layers in sub-1B models via causal patching and ablation sweeps
2. Apply targeted single-layer CAA with null-space constraints (AlphaSteer approach)
3. Evaluate harmful-prompt refusal rate vs. false refusal rate vs. capability preservation
4. Compare targeted vs. dense multi-layer steering across dose-response curves

## Key Hypothesis

> Steering only at the causally identified creation layer (~41% network depth) with a null-space constraint produces fewer side effects than dense multi-layer steering, even at sub-1B scale where superposition is severe.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Run layer selectivity sweep
python scripts/layer_sweep.py --model Qwen/Qwen2.5-0.5B-Instruct --output results/

# 2. Extract contrastive steering vector at best layer
python scripts/extract_vector.py --model Qwen/Qwen2.5-0.5B-Instruct --layer 8

# 3. Evaluate all steering variants
python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --layer 8

# 4. Plot results
python scripts/plot_results.py --results results/
```

## Project Structure

```
steering_project/
├── agents/                  # OpenCode agent skill files
│   ├── LAYER_SWEEP.md       # Skill: causal layer identification
│   ├── VECTOR_EXTRACT.md    # Skill: contrastive vector extraction
│   ├── STEERING_APPLY.md    # Skill: targeted CAA application
│   ├── EVALUATION.md        # Skill: judge scoring pipeline
│   └── VISUALIZATION.md     # Skill: Matplotlib plotting
├── docs/
│   ├── CONTEXT.md           # Research context & literature summary
│   ├── LIMITATIONS.md       # Known failure modes for sub-1B models
│   └── DESIGN_DECISIONS.md  # Why each technique was chosen
├── src/
│   ├── data/                # Prompt datasets and loaders
│   ├── models/              # Model loading, hook registration
│   ├── steering/            # CAA, null-space constraint, CAST logic
│   ├── evaluation/          # Judge scoring, metrics, permutation tests
│   └── visualization/       # Matplotlib plotting helpers
├── tests/                   # Unit tests per module
└── scripts/                 # End-to-end runner scripts
```

## Recommended Models (≤1B, instruction-tuned)

| Model | Params | Layers | Notes |
|-------|--------|--------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 500M | 24 | Best reasoning/size ratio; recommended primary |
| `Qwen/Qwen2.5-1B-Instruct` | 1B | 28 | Upper bound; useful for scaling check |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | 32 | Tiny but deep; good stress test |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | 32 | Just above 1B; useful upper reference |

## Predicted Target Layer (Hypothesis)

Based on the Gemma 3 case study finding (~41% depth), predicted target layers:

| Model | Total Layers | ~41% Layer |
|-------|-------------|------------|
| Qwen2.5-0.5B | 24 | **L9–10** |
| Qwen2.5-1B | 28 | **L11–12** |
| SmolLM2-360M | 32 | **L13** |

The sweep will confirm or refute this prediction empirically.
