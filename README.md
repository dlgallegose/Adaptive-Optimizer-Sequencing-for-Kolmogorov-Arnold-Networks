# Plateau-Driven Dynamic Optimizer Staging for KANs (Reference Implementation)

> **Note:** This repository accompanies a paper that is not yet published.
> Please do not make this repository public until the paper is officially
> online, and update the citation section below once it is.

This repository is a **reference implementation** of the plateau-driven
dynamic optimizer staging algorithm and the seven optimizer protocols
described in *"Adaptive Optimizer Sequencing for Kolmogorov-Arnold
Networks."* It is written independently for public release: it reproduces
the algorithm's logic (Algorithm 1) and protocol definitions (Section 3.3),
but is not the code used to produce the results reported in the paper, and
does not use the paper's datasets.

For demonstration, this repo runs the same protocol sweep on
[**California Housing**](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset),
a well-known scikit-learn / Google Colab regression benchmark (20,640
samples, 8 features, predicting median house value), fetched automatically
via `sklearn.datasets.fetch_california_housing` -- no external files needed.

## What's implemented

- **Algorithm 1 (plateau-driven staging)**: switches optimizers after
  `plateau_patience` epochs without validation improvement, halts on
  `early_stopping_patience` or a wall-clock budget, and always returns the
  best checkpoint seen across all phases.
- **Seven protocols**: AdamW Only, L-BFGS Only, AdamW → L-BFGS,
  L-BFGS → AdamW, SAM + AdamW, SAM+AdamW → L-BFGS, L-BFGS → SAM+AdamW.
- **Four architectures**: simplified reference KAN variants
  (Efficient-KAN-style B-spline, ChebyKAN-style Chebyshev polynomial,
  RBF-KAN-style radial basis) and an MLP baseline, matching the
  single-hidden-layer-width-32 (KAN) / two-hidden-layer (MLP) structure
  described in the paper. L-BFGS-based protocols are excluded from the MLP
  baseline, matching the paper's design.
- **SAM + AdamW**, with the gradient computed at the perturbed point only
  (not summed with the clean-point gradient).

## Quick start

```bash
pip install -r requirements.txt
python run_experiment.py --seeds 5 --time-budget 60
```

This runs all protocols across all four architectures, for seeds 1-5, with
a 60-second wall-clock budget per run, and prints a results table plus
saves `results/california_housing_results.csv` and `.pkl`.

### Options

| Flag | Default | Description |
|---|---|---|
| `--seeds` | 5 | Number of random seeds to run (1..N) |
| `--time-budget` | 60.0 | Wall-clock budget per run, in seconds |
| `--out-dir` | `results` | Output directory for saved results |

## Output format

Results are printed in the same format as the paper's Tables 3-5: one row
per architecture/protocol, ranked within each architecture block by the
MAE × Time efficiency ratio (lower is better), along with R² and MAE
(mean ± std across seeds) and wall-clock time.

```
Architecture   Rank Protocol                             R2             MAE    WallTime(s)   MAExTime
----------------------------------------------------------------------------------------------------
Efficient-KAN     1 L-BFGS -> AdamW           0.8123+/-0.0050  0.312+/-0.010     1.8+/-0.3       0.56
...
```

## Notes on this reference implementation

- Architectures are simplified reference versions of Efficient-KAN, ChebyKAN,
  and RBF-KAN, sufficient to demonstrate the staging algorithm and protocol
  behavior; they are not guaranteed to numerically match any specific
  published implementation.
- Hyperparameters (learning rates, patience values, SAM ρ) default to the
  values reported in the paper's Table 2 but can be changed in
  `optimizer_staging.py`'s `Config` dataclass.
- This code has not been tuned or validated against the paper's original
  private results; it is provided to make the algorithm and protocol
  definitions concretely reproducible on a public dataset.

## Files

- `optimizer_staging.py` -- model definitions, SAM+AdamW, protocol
  definitions, and the Algorithm 1 training loop.
- `run_experiment.py` -- loads California Housing, runs the full
  architecture × protocol × seed sweep, prints and saves results.
- `requirements.txt` -- Python dependencies.

## Citation

A citation will be added here once the associated paper is published. If you
use this code in the meantime, please check back for the full reference, or
contact the authors.

## License

MIT -- see [LICENSE](LICENSE).
