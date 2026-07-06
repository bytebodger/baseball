"""Paired bootstrap comparison of two win-probability methods on the same
held-out 2024-2025 backtest games -- e.g. is GamePredictor's edge over the
ERA/wRC+ logistic regression baseline (see backtest.py) bigger than
resampling noise alone would produce?

Resamples games with replacement -- the SAME resampled indices are used for
both methods on each draw, since this is a paired comparison (both methods
are scored against the identical bootstrap sample of games, not two
independent samples) -- computing each method's accuracy and Brier score on
that resample, then the (method_a - method_b) difference. Repeating this
`n_resamples` times gives an empirical distribution of that difference: if
the fraction of resamples favoring one method is close to 50%, the
difference backtest.py's single-point summary table shows could just as
easily be sampling noise as a real effect.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.inference.backtest import GAME_PREDICTOR_METHOD, LOGISTIC_REGRESSION_METHOD, add_common_args, generate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_METHOD_A = GAME_PREDICTOR_METHOD
DEFAULT_METHOD_B = LOGISTIC_REGRESSION_METHOD


def bootstrap_compare(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_resamples: int = 1000,
    seed: int | None = None,
) -> pd.DataFrame:
    """One row per resample: each method's accuracy/Brier score on that
    resample (games drawn with replacement, same draw index used for both
    methods), plus their differences (a - b, so positive accuracy_diff or
    negative brier_diff both mean "a did better")."""
    n = len(y_true)
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y = y_true[idx]
        a = probs_a[idx]
        b = probs_b[idx]

        accuracy_a = float(((a >= 0.5) == (y == 1.0)).mean())
        accuracy_b = float(((b >= 0.5) == (y == 1.0)).mean())
        brier_a = float(np.mean((a - y) ** 2))
        brier_b = float(np.mean((b - y) ** 2))

        rows.append(
            {
                "accuracy_a": accuracy_a,
                "accuracy_b": accuracy_b,
                "accuracy_diff": accuracy_a - accuracy_b,
                "brier_a": brier_a,
                "brier_b": brier_b,
                "brier_diff": brier_a - brier_b,
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap(results: pd.DataFrame) -> dict:
    """Fractions are ties-excluded: a resample where the two methods land on
    the exact same accuracy (common with a small, fixed set of possible
    accuracy values) or Brier score favors neither."""
    n = len(results)

    def _ci95(col: str) -> tuple[float, float]:
        return float(results[col].quantile(0.025)), float(results[col].quantile(0.975))

    return {
        "n_resamples": n,
        "accuracy_diff_mean": float(results["accuracy_diff"].mean()),
        "accuracy_diff_std": float(results["accuracy_diff"].std()),
        "accuracy_diff_ci95": _ci95("accuracy_diff"),
        "fraction_favoring_a_accuracy": float((results["accuracy_diff"] > 0).mean()),
        "fraction_favoring_b_accuracy": float((results["accuracy_diff"] < 0).mean()),
        "brier_diff_mean": float(results["brier_diff"].mean()),
        "brier_diff_std": float(results["brier_diff"].std()),
        "brier_diff_ci95": _ci95("brier_diff"),
        # lower Brier is better, so A "favored" means brier_diff (a - b) < 0
        "fraction_favoring_a_brier": float((results["brier_diff"] < 0).mean()),
        "fraction_favoring_b_brier": float((results["brier_diff"] > 0).mean()),
    }


def plot_bootstrap_distributions(results: pd.DataFrame, name_a: str, name_b: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, col, title in [
        (axes[0], "accuracy_diff", "Accuracy difference (a - b)"),
        (axes[1], "brier_diff", "Brier score difference (a - b)"),
    ]:
        ax.hist(results[col], bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.5, label="No difference")
        ax.axvline(results[col].mean(), color="firebrick", linestyle="-", linewidth=1.5, label="Observed mean")
        ax.set_title(title)
        ax.set_xlabel(f"{name_a} minus {name_b}")
        ax.set_ylabel("Resamples")
        ax.legend(fontsize=8)

    fig.suptitle(f"Paired bootstrap: {name_a} vs {name_b} ({len(results)} resamples)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap comparison of two backtest.py methods on the held-out 2024-2025 games."
    )
    add_common_args(parser)
    parser.add_argument("--method-a", default=DEFAULT_METHOD_A, help="Must match a method name generate_predictions produces.")
    parser.add_argument("--method-b", default=DEFAULT_METHOD_B)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None, help="Omit for a fresh random draw each run.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "bootstrap")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    predictions, _ = generate_predictions(args)
    if args.method_a not in predictions or args.method_b not in predictions:
        raise ValueError(
            f"--method-a/--method-b must be one of {list(predictions.keys())}, "
            f"got {args.method_a!r} and {args.method_b!r}"
        )

    y_true, probs_a = predictions[args.method_a]
    _, probs_b = predictions[args.method_b]

    logger.info(
        "Bootstrapping %d resamples of %d games: %s vs %s", args.n_resamples, len(y_true), args.method_a, args.method_b
    )
    results = bootstrap_compare(y_true, probs_a, probs_b, n_resamples=args.n_resamples, seed=args.seed)
    summary = summarize_bootstrap(results)

    logger.info(
        "Accuracy diff: mean=%.4f std=%.4f 95%% CI=(%.4f, %.4f) -- %s favored in %.1f%% of resamples, "
        "%s in %.1f%%",
        summary["accuracy_diff_mean"], summary["accuracy_diff_std"], *summary["accuracy_diff_ci95"],
        args.method_a, summary["fraction_favoring_a_accuracy"] * 100,
        args.method_b, summary["fraction_favoring_b_accuracy"] * 100,
    )
    logger.info(
        "Brier diff:    mean=%.4f std=%.4f 95%% CI=(%.4f, %.4f) -- %s favored in %.1f%% of resamples, "
        "%s in %.1f%%",
        summary["brier_diff_mean"], summary["brier_diff_std"], *summary["brier_diff_ci95"],
        args.method_a, summary["fraction_favoring_a_brier"] * 100,
        args.method_b, summary["fraction_favoring_b_brier"] * 100,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "bootstrap_resamples.csv"
    results.to_csv(results_path, index=False)

    plot_path = args.output_dir / "bootstrap_distributions.png"
    plot_bootstrap_distributions(results, args.method_a, args.method_b, plot_path)

    logger.info("Wrote per-resample results to %s and distribution plot to %s", results_path, plot_path)


if __name__ == "__main__":
    main()
