"""
Paired statistical significance testing between the 7 ranking models.

Reasoning: for each of the 30 repeated grouped train/test splits, all 7
models are evaluated on the *same* held-out test set (same repeat index in
`repeated_eval_all_results_with_tcr.csv`). That means, for a given metric,
model A's and model B's scores on repeat i are naturally paired observations
(same test data, same split-to-split variability). Therefore, the right choice
statistically is a paired test (Wilcoxon signed-rank, or a paired t-test), 
rather than an unpaired/independent-samples test. The latter is unnecessarily
conservative for this purpose and would ignore that shared variability.

I report all C(7,2) = 21 pairwise comparisons on MRR (the metric the
project's own README highlights as the main ranking result), using:
  - Wilcoxon signed-rank test (primary; doesn't assume normality of the
    30 paired differences)
  - paired t-test (reported alongside, as a parametric cross-check)
  - Holm-Bonferroni correction across the 21 comparisons (controls the
    family-wise error rate; less conservative than plain Bonferroni)
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
RESULTS_CSV = BASE / "results" / "repeated_eval_all_results_with_tcr.csv"
OUT_CSV = BASE / "results" / "paired_significance_mrr.csv"

METRIC = "mrr"


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    """Standard Holm step-down adjustment. Returns adjusted p-values, same order as input."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvalues[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def main():
    df = pd.read_csv(RESULTS_CSV)
    wide = df.pivot(index="repeat", columns="model", values=METRIC)
    models = sorted(wide.columns)

    rows = []
    for model_a, model_b in combinations(models, 2):
        a = wide[model_a].values
        b = wide[model_b].values
        diff = a - b

        # Wilcoxon signed-rank test fails/warns if all differences are exactly zero;
        # not expected here, but guarded for robustness.
        if np.allclose(diff, 0):
            wilcoxon_p = 1.0
        else:
            wilcoxon_p = stats.wilcoxon(a, b).pvalue

        ttest_p = stats.ttest_rel(a, b).pvalue

        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            f"{METRIC}_mean_a": a.mean(),
            f"{METRIC}_mean_b": b.mean(),
            "mean_diff (a - b)": diff.mean(),
            "wilcoxon_p": wilcoxon_p,
            "paired_ttest_p": ttest_p,
        })

    result_df = pd.DataFrame(rows)
    result_df["wilcoxon_p_holm"] = holm_bonferroni(result_df["wilcoxon_p"].values)
    result_df["significant_after_correction (alpha=0.05)"] = result_df["wilcoxon_p_holm"] < 0.05

    result_df = result_df.sort_values("wilcoxon_p_holm")
    result_df.to_csv(OUT_CSV, index=False)

    print(f"Paired comparisons on {METRIC.upper()} across {wide.shape[0]} repeats, "
          f"{len(models)} models ({len(rows)} pairwise comparisons):\n")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(result_df.to_string(index=False))

    print(f"\nSaved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
