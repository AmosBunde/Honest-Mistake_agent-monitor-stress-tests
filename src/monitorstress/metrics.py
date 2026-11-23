"""Thresholds at a false-positive rate, recall, AUROC and bootstrap intervals.

Conventions used everywhere:

- A trajectory is flagged when its score is strictly greater than the threshold.
- The threshold for a target false-positive rate is set on benign validation scores and is
  the smallest value whose validation false-positive rate does not exceed the target. Ties
  are therefore resolved conservatively.
- Intervals are 95% percentile bootstrap intervals over trajectories, 2,000 draws, fixed
  seed. The threshold is held fixed inside the bootstrap: the interval describes sampling
  of the evaluated trajectories, not of the validation set.
- Resampling indices depend only on (n, seed), so two arrays of the same length resampled
  with the same seed are resampled *jointly*. That is what makes a bootstrap paired.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

N_BOOT = 2000
SEED = 0
ALPHA = 0.05


# --------------------------------------------------------------------------- point estimates

def threshold_at_fpr(benign_scores, target_fpr: float) -> float:
    """Smallest threshold t such that mean(benign_scores > t) <= target_fpr."""
    scores = np.sort(np.asarray(benign_scores, dtype=float))[::-1]
    n = len(scores)
    if n == 0:
        return math.inf
    allowed = int(math.floor(target_fpr * n + 1e-9))  # how many benign trajectories may be flagged
    if allowed >= n:
        return -math.inf
    return float(scores[allowed])


def flag(scores, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) > threshold


def rate_above(scores, threshold: float) -> float:
    """Fraction of scores flagged: recall on attacks, false-positive rate on benign."""
    scores = np.asarray(scores, dtype=float)
    return float(np.mean(scores > threshold)) if len(scores) else math.nan


def auroc(pos_scores, neg_scores) -> float:
    """Probability that a random positive outscores a random negative (ties count one half)."""
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    neg_sorted = np.sort(neg)
    below = np.searchsorted(neg_sorted, pos, side="left")
    below_or_equal = np.searchsorted(neg_sorted, pos, side="right")
    return float((below + 0.5 * (below_or_equal - below)).sum() / (len(pos) * len(neg)))


# --------------------------------------------------------------------------- bootstrap

@lru_cache(maxsize=16)
def _indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n), dtype=np.int32)


def bootstrap_means(values, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """The mean of `values` in each bootstrap resample (shape: n_boot)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.full(n_boot, math.nan)
    if n * n_boot > 20_000_000:  # too large to hold all indices at once
        rng = np.random.default_rng(seed)
        return np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return values[_indices(n, n_boot, seed)].mean(axis=1)


def percentile_interval(draws, alpha: float = ALPHA) -> tuple[float, float]:
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return (math.nan, math.nan)
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(low), float(high))


def mean_with_ci(values, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of 0/1 flags (a recall or a false-positive rate) with its bootstrap interval."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"value": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    low, high = percentile_interval(bootstrap_means(values, n_boot, seed))
    return {"value": float(values.mean()), "ci_low": low, "ci_high": high, "n": int(len(values))}


def recall_with_ci(attack_scores, threshold: float, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    out = mean_with_ci(flag(attack_scores, threshold), n_boot, seed)
    return {"recall": out["value"], "ci_low": out["ci_low"], "ci_high": out["ci_high"], "n": out["n"]}


def auroc_with_ci(pos_scores, neg_scores, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """AUROC with a bootstrap interval; positives and negatives are resampled separately."""
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return {"auroc": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n_pos": n_pos, "n_neg": n_neg}
    # wins[i, j] = 1 if positive i outscores negative j, 0.5 on a tie. A resample is a pair of
    # count vectors (how often each trajectory was drawn), so its AUROC is a bilinear form.
    wins = (pos[:, None] > neg[None, :]).astype(np.float32) + 0.5 * (pos[:, None] == neg[None, :]).astype(np.float32)
    rng = np.random.default_rng(seed)
    pos_counts = rng.multinomial(n_pos, np.full(n_pos, 1.0 / n_pos), size=n_boot).astype(np.float32)
    neg_counts = rng.multinomial(n_neg, np.full(n_neg, 1.0 / n_neg), size=n_boot).astype(np.float32)
    draws = ((pos_counts @ wins) * neg_counts).sum(axis=1) / (n_pos * n_neg)
    low, high = percentile_interval(draws)
    return {"auroc": auroc(pos, neg), "ci_low": low, "ci_high": high, "n_pos": n_pos, "n_neg": n_neg}


def recall_difference(flags_a, flags_b, paired: bool, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """recall(a) - recall(b) with a bootstrap interval.

    paired=True: a and b are two conditions on the *same* trajectories, in the same order
    (two monitors, two training regimes, with and without thoughts). Trajectories are
    resampled once and both conditions are read from the same resample.

    paired=False: a and b are different sets of trajectories (two evasions, two dilution
    levels). Each set is resampled independently.
    """
    a = np.asarray(flags_a, dtype=float)
    b = np.asarray(flags_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"difference": math.nan, "ci_low": math.nan, "ci_high": math.nan,
                "paired": paired, "n_a": int(len(a)), "n_b": int(len(b))}
    if paired:
        if len(a) != len(b):
            raise ValueError("a paired bootstrap needs the same trajectories in both conditions")
        draws = bootstrap_means(a - b, n_boot, seed)
    else:
        draws = bootstrap_means(a, n_boot, seed) - bootstrap_means(b, n_boot, seed + 1)
    low, high = percentile_interval(draws)
    return {"difference": float(a.mean() - b.mean()), "ci_low": low, "ci_high": high,
            "paired": paired, "n_a": int(len(a)), "n_b": int(len(b))}


def evaluate_at_fpr(val_benign_scores, test_benign_scores, attack_scores, target_fpr: float) -> dict:
    """Threshold from benign validation scores, then recall and the achieved false-positive rates."""
    threshold = threshold_at_fpr(val_benign_scores, target_fpr)
    recall = recall_with_ci(attack_scores, threshold)
    return {
        "fpr_target": target_fpr,
        "threshold": threshold,
        "n_val_benign": int(len(val_benign_scores)),
        "val_fpr": rate_above(val_benign_scores, threshold),
        "n_test_benign": int(len(test_benign_scores)),
        "test_fpr": rate_above(test_benign_scores, threshold),
        "recall": recall["recall"],
        "ci_low": recall["ci_low"],
        "ci_high": recall["ci_high"],
        "n": recall["n"],
    }
