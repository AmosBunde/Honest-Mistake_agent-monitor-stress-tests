import math

import numpy as np
import pytest

from monitorstress import metrics


def test_threshold_respects_target_on_validation():
    rng = np.random.default_rng(0)
    for n in (10, 99, 100, 2000):
        scores = rng.normal(size=n)
        for target in (0.01, 0.05, 0.5):
            threshold = metrics.threshold_at_fpr(scores, target)
            achieved = metrics.rate_above(scores, threshold)
            assert achieved <= target + 1e-12
            # and it is the smallest such threshold: one more flagged trajectory would exceed the target
            assert (np.sum(scores >= threshold)) / n > target or threshold == -math.inf


def test_threshold_is_conservative_with_ties():
    scores = np.array([0.0] * 90 + [1.0] * 10)  # 10% of benign share the top score
    threshold = metrics.threshold_at_fpr(scores, 0.05)
    assert threshold == 1.0
    assert metrics.rate_above(scores, threshold) == 0.0
    assert metrics.rate_above([1.0, 2.0], threshold) == 0.5  # a score must be strictly above to be flagged


def test_threshold_edge_cases():
    assert metrics.threshold_at_fpr([], 0.01) == math.inf
    assert metrics.threshold_at_fpr([1, 2, 3], 1.0) == -math.inf
    assert metrics.threshold_at_fpr([1, 2, 3], 0.01) == 3.0  # nothing may be flagged out of three


def test_auroc_matches_sklearn_and_handles_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    pos = np.round(rng.normal(1.0, 1.0, 300), 1)
    neg = np.round(rng.normal(0.0, 1.0, 500), 1)
    expected = roc_auc_score([1] * len(pos) + [0] * len(neg), np.concatenate([pos, neg]))
    assert metrics.auroc(pos, neg) == pytest.approx(expected, abs=1e-12)
    assert metrics.auroc([1, 1], [1, 1]) == 0.5
    assert math.isnan(metrics.auroc([], [1.0]))


def test_auroc_interval_brackets_estimate_and_is_reproducible():
    rng = np.random.default_rng(2)
    pos, neg = rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 300)
    a = metrics.auroc_with_ci(pos, neg)
    b = metrics.auroc_with_ci(pos, neg)
    assert a == b
    assert a["ci_low"] < a["auroc"] < a["ci_high"]
    assert a["ci_high"] - a["ci_low"] < 0.15
    assert (a["n_pos"], a["n_neg"]) == (200, 300)


def test_recall_interval():
    scores = np.array([1.0] * 30 + [0.0] * 70)
    out = metrics.recall_with_ci(scores, 0.5)
    assert out["recall"] == pytest.approx(0.30)
    assert out["n"] == 100
    assert 0.18 < out["ci_low"] < 0.30 < out["ci_high"] < 0.42
    # close to the normal approximation
    half = 1.96 * math.sqrt(0.3 * 0.7 / 100)
    assert out["ci_low"] == pytest.approx(0.30 - half, abs=0.03)
    assert out["ci_high"] == pytest.approx(0.30 + half, abs=0.03)
    empty = metrics.recall_with_ci([], 0.5)
    assert empty["n"] == 0 and math.isnan(empty["recall"])


def test_degenerate_recall_has_degenerate_interval():
    out = metrics.recall_with_ci(np.ones(40), 0.5)
    assert (out["recall"], out["ci_low"], out["ci_high"]) == (1.0, 1.0, 1.0)


def test_paired_difference_is_tighter_than_unpaired():
    rng = np.random.default_rng(3)
    a = rng.random(400) < 0.6
    b = a.copy()
    b[:20] = ~b[:20]  # two conditions that agree on 95% of trajectories
    paired = metrics.recall_difference(a, b, paired=True)
    unpaired = metrics.recall_difference(a, b, paired=False)
    assert paired["difference"] == pytest.approx(unpaired["difference"])
    assert paired["ci_high"] - paired["ci_low"] < 0.5 * (unpaired["ci_high"] - unpaired["ci_low"])
    assert paired["ci_low"] <= paired["difference"] <= paired["ci_high"]


def test_paired_difference_needs_equal_lengths_and_unpaired_does_not():
    with pytest.raises(ValueError):
        metrics.recall_difference([1, 0, 1], [1, 0], paired=True)
    out = metrics.recall_difference([1, 1, 1, 1], [0, 0], paired=False)
    assert out["difference"] == 1.0 and (out["n_a"], out["n_b"]) == (4, 2)


def test_identical_conditions_have_zero_paired_difference():
    flags = np.random.default_rng(4).random(100) < 0.5
    out = metrics.recall_difference(flags, flags, paired=True)
    assert (out["difference"], out["ci_low"], out["ci_high"]) == (0.0, 0.0, 0.0)


def test_evaluate_at_fpr():
    val = np.arange(100, dtype=float)          # threshold at 1%: one may be flagged -> 98.0
    test_benign = np.array([0.0, 50.0, 99.0, 98.5])
    incidents = np.array([98.0, 98.1, 120.0, 3.0])
    out = metrics.evaluate_at_fpr(val, test_benign, incidents, 0.01)
    assert out["threshold"] == 98.0
    assert out["val_fpr"] == pytest.approx(0.01)
    assert out["test_fpr"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)  # 98.0 itself is not above the threshold
    assert (out["n"], out["n_val_benign"], out["n_test_benign"]) == (4, 100, 4)
    assert metrics.N_BOOT == 2000
