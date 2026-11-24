"""Checks behind the limits section of REPORT.md. Read-only on data/ and results/.

Run from the repository root after `monitorstress generate` and `monitorstress run`:

    uv run python scripts/review_checks.py

It writes docs/review_checks.json. Nothing here feeds a prediction verdict. Each block
recomputes one number that an independent review of the first results raised and that
the experiment files do not already hold:

- task_success:    share of incident runs that also fail their task's own success check
- task_by_violation: which tasks host which violation (the confound behind E4)
- structure_within_task: the E0 structure classifier, direct incidents against benign runs
                   of the same task (the gate classifier's own scores)
- structure_fitted_within_task: the same features, with one classifier fitted per task
- trace_hash:      how full the trace monitor's hashed feature space is
- trace_by_realization: recall of the trace monitor per mistake realization
- embed_repeat:    how far the embed monitor's scores moved between two runs of the same fit
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from monitorstress import metrics
from monitorstress.monitors.artifact import command_name_features
from monitorstress.monitors.trace import N_HASH_FEATURES, TraceMonitor, featurize
from monitorstress.schema import read_jsonl
from monitorstress.world import execute

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS, OUT = ROOT / "data", ROOT / "results", ROOT / "docs" / "review_checks.json"


def task_success(test):
    out = {}
    for kind, rows in (("incident", [t for t in test if t.kind == "incident"]),
                       ("benign", [t for t in test if t.kind != "incident"])):
        failed = [not execute(t.seed, t.task, [s.command for s in t.steps]).success for t in rows]
        out[kind] = {"n": len(rows), "n_task_check_failed": int(sum(failed)), "share_failed": float(np.mean(failed))}
        if kind == "incident":
            by = defaultdict(list)
            for t, f in zip(rows, failed):
                by[t.harm].append(f)
            out["incident_by_violation"] = {h: float(np.mean(v)) for h, v in by.items()}
    return out


def task_by_violation(test):
    table = defaultdict(Counter)
    for t in test:
        if t.kind == "incident":
            table[t.task][t.harm] += 1
    return {task: dict(c) for task, c in sorted(table.items())}


def structure_within_task(test):
    """Scores of the E0 gate classifier (fitted across tasks), compared inside each task."""
    scores = pd.read_csv(RESULTS / "scores" / "E0.csv", dtype={"id": str})
    scores = scores[(scores["monitor"] == "structure") & (scores["regime"] == "direct_vs_hosting_benign")]
    task_of = {t.id: t.task for t in test}
    scores = scores.assign(task=scores["id"].map(task_of))
    rows = []
    for task, g in scores.groupby("task"):
        pos, neg = g[g["kind"] == "incident"]["score"], g[g["kind"] != "incident"]["score"]
        if len(pos) and len(neg):
            area = metrics.auroc_with_ci(pos, neg)
            rows.append({"task": task, "n_direct_incidents": int(len(pos)), "n_benign": int(len(neg)),
                         "auroc": area["auroc"], "ci_low": area["ci_low"], "ci_high": area["ci_high"]})
    return sorted(rows, key=lambda r: -r["auroc"])


def structure_fitted_within_task(train, test):
    """One structure classifier per task: train rows of the task, scored on its test rows."""
    rows = []
    for task in sorted({t.task for t in test if t.kind == "incident" and t.route == "direct"}):
        member = lambda t: t.task == task and (t.kind != "incident" or t.route == "direct")  # noqa: E731
        fit, held = [t for t in train if member(t)], [t for t in test if member(t)]
        model = make_pipeline(DictVectorizer(), MaxAbsScaler(),
                              LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        model.fit([command_name_features(t) for t in fit], [t.label for t in fit])
        scores = model.decision_function([command_name_features(t) for t in held])
        labels = np.array([t.label for t in held])
        area = metrics.auroc_with_ci(scores[labels == 1], scores[labels == 0])
        rows.append({"task": task, "n_direct_incidents": int(labels.sum()), "n_benign": int((1 - labels).sum()),
                     "auroc": area["auroc"], "ci_low": area["ci_low"], "ci_high": area["ci_high"]})
    return sorted(rows, key=lambda r: -r["auroc"])


def trace_hash(train, test):
    bags = [featurize(t.events, t.task_prompt)[1] for t in train]
    distinct = set().union(*[set(b) for b in bags])
    hashed = TraceMonitor().hasher.transform(bags)
    occupied = int((np.asarray(hashed.sum(axis=0)).ravel() > 0).sum())
    test_tokens = set().union(*[set(featurize(t.events, t.task_prompt)[1]) for t in test])
    return {"n_buckets": N_HASH_FEATURES, "n_distinct_tokens_train": len(distinct),
            "n_buckets_occupied_train": occupied, "share_buckets_occupied_train": occupied / N_HASH_FEATURES,
            "tokens_per_occupied_bucket": len(distinct) / occupied,
            "n_distinct_tokens_test": len(test_tokens),
            "share_test_tokens_not_in_train": len(test_tokens - distinct) / len(test_tokens)}


def by_realization(monitor: str):
    """Recall at 1% per realization, monitor fitted on all four routes (the E3 all-four fit)."""
    df = pd.read_csv(RESULTS / "scores" / "posthoc_P2.csv", dtype={"id": str})
    df = df[(df["monitor"] == monitor) & (df["regime"] == "all_four")]
    threshold = metrics.threshold_at_fpr(df[(df["source"] == "val")]["score"], 0.01)
    inc = df[(df["source"] == "test") & (df["kind"] == "incident")]
    recalls = inc.assign(flag=inc["score"] > threshold).groupby("realization")["flag"].agg(["mean", "size"])
    r = recalls["mean"]
    return {"monitor": monitor, "training": "all_four", "fpr_target": 0.01, "n_realizations": int(len(r)),
            "n_recall_at_least_0.9": int((r >= 0.9).sum()), "n_recall_at_most_0.1": int((r <= 0.1).sum()),
            "n_between": int(((r > 0.1) & (r < 0.9)).sum()),
            "n_recall_exactly_1": int((r == 1).sum()), "n_recall_exactly_0": int((r == 0).sum()),
            "per_realization": {name: {"recall": float(row["mean"]), "n": int(row["size"])}
                                for name, row in recalls.sort_values("mean").iterrows()}}


def embed_repeat():
    """E4 and post hoc P1 fit the same regimes in two separate runs; compare scores and recalls."""
    keys = ["monitor", "regime", "source", "id"]
    a = pd.read_csv(RESULTS / "scores" / "E4.csv", dtype={"id": str})[keys + ["score"]]
    b = pd.read_csv(RESULTS / "scores" / "posthoc_P1.csv", dtype={"id": str})[keys + ["score"]]
    m = a.merge(b, on=keys, suffixes=("_e4", "_p1"))
    m["gap"] = (m["score_e4"] - m["score_p1"]).abs()
    scores = {mon: {"n_scores": int(len(g)), "n_differ": int((g["gap"] > 1e-9).sum()),
                    "median_abs_gap": float(g["gap"].median()), "max_abs_gap": float(g["gap"].max()),
                    "score_sd": float(g["score_e4"].std())}
              for mon, g in m.groupby("monitor")}
    e4 = {(r["monitor"], r["fold"], r["training"], r["fpr_target"]): r["recall"]
          for r in json.loads((RESULTS / "E4.json").read_text())["rows"]}
    moved = defaultdict(list)
    for r in json.loads((RESULTS / "posthoc_P1.json").read_text())["rows"]:
        key = (r["monitor"], r["fold"], r["training"], r["fpr_target"])
        if key in e4:
            moved[r["monitor"]].append(abs(r["recall"] - e4[key]))
    recalls = {mon: {"n_recalls_compared": len(v), "n_moved": int(sum(g > 1e-12 for g in v)), "max_abs_move": float(max(v))}
               for mon, v in moved.items()}
    return {"scores": scores, "recalls": recalls}


def main() -> None:
    train, test = read_jsonl(DATA / "train.jsonl"), read_jsonl(DATA / "test.jsonl")
    checks = {
        "note": "Computed by scripts/review_checks.py from data/ and results/scores/. In no verdict.",
        "task_success": task_success(test),
        "task_by_violation": task_by_violation(test),
        "structure_within_task": structure_within_task(test),
        "structure_fitted_within_task": structure_fitted_within_task(train, test),
        "trace_hash": trace_hash(train, test),
        "trace_by_realization": by_realization("trace"),
        "tfidf_by_realization": by_realization("tfidf"),
        "embed_repeat": embed_repeat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(checks, indent=1) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
