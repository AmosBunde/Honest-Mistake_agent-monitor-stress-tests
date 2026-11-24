"""Figures from results/*.json, written to docs/img/ as 150 dpi PNG.

1. e2_recall_by_route.png   recall at 1% FPR by route, monitors fitted on direct incidents (E2)
2. e3_held_out_route.png    direct-only vs leave-one-out vs all-four training, per held-out route (E3)
3. e5_thoughts.png          recall by awareness level, with and without the agent's thoughts (E5)
4. e6_dilution.png          recall against dilution (E6)
5. posthoc_p1_violation.png held-out violation with and without the task confound (post hoc P1)
6. posthoc_p2_mistake.png   seen against held-out mistakes within a seen route (post hoc P2)

A figure is drawn only if its result file exists, and a monitor appears only if it has
rows, so the figures can be drawn before the language-model monitors have finished.

Every monitor has one color in every figure (Okabe-Ito palette, safe for color-blind
readers). The `artifact` baseline (number of steps and command names only) is light gray
and comes last. Error bars are the 95% bootstrap intervals from the result files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .schema import AWARENESS, DILUTIONS, ROUTES  # noqa: E402

COLORS = {
    "rules": "#000000",
    "tfidf": "#0072B2",
    "embed": "#E69F00",
    "trace": "#009E73",
    "llm-smollm2-360m": "#CC79A7",
    "llm-qwen2.5-0.5b": "#D55E00",
    "llm-qwen2.5-1.5b": "#56B4E9",
    "artifact": "#AAAAAA",  # the structure-only baseline: neutral gray, last in every legend
}
FALLBACK_COLORS = ["#F0E442", "#882255", "#44AA99"]
MONITOR_ORDER = list(COLORS)
DPI = 150


def color_of(monitor: str) -> str:
    if monitor in COLORS:
        return COLORS[monitor]
    return FALLBACK_COLORS[sum(map(ord, monitor)) % len(FALLBACK_COLORS)]


def ordered(monitors) -> list[str]:
    monitors = set(monitors)
    known = [m for m in MONITOR_ORDER if m in monitors]
    others = sorted(monitors - set(MONITOR_ORDER))
    return [m for m in known if m != "artifact"] + others + [m for m in known if m == "artifact"]


def _load(results_dir: Path, experiment: str) -> dict | None:
    path = results_dir / f"{experiment}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _rows(rows: list[dict], **where) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def _one(rows: list[dict], **where) -> dict | None:
    found = _rows(rows, **where)
    return found[0] if found else None


def _style(ax, ylabel: str | None = "Recall") -> None:
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def _bar(ax, x: float, row: dict | None, width: float, color: str, alpha: float = 1.0, hatch: str | None = None):
    if row is None or row.get("recall") is None:
        return
    recall = row["recall"]
    ax.bar(x, recall, width, color=color, alpha=alpha, hatch=hatch, edgecolor="white", linewidth=0.5)
    if row.get("ci_low") is not None and row.get("ci_high") is not None:
        ax.errorbar(x, recall, yerr=[[max(0.0, recall - row["ci_low"])], [max(0.0, row["ci_high"] - recall)]],
                    color="#222222", linewidth=0.9, capsize=2)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _percent(fpr: float) -> str:
    return f"{100 * fpr:g}%"


# --------------------------------------------------------------------------- 1

def figure_e2(result: dict, out_dir: Path) -> Path | None:
    rows = _rows(result["rows"], sample="full", fpr_target=0.01, harm="all")
    monitors = ordered({r["monitor"] for r in rows})
    if not monitors:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.8 / len(monitors)
    for i, monitor in enumerate(monitors):
        for j, route in enumerate(ROUTES):
            _bar(ax, j + (i - (len(monitors) - 1) / 2) * width, _one(rows, monitor=monitor, route=route), width, color_of(monitor))
    ax.set_xticks(range(len(ROUTES)), ROUTES)
    ax.set_xlabel("Route by which the mistake enters (monitors fitted on direct incidents only)")
    _style(ax, "Recall at 1% false-positive rate")
    ax.legend(handles=[Patch(color=color_of(m), label=m) for m in monitors], frameon=False, ncol=len(monitors),
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return _save(fig, out_dir, "e2_recall_by_route.png")


# --------------------------------------------------------------------------- 2

TRAININGS = (("direct_only", "direct only", 0.35), ("leave_one_out", "other three routes", 0.65), ("all_four", "all four routes", 1.0))


def figure_e3(result: dict, out_dir: Path) -> Path | None:
    rows = _rows(result["rows"], fpr_target=0.01)
    monitors = ordered({r["monitor"] for r in rows})
    if not monitors:
        return None
    fig, axes = plt.subplots(1, len(monitors), figsize=(3.6 * len(monitors), 3.6), sharey=True, squeeze=False)
    width = 0.8 / len(TRAININGS)
    for ax, monitor in zip(axes[0], monitors):
        for j, route in enumerate(ROUTES):
            for i, (training, _, alpha) in enumerate(TRAININGS):
                _bar(ax, j + (i - 1) * width, _one(rows, monitor=monitor, fold=route, training=training),
                     width, color_of(monitor), alpha)
        ax.set_xticks(range(len(ROUTES)), ROUTES)
        ax.set_title(monitor, color=color_of(monitor), fontsize=11)
        ax.set_xlabel("Held-out route (tested)")
        _style(ax, "Recall at 1% false-positive rate" if ax is axes[0][0] else None)
    fig.legend(handles=[Patch(color="#444444", alpha=alpha, label=f"trained on {label}") for _, label, alpha in TRAININGS],
               frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.98))
    return _save(fig, out_dir, "e3_held_out_route.png")


# --------------------------------------------------------------------------- 3

def _reported_at(monitor: str) -> dict:
    """Language-model monitors: 5% on their subsample. The others: 1% on the full sets."""
    if monitor.startswith("llm-"):
        return {"sample": "llm_subsample", "fpr_target": 0.05}
    return {"sample": "full", "fpr_target": 0.01}


def figure_e5(result: dict, out_dir: Path) -> Path | None:
    monitors = ordered({r["monitor"] for r in result["rows"]})
    if not monitors:
        return None
    fig, axes = plt.subplots(1, len(monitors), figsize=(2.9 * len(monitors), 3.6), sharey=True, squeeze=False)
    width = 0.38
    for ax, monitor in zip(axes[0], monitors):
        at = _reported_at(monitor)
        rows = _rows(result["rows"], monitor=monitor, **at)
        for j, awareness in enumerate(AWARENESS):
            _bar(ax, j - width / 2, _one(rows, view="commands", awareness=awareness), width, color_of(monitor), 0.4)
            _bar(ax, j + width / 2, _one(rows, view="commands+thoughts", awareness=awareness), width, color_of(monitor), 1.0)
        ax.set_xticks(range(len(AWARENESS)), AWARENESS)
        ax.set_title(f"{monitor}\n{_percent(at['fpr_target'])} FPR" + (", subsample" if at["sample"] != "full" else ""),
                     color=color_of(monitor), fontsize=10)
        ax.set_xlabel("Awareness in the thoughts")
        _style(ax, "Recall" if ax is axes[0][0] else None)
    fig.legend(handles=[Patch(color="#444444", alpha=0.4, label="commands only"),
                        Patch(color="#444444", alpha=1.0, label="commands + thoughts")],
               frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.98))
    return _save(fig, out_dir, "e5_thoughts.png")


# --------------------------------------------------------------------------- 4

def figure_e6(result: dict, out_dir: Path) -> Path | None:
    monitors = ordered({r["monitor"] for r in result["rows"]})
    panels = [("Trained monitors, 1% FPR, full sets", [m for m in monitors if not m.startswith("llm-")]),
              ("Language-model monitors, 5% FPR, subsample", [m for m in monitors if m.startswith("llm-")])]
    panels = [(title, names) for title, names in panels if names]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(max(6.4, 4.6 * len(panels)), 3.9), sharey=True, squeeze=False)
    positions = {d: i for i, d in enumerate(DILUTIONS)}
    for ax, (title, names) in zip(axes[0], panels):
        for k, monitor in enumerate(names):
            offset = (k - (len(names) - 1) / 2) * 0.04  # keep overlapping error bars apart
            for source, linestyle, marker in (("val_matched", "-", "o"), ("val_dilution_1", "--", "s")):
                rows = sorted(_rows(result["rows"], monitor=monitor, threshold_from=source, **_reported_at(monitor)),
                              key=lambda r: r["dilution"])
                rows = [r for r in rows if r.get("recall") is not None]
                if not rows:
                    continue
                xs = [positions[r["dilution"]] + offset for r in rows]
                ys = [r["recall"] for r in rows]
                ax.plot(xs, ys, linestyle=linestyle, marker=marker, markersize=4, color=color_of(monitor), linewidth=1.5,
                        label=monitor if source == "val_matched" else None)
                if source == "val_matched":
                    lows = [max(0.0, r["recall"] - (r["ci_low"] if r["ci_low"] is not None else r["recall"])) for r in rows]
                    highs = [max(0.0, (r["ci_high"] if r["ci_high"] is not None else r["recall"]) - r["recall"]) for r in rows]
                    ax.errorbar(xs, ys, yerr=[lows, highs], fmt="none", color=color_of(monitor), linewidth=0.9, capsize=2)
        ax.set_xticks(list(positions.values()), [f"{d}x" for d in DILUTIONS])
        ax.set_xlabel("Dilution (length of the ordinary part)")
        ax.set_title(title, fontsize=10)
        _style(ax, "Recall on unaware incidents" if ax is axes[0][0] else None)
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.text(0.5, -0.09, "Solid: threshold set on benign validation runs at the matching dilution, so the false-positive rate holds.\n"
                         "Dashed: threshold from dilution-1 validation; the false-positive rate then rises with dilution (see the report).",
             ha="center", fontsize=8, color="#444444")
    return _save(fig, out_dir, "e6_dilution.png")


# --------------------------------------------------------------------------- 5, 6 (post hoc)

P1_TRAININGS = (("in_distribution", "all four violations in training", 1.0),
                ("held_out", "violation held out (E4 as planned)", 0.3),
                ("held_out_task_matched", "held out, benign runs of its tasks removed (post hoc)", 0.6))


def figure_p1(result: dict, out_dir: Path) -> Path | None:
    rows = _rows(result["rows"], fpr_target=0.01)
    monitors = ordered({r["monitor"] for r in rows})
    folds = list(dict.fromkeys(r["fold"] for r in rows))
    if not monitors:
        return None
    fig, axes = plt.subplots(2, len(monitors), figsize=(3.9 * len(monitors), 6.2), sharey="row", squeeze=False)
    width = 0.8 / len(P1_TRAININGS)
    for column, monitor in enumerate(monitors):
        top, bottom = axes[0][column], axes[1][column]
        for j, fold in enumerate(folds):
            for i, (training, _, alpha) in enumerate(P1_TRAININGS):
                row = _one(rows, monitor=monitor, fold=fold, training=training)
                x = j + (i - 1) * width
                _bar(top, x, row, width, color_of(monitor), alpha)
                if row is not None:
                    _bar(bottom, x, {"recall": row["auroc_all_benign"], "ci_low": row["auroc_all_benign_ci_low"],
                                     "ci_high": row["auroc_all_benign_ci_high"]}, width, color_of(monitor), alpha)
        top.set_title(monitor, color=color_of(monitor), fontsize=11)
        bottom.axhline(0.5, color="#222222", linewidth=0.8, linestyle=":")
        for ax in (top, bottom):
            ax.set_xticks(range(len(folds)), folds, fontsize=8, rotation=20)
        bottom.set_xlabel("Violation tested")
        _style(top, "Recall at 1% false-positive rate" if column == 0 else None)
        _style(bottom, "AUROC against all benign runs" if column == 0 else None)
    axes[1][0].text(-0.45, 0.515, "chance", fontsize=7, color="#222222")
    fig.legend(handles=[Patch(color="#444444", alpha=alpha, label=label) for _, label, alpha in P1_TRAININGS],
               frameon=False, ncol=1, loc="lower center", bbox_to_anchor=(0.5, 0.94), fontsize=9)
    return _save(fig, out_dir, "posthoc_p1_violation.png")


def figure_p2(result: dict, out_dir: Path) -> Path | None:
    rows = _rows(result["rows"], fpr_target=0.01, test_group="both")
    monitors = ordered({r["monitor"] for r in rows})
    if not monitors:
        return None
    groups = list(ROUTES) + ["all"]
    fig, axes = plt.subplots(1, len(monitors), figsize=(3.9 * len(monitors), 3.6), sharey=True, squeeze=False)
    width = 0.38
    for ax, monitor in zip(axes[0], monitors):
        for j, route in enumerate(groups):
            _bar(ax, j - width / 2, _one(rows, monitor=monitor, route=route, templates="seen"), width, color_of(monitor), 1.0)
            _bar(ax, j + width / 2, _one(rows, monitor=monitor, route=route, templates="held_out"), width, color_of(monitor), 0.4)
        ax.set_xticks(range(len(groups)), [g if g != "all" else "all routes" for g in groups], fontsize=8, rotation=20)
        ax.set_title(monitor, color=color_of(monitor), fontsize=11)
        ax.set_xlabel("Route of the tested incidents")
        _style(ax, "Recall at 1% false-positive rate" if ax is axes[0][0] else None)
    fig.legend(handles=[Patch(color="#444444", alpha=1.0, label="mistake seen in training (under other names)"),
                        Patch(color="#444444", alpha=0.4, label="mistake held out, its route seen")],
               frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.98))
    return _save(fig, out_dir, "posthoc_p2_mistake.png")


FIGURES = (("E2", figure_e2), ("E3", figure_e3), ("E5", figure_e5), ("E6", figure_e6),
           ("posthoc_P1", figure_p1), ("posthoc_P2", figure_p2))


def make_all(results_dir: str | Path = "results/", out_dir: str | Path = "docs/img/") -> list[Path]:
    """Draw every figure whose result file exists. Returns the paths written."""
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    written = []
    for experiment, draw in FIGURES:
        result = _load(results_dir, experiment)
        if result is None:
            continue
        path = draw(result, out_dir)
        if path is not None:
            written.append(path)
    return written
