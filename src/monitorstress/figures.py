"""Figures from results/*.json, written to docs/img/ as 150 dpi PNG.

1. e2_recall_by_route.png   recall at 1% FPR by route, monitors fitted on direct incidents (E2)
2. e3_held_out_route.png    direct-only vs leave-one-out vs all-four training, per held-out route (E3)
3. e5_thoughts.png          recall by awareness level, with and without the agent's thoughts (E5)
4. e6_dilution.png          recall against dilution (E6)

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
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.7), sharey=True, squeeze=False)
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
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.5, -0.06, "Solid: threshold set on validation at the matching dilution.  "
                         "Dashed: threshold from dilution-1 validation.", ha="center", fontsize=8, color="#444444")
    return _save(fig, out_dir, "e6_dilution.png")


FIGURES = (("E2", figure_e2), ("E3", figure_e3), ("E5", figure_e5), ("E6", figure_e6))


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
