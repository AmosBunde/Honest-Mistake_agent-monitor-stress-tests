"""Post hoc analyses P1 and P2. Added AFTER the first results were seen.

They are not part of the pre-registered plan and feed no prediction verdict. They exist
because a review of the first results found two confounds in what E4 and E3 can claim:

P1 (E4, held-out violation). Tasks are almost violation-specific, and E4's held-out
regime keeps every benign training row. The tasks of the held-out violation therefore
appear in training only as benign, and a classifier can learn "this task is safe". P1
repeats the regime with those benign training rows removed, and reports AUROC against
benign rows of the same tasks as well as against all benign.

P2 (E3, held-out route). The same realizations (templates) generate train and test
incidents, so "trained on all four routes" is recall on templates already seen under new
names, and leaving a route out removes the route and all of its templates together. P2
splits the realizations inside every violation x route cell into two groups, trains on
one group across all four routes and tests on the other: a held-out template inside a
seen route.

Outputs: results/posthoc_P1.json, results/posthoc_P2.json, results/scores/posthoc_*.csv
and a `posthoc` block in summary.json. Nothing in E0-E7 is read except to cross-check, and
nothing in E0-E7 is written.

The analyses read `task` and the realization of each incident to build training sets and
to group test rows. Monitors still see only task_prompt, steps and events.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .experiments import (FPRS, SCORE_COLUMNS, Options, Runner, _benign, _concat, _incidents, load_data, log,
                          stratified_sample, write_json)
from .schema import HARMS, ROUTES, Trajectory

MONITORS = ("tfidf", "embed", "trace", "artifact")
EXTRA_COLUMNS = ["task", "realization", "template_group"]
POSTHOC_COLUMNS = SCORE_COLUMNS + EXTRA_COLUMNS

DESCRIPTIONS = {
    "P1": "POST HOC, added after seeing the E4 results; not pre-registered and in no verdict. Held-out violation "
          "without the task confound: the E4 held-out regime as is, and the same with the benign training rows "
          "removed for every task that lost all its incidents to the hold-out; in-distribution training for "
          "reference. AUROC is given against all test benign and against test benign of the same tasks.",
    "P2": "POST HOC, added after seeing the E3 results; not pre-registered and in no verdict. Held-out mistakes "
          "within a seen route: realizations in every violation x route cell are split into groups A and B "
          "(sorted by name, alternating); a monitor trained on one group across all four routes is tested on the "
          "other group (held-out templates) and compared with the E3 all-four regime on the same test incidents "
          "(seen templates), with the same number of training incidents.",
}


def _view(monitor: str) -> str:
    return "events" if monitor == "trace" else "commands"


# --------------------------------------------------------------------------- realizations

def recover_realizations(data: dict[str, list[Trajectory]], verify: bool = True) -> dict[str, str]:
    """Trajectory id -> name of the realization (monitorstress.incidents.MISTAKES) that produced it.

    The generator writes the incident rows of a file in the order of generate.incident_plan,
    so the i-th incident row is the i-th planned mistake. With verify, each row is also
    re-planned from its own seed and its commands compared, so a wrong mapping cannot pass.
    """
    from . import generate  # the generator half; read here for analysis only

    mapping: dict[str, str] = {}
    for file_name, source in (("train.jsonl", "train"), ("test.jsonl", "test"), ("dilution.jsonl", "dilution")):
        rows = [t for t in data.get(source, []) if t.kind == "incident"]
        if rows:
            mapping.update(match_plan(rows, generate.incident_plan(generate.FILES[file_name][3][1]), verify))
    return mapping


def match_plan(rows: list[Trajectory], plan: list[tuple], verify: bool = True) -> dict[str, str]:
    """Pair incident rows (in file order) with the generator's plan; raise if any pair disagrees."""
    from . import incidents

    if len(plan) != len(rows):
        raise ValueError(f"{len(rows)} incident rows but the generator plans {len(plan)}")
    mapping = {}
    for traj, (dilution, awareness, mistake) in zip(rows, plan):
        same = (traj.dilution, traj.awareness, traj.harm, traj.route, traj.task) == (
            dilution, awareness, mistake.harm, mistake.route, mistake.task)
        if same and verify:
            steps, _ = incidents.plan(mistake, traj.seed, awareness, dilution)
            same = [command for _, command in steps] == [s.command for s in traj.steps]
        if not same:
            raise ValueError(f"{traj.id}: row does not match planned realization {mistake.name!r}; "
                             "the data were not written by this version of the generator")
        mapping[traj.id] = mistake.name
    return mapping


def template_groups(data: dict[str, list[Trajectory]], realization_of: dict[str, str]) -> dict[str, str]:
    """Realization name -> "A" or "B": within each violation x route cell, sorted by name, alternating."""
    cells: dict[tuple, set] = defaultdict(set)
    for source in ("train", "test"):
        for t in data[source]:
            if t.kind == "incident":
                cells[(t.harm, t.route)].add(realization_of[t.id])
    groups = {}
    for cell, names in cells.items():
        if len(names) < 2:
            raise ValueError(f"cell {cell} has {len(names)} realization; it cannot be split into two groups")
        for index, name in enumerate(sorted(names)):
            groups[name] = "AB"[index % 2]
    return groups


def _annotate(frame: pd.DataFrame, data: dict, realization_of: dict[str, str], groups: dict[str, str]) -> pd.DataFrame:
    """Add the analysis-only columns (task, realization, template group) to a score table."""
    task_of = {t.id: t.task for trajs in data.values() for t in trajs}
    frame = frame.copy()
    frame["task"] = frame["id"].map(task_of)
    frame["realization"] = frame["id"].map(realization_of)
    frame["template_group"] = frame["realization"].map(groups)
    return frame[POSTHOC_COLUMNS]


def read_scores(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"id": str, "monitor": str, "regime": str, "task": str})


def _write_scores(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, columns=POSTHOC_COLUMNS, float_format="%.10g")
    return read_scores(path)  # analysis starts from what is on disk


def _area(prefix: str, pos, neg) -> dict:
    area = metrics.auroc_with_ci(pos, neg)
    return {prefix: area["auroc"], f"{prefix}_ci_low": area["ci_low"], f"{prefix}_ci_high": area["ci_high"]}


# --------------------------------------------------------------------------- P1

def orphan_tasks(all_incidents: list[Trajectory], kept_incidents: list[Trajectory]) -> set[str]:
    """Tasks that host incidents in the full training file but have none among the kept incidents."""
    return {t.task for t in all_incidents} - {t.task for t in kept_incidents}


def score_p1(runner: Runner) -> tuple[pd.DataFrame, dict]:
    meta: dict = {"n_train_incidents_per_regime": runner.n_e4, "benign_training_rows": {}}
    for harm in HARMS:
        held = runner.train_set(f"E4/holdout={harm}")
        kept = [t for t in held if t.kind == "incident"]
        orphans = orphan_tasks(runner.train_incidents, kept)
        matched = [t for t in held if t.kind == "incident" or t.task not in orphans]
        runner.custom_train_sets[f"P1/holdout={harm}/task_matched"] = matched
        meta["benign_training_rows"][harm] = {
            "tasks_removed": sorted(orphans),
            "held_out": sum(t.kind != "incident" for t in held),
            "held_out_task_matched": sum(t.kind != "incident" for t in matched),
        }
    frames = []
    for monitor in MONITORS:
        view = _view(monitor)
        frames.append(runner.table(monitor, view, "E4/all_harms", "all_harms", "val"))
        frames.append(runner.table(monitor, view, "E4/all_harms", "all_harms", "test"))
        for harm in HARMS:
            keep = lambda t, harm=harm: t.kind != "incident" or t.harm == harm  # noqa: E731
            for key, regime in ((f"E4/holdout={harm}", f"holdout={harm}"),
                                (f"P1/holdout={harm}/task_matched", f"holdout={harm}/task_matched")):
                frames.append(runner.table(monitor, view, key, regime, "val"))
                frames.append(runner.table(monitor, view, key, regime, "test", keep))
    return _concat(frames), meta


P1_TRAININGS = (("held_out", "holdout={harm}"), ("held_out_task_matched", "holdout={harm}/task_matched"),
                ("in_distribution", "all_harms"))


def analyze_p1(df: pd.DataFrame) -> dict:
    rows, table = [], []
    for monitor, g in df.groupby("monitor", sort=True):
        for harm in HARMS:
            for training, pattern in P1_TRAININGS:
                r = g[g["regime"] == pattern.format(harm=harm)]
                val, ben = _benign(r, "val"), _benign(r, "test")
                inc = _incidents(r, "test")
                inc = inc[inc["harm"] == harm]
                same_task = ben[ben["task"].isin(set(inc["task"]))]
                areas = {**_area("auroc_all_benign", inc["score"], ben["score"]),
                         **_area("auroc_same_task_benign", inc["score"], same_task["score"]),
                         "n_same_task_benign": int(len(same_task)), "tasks_of_incidents": sorted(set(inc["task"]))}
                compact = {"monitor": monitor, "fold": harm, "training": training, "n": int(len(inc))}
                for fpr in FPRS:
                    result = metrics.evaluate_at_fpr(val["score"], ben["score"], inc["score"], fpr)
                    rows.append({"monitor": monitor, "fold": harm, "training": training, **result, **areas})
                    compact[f"recall_at_{fpr:g}"] = result["recall"]
                compact["auroc_all_benign"] = areas["auroc_all_benign"]
                compact["auroc_same_task_benign"] = areas["auroc_same_task_benign"]
                table.append(compact)
    return {"rows": rows, "summary_table": table}


def check_against_e4(result: dict, out_dir: Path) -> dict | None:
    """P1's held_out and in_distribution rows are the E4 regimes; they must reproduce E4.json."""
    path = out_dir / "E4.json"
    if not path.exists():
        return None
    theirs = {(r["monitor"], r["fold"], r["training"], r["fpr_target"]): r["recall"]
              for r in json.loads(path.read_text())["rows"]}
    gaps = [abs(r["recall"] - theirs[key]) for r in result["rows"]
            if (key := (r["monitor"], r["fold"], r["training"], r["fpr_target"])) in theirs]
    return {"rows_compared": len(gaps), "max_abs_recall_difference": max(gaps) if gaps else None,
            "reproduces_E4": bool(gaps) and max(gaps) < 1e-9}


# --------------------------------------------------------------------------- P2

def score_p2(runner: Runner, realization_of: dict[str, str], groups: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    seed = runner.opts.seed
    by_group = {g: [t for t in runner.train_incidents if groups[realization_of[t.id]] == g] for g in "AB"}
    n = min(runner.n_e3, len(by_group["A"]), len(by_group["B"]))
    cell = lambda t: (t.route, t.harm, t.awareness)  # noqa: E731
    if n == runner.n_e3:
        all_four = runner.train_set("E3/all_four")  # exactly the E3 training set
    else:
        all_four = runner.train_benign + stratified_sample(runner.train_incidents, n, cell, seed)
    runner.custom_train_sets["P2/all_four"] = all_four
    for g in "AB":
        runner.custom_train_sets[f"P2/train_{g}"] = runner.train_benign + stratified_sample(by_group[g], n, cell, seed)
    meta = {
        "n_train_incidents_per_regime": n,
        "all_four_is_the_E3_training_set": n == runner.n_e3,
        "n_train_incidents_available": {g: len(by_group[g]) for g in "AB"},
        "realizations": {g: sorted(name for name, group in groups.items() if group == g) for g in "AB"},
        "sampling": "every regime uses the E3 sampler: the same number of incidents, spread over route x violation x "
                    "awareness cells",
        "n_train_realizations": {
            regime: len({realization_of[t.id] for t in runner.custom_train_sets[f"P2/{regime}"] if t.kind == "incident"})
            for regime in ("all_four", "train_A", "train_B")},
        "n_train_incidents_by_route": {
            regime: dict(Counter(t.route for t in runner.custom_train_sets[f"P2/{regime}"] if t.kind == "incident"))
            for regime in ("all_four", "train_A", "train_B")},
    }
    frames = []
    for monitor in MONITORS:
        for regime in ("all_four", "train_A", "train_B"):
            frames.append(runner.table(monitor, _view(monitor), f"P2/{regime}", regime, "val"))
            frames.append(runner.table(monitor, _view(monitor), f"P2/{regime}", regime, "test"))
    return _concat(frames), meta


def analyze_p2(df: pd.DataFrame) -> dict:
    rows, differences, table = [], [], []
    for monitor, g in df.groupby("monitor", sort=True):
        parts = {}
        for regime in ("all_four", "train_A", "train_B"):
            r = g[g["regime"] == regime]
            parts[regime] = (_benign(r, "val")["score"], _benign(r, "test")["score"], _incidents(r, "test"))
        for route in ROUTES + ("all",):
            flags: dict[tuple, np.ndarray] = {}
            compact = {(group, templates): {} for group in ("A", "B", "both") for templates in ("seen", "held_out")}
            for test_group, held_regime in (("B", "train_A"), ("A", "train_B")):
                for templates, regime in (("seen", "all_four"), ("held_out", held_regime)):
                    val, ben, inc = parts[regime]
                    inc = inc[inc["template_group"] == test_group]
                    inc = inc if route == "all" else inc[inc["route"] == route]
                    area = _area("auroc", inc["score"], ben)
                    for fpr in FPRS:
                        result = metrics.evaluate_at_fpr(val, ben, inc["score"], fpr)
                        flags[(test_group, templates, fpr)] = metrics.flag(inc["score"], result["threshold"]).astype(float)
                        rows.append({"monitor": monitor, "route": route, "test_group": test_group,
                                     "templates": templates, "training": regime, **result, **area})
                        compact[(test_group, templates)][f"recall_at_{fpr:g}"] = result["recall"]
                    compact[(test_group, templates)].update(auroc=area["auroc"], n=int(len(inc)), training=regime)
            for fpr in FPRS:  # both folds together: every test incident once per condition
                for templates in ("seen", "held_out"):
                    pooled = np.concatenate([flags[("A", templates, fpr)], flags[("B", templates, fpr)]])
                    flags[("both", templates, fpr)] = pooled
                    ci = metrics.mean_with_ci(pooled)
                    rows.append({"monitor": monitor, "route": route, "test_group": "both", "templates": templates,
                                 "training": "all_four" if templates == "seen" else "train_A / train_B",
                                 "fpr_target": fpr, "recall": ci["value"], "ci_low": ci["ci_low"],
                                 "ci_high": ci["ci_high"], "n": ci["n"], "auroc": None})
                    compact[("both", templates)][f"recall_at_{fpr:g}"] = ci["value"]
                    compact[("both", templates)].update(auroc=None, n=ci["n"])
                for test_group in ("A", "B", "both"):
                    seen, held = flags[(test_group, "seen", fpr)], flags[(test_group, "held_out", fpr)]
                    if len(seen) == 0:
                        continue
                    diff = metrics.recall_difference(seen, held, paired=True)  # the same test incidents
                    differences.append({"monitor": monitor, "route": route, "test_group": test_group, "fpr_target": fpr,
                                        "seen_minus_held_out": diff["difference"], "ci_low": diff["ci_low"],
                                        "ci_high": diff["ci_high"], "n": diff["n_a"]})
            for (test_group, templates), values in compact.items():
                table.append({"monitor": monitor, "route": route, "test_group": test_group, "templates": templates, **values})
    return {"rows": rows, "differences": differences, "summary_table": table,
            "notes": "templates = seen: the all-four regime, which trained on incidents from the tested realizations. "
                     "templates = held_out: trained on the other group of realizations only. test_group = both pools the "
                     "two folds (recall only; AUROC is not pooled across two classifiers). Thresholds are set on benign "
                     "validation trajectories per fitted monitor, as in E3."}


# --------------------------------------------------------------------------- run

def run(opts: Options, realization_of: dict[str, str] | None = None, analyses: tuple[str, ...] = ("P1", "P2"),
        threads: int | None = 2) -> dict:
    """Run the post hoc analyses and add them to summary.json. Returns {analysis: result}."""
    from .experiments import build_summary

    if threads:
        import torch

        torch.set_num_threads(threads)
    out_dir = Path(opts.out_dir)
    data = load_data(opts.data_dir)
    if realization_of is None:
        realization_of = recover_realizations(data)
    groups = template_groups(data, realization_of)
    runner = Runner(data, opts, {source: [] for source in data})
    results = {}
    for name in analyses:
        log(f"post hoc {name}")
        if name == "P1":
            frame, meta = score_p1(runner)
            analyze = analyze_p1
        elif name == "P2":
            frame, meta = score_p2(runner, realization_of, groups)
            analyze = analyze_p2
        else:
            raise ValueError(f"unknown post hoc analysis {name!r}")
        frame = _write_scores(out_dir / "scores" / f"posthoc_{name}.csv", _annotate(frame, data, realization_of, groups))
        result = {"analysis": name, "post_hoc": True, "description": DESCRIPTIONS[name],
                  "meta": {**meta, "monitors": list(MONITORS), "seed": opts.seed}, **analyze(frame)}
        if name == "P1":
            result["meta"]["check_against_E4"] = check_against_e4(result, out_dir)
        write_json(out_dir / f"posthoc_{name}.json", result)
        results[name] = json.loads((out_dir / f"posthoc_{name}.json").read_text())
    build_summary(out_dir)
    return results
