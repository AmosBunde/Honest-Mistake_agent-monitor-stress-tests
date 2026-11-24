import json
import random

import pandas as pd
import pytest

from conftest import HOSTING_TASKS, fixture_realizations
from monitorstress import experiments, posthoc
from monitorstress.experiments import Options
from monitorstress.schema import HARMS, ROUTES

E_FILES = ("E1.json", "E3.json", "E4.json", "scores/E1.csv", "scores/E3.csv", "scores/E4.csv")


@pytest.fixture(scope="module")
def run_dir(data_dir, embed_cache_dir, fixture_data, tmp_path_factory):
    """E1, E3 and E4 on the fixture, then the post hoc analyses on top."""
    out = tmp_path_factory.mktemp("posthoc")
    opts = Options(data_dir=data_dir, out_dir=out, embed_cache_dir=embed_cache_dir, skip_llm=True,
                   experiments=("E1", "E3", "E4"), hosting_tasks=HOSTING_TASKS,
                   llm_n_benign=40, llm_n_per_cell=1, llm_n_benign_dilution=10)
    experiments.run(opts)
    before = {name: (out / name).read_bytes() for name in E_FILES}
    predictions = json.loads((out / "summary.json").read_text())["predictions"]
    results = posthoc.run(opts, realization_of=fixture_realizations(fixture_data))
    return out, before, predictions, results


def test_preregistered_outputs_are_untouched(run_dir):
    out, before, predictions, _ = run_dir
    for name in E_FILES:
        assert (out / name).read_bytes() == before[name], name
    summary = json.loads((out / "summary.json").read_text())
    assert summary["predictions"] == predictions
    assert set(summary["posthoc"]) == {"P1", "P2"}
    for name, block in summary["posthoc"].items():
        assert "POST HOC" in block["description"] and "after seeing" in block["description"]
        assert block["file"] == f"posthoc_{name}.json" and block["table"]
    assert not any("posthoc" in json.dumps(p).lower() for p in summary["predictions"])


def test_p1_reproduces_e4_and_adds_the_task_matched_regime(run_dir):
    out, _, _, results = run_dir
    p1 = results["P1"]
    assert p1["post_hoc"] is True
    check = p1["meta"]["check_against_E4"]
    assert check["reproduces_E4"] and check["rows_compared"] == len(posthoc.MONITORS) * 4 * 2 * 2
    assert {(r["monitor"], r["fold"], r["training"]) for r in p1["rows"]} == {
        (m, h, t) for m in posthoc.MONITORS for h in HARMS for t in ("held_out", "held_out_task_matched", "in_distribution")}
    frame = posthoc.read_scores(out / "scores" / "posthoc_P1.csv")
    assert list(frame.columns) == posthoc.POSTHOC_COLUMNS and frame["task"].notna().all()
    for row in p1["rows"]:
        assert row["val_fpr"] <= row["fpr_target"] + 1e-12 and row["n"] == 48
        assert 0 <= row["auroc_all_benign"] <= 1 and row["n_same_task_benign"] > 0
        r = frame[(frame["monitor"] == row["monitor"]) & (frame["regime"] == dict(posthoc.P1_TRAININGS)[row["training"]].format(harm=row["fold"]))]
        benign = r[(r["source"] == "test") & (r["kind"] != "incident")]
        assert row["n_same_task_benign"] == int(benign["task"].isin(row["tasks_of_incidents"]).sum())
    # In the fixture every hosting task hosts several violations, so no task is orphaned and the two
    # held-out regimes coincide (orphan_tasks itself is tested below).
    for harm in HARMS:
        rows = p1["meta"]["benign_training_rows"][harm]
        assert rows["tasks_removed"] == [] and rows["held_out_task_matched"] == rows["held_out"] == 144
        for monitor in posthoc.MONITORS:
            a = experiments.pick(p1["rows"], monitor=monitor, fold=harm, training="held_out", fpr_target=0.01)
            b = experiments.pick(p1["rows"], monitor=monitor, fold=harm, training="held_out_task_matched", fpr_target=0.01)
            assert a["recall"] == b["recall"] and a["auroc_all_benign"] == pytest.approx(b["auroc_all_benign"])
    assert len(p1["summary_table"]) == len(posthoc.MONITORS) * 4 * 3
    assert {"recall_at_0.01", "recall_at_0.05", "auroc_all_benign", "auroc_same_task_benign"} <= set(p1["summary_table"][0])


def test_orphan_tasks(fixture_data):
    incidents = [t for t in fixture_data["train"] if t.kind == "incident"]
    kept = [t for t in incidents if t.task != "backup"]
    assert posthoc.orphan_tasks(incidents, kept) == {"backup"}
    assert posthoc.orphan_tasks(incidents, incidents) == set()


def test_template_groups_alternate_within_each_cell(fixture_data):
    names = fixture_realizations(fixture_data)
    groups = posthoc.template_groups(fixture_data, names)
    assert len(groups) == 16 * 3
    for harm in HARMS:
        for route in ROUTES:
            assert [groups[f"{harm}_{route}_{k}"] for k in range(3)] == ["A", "B", "A"]
    one = {i: "only" for i in names}
    with pytest.raises(ValueError):
        posthoc.template_groups(fixture_data, one)


def test_p2_trains_and_tests_on_disjoint_templates(run_dir, fixture_data, data_dir, embed_cache_dir):
    out, _, _, results = run_dir
    p2 = results["P2"]
    meta = p2["meta"]
    assert meta["n_train_incidents_per_regime"] == 36 and meta["all_four_is_the_E3_training_set"]
    assert not set(meta["realizations"]["A"]) & set(meta["realizations"]["B"])
    for regime in ("all_four", "train_A", "train_B"):
        assert sum(meta["n_train_incidents_by_route"][regime].values()) == 36

    names = fixture_realizations(fixture_data)
    groups = posthoc.template_groups(fixture_data, names)
    runner = experiments.Runner(fixture_data, Options(data_dir=data_dir, out_dir=out, embed_cache_dir=embed_cache_dir),
                                {k: [] for k in fixture_data})
    by_group = {g: {t.id for t in fixture_data["train"] if t.kind == "incident" and groups[names[t.id]] == g} for g in "AB"}
    assert by_group["A"] and by_group["B"] and not by_group["A"] & by_group["B"]

    frame = posthoc.read_scores(out / "scores" / "posthoc_P2.csv")
    assert set(frame["regime"]) == {"all_four", "train_A", "train_B"}
    incidents = frame[frame["kind"] == "incident"]
    assert incidents["realization"].notna().all() and set(incidents["template_group"]) == {"A", "B"}
    e3 = experiments.read_scores(out / "scores" / "E3.csv")
    for monitor in posthoc.MONITORS:  # the seen-template regime is E3's all_four, score for score
        mine = frame[(frame["monitor"] == monitor) & (frame["regime"] == "all_four")].sort_values(["source", "id"])
        theirs = e3[(e3["monitor"] == monitor) & (e3["regime"] == "all_four")].sort_values(["source", "id"])
        assert mine["score"].to_numpy() == pytest.approx(theirs["score"].to_numpy())

    held = [r for r in p2["rows"] if r["templates"] == "held_out" and r["test_group"] in ("A", "B")]
    assert {(r["test_group"], r["training"]) for r in held} == {("B", "train_A"), ("A", "train_B")}
    assert {r["route"] for r in p2["rows"]} == {*ROUTES, "all"}
    for monitor in posthoc.MONITORS:
        a = experiments.pick(p2["rows"], monitor=monitor, route="all", test_group="A", templates="seen", fpr_target=0.01)
        b = experiments.pick(p2["rows"], monitor=monitor, route="all", test_group="B", templates="seen", fpr_target=0.01)
        both = experiments.pick(p2["rows"], monitor=monitor, route="all", test_group="both", templates="seen", fpr_target=0.01)
        assert a["n"] + b["n"] == both["n"] == 192 and a["n"] == 128  # two of three pseudo-templates are in group A
        assert both["recall"] == pytest.approx((a["recall"] * a["n"] + b["recall"] * b["n"]) / both["n"])
        held_both = experiments.pick(p2["rows"], monitor=monitor, route="all", test_group="both", templates="held_out", fpr_target=0.01)
        diff = experiments.pick(p2["differences"], monitor=monitor, route="all", test_group="both", fpr_target=0.01)
        assert diff["seen_minus_held_out"] == pytest.approx(both["recall"] - held_both["recall"])
        assert diff["ci_low"] <= diff["seen_minus_held_out"] <= diff["ci_high"]
    assert len(p2["summary_table"]) == len(posthoc.MONITORS) * 5 * 3 * 2


def test_realizations_are_recovered_from_the_generator_plan():
    from monitorstress import generate

    parts = [(1, ("unaware",), 3)]
    rows = generate.incident_rows("x", "test", 14, parts, 2026)
    plan = generate.incident_plan(parts)
    mapping = posthoc.match_plan(rows, plan)
    assert [mapping[t.id] for t in rows] == [mistake.name for _, _, mistake in plan]
    assert len(set(mapping.values())) > 16  # more than one realization per cell
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    with pytest.raises(ValueError):
        posthoc.match_plan(shuffled, plan)
    with pytest.raises(ValueError):
        posthoc.match_plan(rows[:-1], plan)


def test_reanalyze_changes_nothing(run_dir):
    out, before, predictions, _ = run_dir
    summary = experiments.reanalyze(out)
    for name in E_FILES:
        assert (out / name).read_bytes() == before[name], name
    assert summary["predictions"] == predictions and set(summary["posthoc"]) == {"P1", "P2"}


def test_reporting_additions(run_dir):
    out, _, _, _ = run_dir
    e1 = json.loads((out / "E1.json").read_text())
    points = e1["rules_operating_points"]
    frame = experiments.read_scores(out / "scores" / "E1.csv")
    rules = frame[frame["monitor"] == "rules"]
    assert [p["threshold"] for p in points] == sorted(rules["score"].unique(), reverse=True)
    assert points[0]["val_fpr"] == points[0]["test_fpr"] == points[0]["recall_direct"] == 0.0
    for p in points:
        val = rules[rules["source"] == "val"]["score"]
        direct = rules[(rules["kind"] == "incident")]["score"]
        assert p["val_fpr"] == pytest.approx((val > p["threshold"]).mean())
        assert p["recall_direct"] == pytest.approx((direct > p["threshold"]).mean())
    assert all(a["val_fpr"] <= b["val_fpr"] for a, b in zip(points, points[1:]))

    e3 = json.loads((out / "E3.json").read_text())
    for mean in e3["gap_closed_mean"]:
        assert "ALL four folds" in mean["pooled_gap_closed_note"]
        folds = [g for g in e3["gap_closed"] if g["monitor"] == mean["monitor"] and g["fpr_target"] == mean["fpr_target"]
                 and g["fold"] in mean["folds_included"]]
        if folds:
            expected = sum(g["closed"] for g in folds) / sum(g["gap"] for g in folds)
            assert mean["pooled_gap_closed_included_folds"] == pytest.approx(expected)
        else:
            assert mean["pooled_gap_closed_included_folds"] is None


def test_posthoc_figures_are_drawn(run_dir, tmp_path):
    from monitorstress import figures

    out = run_dir[0]
    written = [p.name for p in figures.make_all(out, tmp_path / "img")]
    assert "posthoc_p1_violation.png" in written and "posthoc_p2_mistake.png" in written
    assert all((tmp_path / "img" / name).stat().st_size > 10_000 for name in written)
