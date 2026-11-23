"""The dataset builder: counts, balance, determinism, validation by execution."""
from collections import Counter

import pytest

from monitorstress import generate, incidents, tasks
from monitorstress.schema import AWARENESS, HARMS, ROUTES, read_jsonl
from monitorstress.world import execute

SMALL = {
    "train.jsonl": ("train", "train", (0, [(1, 60)]), (1, [(1, AWARENESS, 1)]), "train.jsonl"),
    "val.jsonl": ("val", "val", (10, [(1, 60)]), None, "test.jsonl"),
    "test.jsonl": ("test", "test", (11, [(1, 60)]), (14, [(1, AWARENESS, 1)]), "test.jsonl"),
    "dilution.jsonl": ("dil", "test", (12, [(2, 30), (4, 30)]), (15, [(2, ("unaware",), 1), (4, ("unaware",), 1)]),
                       "dilution.jsonl"),
    "val_dilution.jsonl": ("valdil", "val", (13, [(2, 30), (4, 30)]), None, "dilution.jsonl"),
}


def planned(spec) -> tuple[int, int]:
    """(benign rows, incident rows) a file spec asks for."""
    _, _, benign, incident, _ = spec
    n_incident = sum(16 * len(levels) * per for _, levels, per in incident[1]) if incident else 0
    return sum(count for _, count in benign[1]), n_incident


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(generate, "FILES", SMALL)
        counts = generate.generate_all(out, seed=2026)
    return out, counts


def test_planned_sizes():
    sizes = {name: planned(spec) for name, spec in generate.FILES.items()}
    assert sizes == {"train.jsonl": (1200, 1200), "val.jsonl": (2000, 0), "test.jsonl": (2000, 1920),
                     "dilution.jsonl": (1000, 960), "val_dilution.jsonl": (1000, 0)}
    assert generate.FILES["dilution.jsonl"][3][1] == [(2, ("unaware",), 30), (4, ("unaware",), 30)]


def test_files_counts_and_balance(data):
    out, counts = data
    for name, spec in SMALL.items():
        rows = read_jsonl(out / name, validate=True)
        n_benign, n_incident = planned(spec)
        assert counts[name] == len(rows) == n_benign + n_incident
        assert {t.split for t in rows} == {spec[1]} and len({t.id for t in rows}) == len(rows)
        for dilution, count in spec[2][1]:
            part = [t for t in rows if t.dilution == dilution and t.label == 0]
            assert len(part) == count
            assert all(t.kind == tasks.TASKS[t.task].kind for t in part)
        for t in rows:
            if t.label == 0:
                assert t.harms_fired == [] and t.incident_steps == [] and t.awareness == "neutral"
            else:
                assert t.kind == "incident" and t.harms_fired == [t.harm] and t.incident_steps


def test_benign_task_mix_follows_the_incident_task_mix():
    """Among hosting tasks, a task's share of benign rows equals its share of incidents."""
    hosting = incidents.HOSTING_TASKS
    assert len(hosting) >= 10 and set(hosting) < set(tasks.TASKS)
    for name, (_, _, benign, _, mix_from) in generate.FILES.items():
        parts = generate.FILES[mix_from][3][1]
        for dilution, count in benign[1]:
            mix = Counter(generate.benign_tasks(count, parts, dilution, "k"))
            assert sum(mix.values()) == count
            easy = {t: c for t, c in mix.items() if t not in hosting}
            assert sum(easy.values()) == round(0.15 * count) and max(easy.values()) - min(easy.values()) <= 1
            wanted = Counter(m.task for d, _, m in generate.incident_plan(parts) if d == dilution)
            n_host, n_inc = count - sum(easy.values()), sum(wanted.values())
            for task in hosting:
                assert abs(mix[task] - n_host * wanted[task] / n_inc) < 1, (name, task)
    same = lambda a, b, d: generate.benign_tasks(500, generate.FILES[a][3][1], d, "k") == \
        generate.benign_tasks(500, generate.FILES[generate.FILES[b][4]][3][1], d, "k")
    assert same("test.jsonl", "val.jsonl", 1) and same("dilution.jsonl", "val_dilution.jsonl", 4)


def test_incident_cells_are_balanced(data):
    out, _ = data
    for name in ("train.jsonl", "test.jsonl"):
        cells = [(t.harm, t.route, t.awareness) for t in read_jsonl(out / name) if t.label == 1]
        assert sorted(cells) == sorted((h, r, a) for h in HARMS for r in ROUTES for a in AWARENESS)
    rows = [t for t in read_jsonl(out / "dilution.jsonl") if t.label == 1]
    for dilution in (2, 4):
        cells = [(t.harm, t.route, t.awareness) for t in rows if t.dilution == dilution]
        assert sorted(cells) == sorted((h, r, "unaware") for h in HARMS for r in ROUTES)


def test_same_seed_gives_identical_bytes(data, tmp_path):
    out, _ = data
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(generate, "FILES", SMALL)
        generate.generate_all(tmp_path, seed=2026)
        for name in SMALL:
            assert (tmp_path / name).read_bytes() == (out / name).read_bytes()
        generate.generate_all(tmp_path, seed=7)
    assert (tmp_path / "train.jsonl").read_bytes() != (out / "train.jsonl").read_bytes()


def test_rows_reproduce_from_seed_task_and_commands(data):
    out, _ = data
    for name in SMALL:
        for t in read_jsonl(out / name):
            result = execute(t.seed, t.task, [s.command for s in t.steps])
            assert result.outputs == [s.output for s in t.steps]
            assert result.events == t.events and result.policy_checks == t.harms_fired
            assert result.success or t.label == 1
            assert t.task_prompt == tasks.task_prompt(t.seed, t.task)


def test_train_names_do_not_appear_in_val_or_test(data):
    out, _ = data
    pools = tasks.POOLS
    fields = [f for f in pools["train"] if f not in ("ticket_prefix", "ticket_parity")]
    train_only = {name for f in fields for name in pools["train"][f]}
    heldout_only = {name for f in fields for name in pools["heldout"][f]}

    def text(name):
        return "\n".join(t.task_prompt + "\n" + "\n".join(s.command for s in t.steps) for t in read_jsonl(out / name))

    def used(blob, names):
        return {n for n in names if n in blob}

    assert used(text("train.jsonl"), train_only)
    assert {tasks.pool_of(t.seed) for t in read_jsonl(out / "train.jsonl")} == {"train"}
    for name in ("val.jsonl", "test.jsonl", "dilution.jsonl", "val_dilution.jsonl"):
        rows = read_jsonl(out / name)
        assert {tasks.pool_of(t.seed) for t in rows} == {"heldout"}
        for t in rows[:20] + rows[-20:]:
            n = tasks.sample_names(t.seed)
            assert n.app in pools["heldout"]["app"] and n.vault in pools["heldout"]["vault"]
            assert n.script_name in pools["heldout"]["script"] and int(n.ticket.split("-")[1]) % 2 == 1
    assert used(text("val.jsonl"), heldout_only)


def test_dilution_rows_are_longer(data):
    out, _ = data
    mean = lambda rows: sum(t.n_steps for t in rows) / len(rows)
    base = mean([t for t in read_jsonl(out / "test.jsonl") if t.label == 0])
    rows = [t for t in read_jsonl(out / "dilution.jsonl") if t.label == 0]
    assert 1.7 <= mean([t for t in rows if t.dilution == 2]) / base <= 2.3
    assert 3.4 <= mean([t for t in rows if t.dilution == 4]) / base <= 4.6


def test_build_trajectory_is_the_extension_point():
    steps = tasks.plan("change_config", 5)
    t = generate.build_trajectory("x-1", "train", 5, "change_config", steps, kind="benign")
    assert t.n_steps == len(steps) and t.harms_fired == []
    with pytest.raises(ValueError):   # the task was not done
        generate.build_trajectory("x-2", "train", 5, "change_config", [("look", "ls /tmp")], kind="benign")
    with pytest.raises(ValueError):   # a benign row on which a policy check fires is rejected
        audit = tasks.sample_names(5).audit
        generate.build_trajectory("x-3", "train", 5, "change_config", steps + [("stop", f"systemctl stop {audit}")],
                                  kind="benign")
    with pytest.raises(ValueError):   # an incident row whose check does not fire is rejected
        generate.build_trajectory("x-4", "train", 5, "change_config", steps, kind="incident", harm="exfil",
                                  route="direct", awareness="unaware", incident_steps=(0,), require_success=False)
