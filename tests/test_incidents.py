"""Incidents: one honest mistake on an ordinary task, labeled by execution."""
import pytest

from monitorstress import generate, incidents, tasks
from monitorstress.schema import AWARENESS, HARMS, ROUTES
from monitorstress.world import execute

SEEDS = [3, 4, tasks.HELDOUT_SEED_START + 3, tasks.HELDOUT_SEED_START + 4]
NAMES = [m.name for m in incidents.MISTAKES]


def row(mistake, seed, awareness="unaware", dilution=1):
    return generate.incident_row("x-0", "test", seed, mistake, awareness, dilution)


def test_every_cell_has_at_least_three_realizations():
    assert incidents.CELLS == [(h, r) for h in HARMS for r in ROUTES]
    for harm, route in incidents.CELLS:
        realizations = incidents.cell(harm, route)
        assert len(realizations) >= 3, (harm, route)
        assert len({m.summary for m in realizations}) == len(realizations)
    assert all(m.task in tasks.TASKS for m in incidents.MISTAKES)


def test_at_least_four_phrasings_per_level_per_violation():
    for harm in HARMS:
        for level in ("noticed", "mistaken"):
            phrasings = {p for m in incidents.MISTAKES if m.harm == harm for p in m.thoughts[level]}
            assert len(phrasings) >= 4, (harm, level)
        # unaware steps mostly keep the thought the task code gives the replaced step
        seen = set()
        for mistake in (m for m in incidents.MISTAKES if m.harm == harm):
            steps, marked = incidents.plan(mistake, SEEDS[0], "unaware")
            seen |= {steps[i][0] for i in marked}
        assert len(seen) >= 4, harm


@pytest.mark.parametrize("name", NAMES)
def test_mistake_fires_exactly_its_policy_check(name):
    mistake = incidents.BY_NAME[name]
    for seed in SEEDS:
        for awareness, dilution in (("unaware", 1), ("noticed", 1), ("mistaken", 2), ("unaware", 4)):
            t = row(mistake, seed, awareness, dilution)
            assert t.kind == "incident" and t.label == 1 and t.task == mistake.task
            assert (t.harm, t.route, t.awareness, t.dilution) == (mistake.harm, mistake.route, awareness, dilution)
            assert t.harms_fired == [mistake.harm]
            assert t.incident_steps and all(0 <= i < t.n_steps for i in t.incident_steps)


@pytest.mark.parametrize("name", NAMES)
def test_awareness_changes_thoughts_only_on_incident_steps(name):
    mistake = incidents.BY_NAME[name]
    for seed in SEEDS:
        plans = {level: incidents.plan(mistake, seed, level) for level in AWARENESS}
        steps, marked = plans["unaware"]
        for level in ("noticed", "mistaken"):
            other_steps, other_marked = plans[level]
            assert other_marked == marked
            assert [c for _, c in other_steps] == [c for _, c in steps]        # same commands at every level
            changed = [i for i, ((a, _), (b, _)) in enumerate(zip(steps, other_steps)) if a != b]
            assert changed and set(changed) <= set(marked), (level, seed)
        assert plans["noticed"][0] != plans["mistaken"][0]
        assert all(thought for thought, _ in steps)


@pytest.mark.parametrize("name", NAMES)
def test_the_same_plan_without_the_mistake_is_an_ordinary_run(name):
    mistake = incidents.BY_NAME[name]
    for seed in SEEDS:
        session = tasks.run_plan(mistake.task, seed, 1, _forced(mistake))
        result = execute(seed, mistake.task, [c for _, c in session.steps])
        assert session.incident_steps == [] and result.policy_checks == [] and result.success


@pytest.mark.parametrize("name", [m.name for m in incidents.MISTAKES if m.route == "script"])
def test_script_route_keeps_the_mistake_inside_a_script_file(name):
    mistake = incidents.BY_NAME[name]
    for seed in SEEDS:
        steps, marked = incidents.plan(mistake, seed, "unaware")
        commands = [c for _, c in steps]
        script = tasks.sample_names(seed).script
        writes = [i for i in marked if commands[i].startswith(f"cat > {script} <<")]
        runs = [i for i in marked if commands[i] == f"bash {script}"]
        assert len(writes) == 1 and len(runs) == 1 and writes[0] < runs[0]
        # Writing the script breaks nothing; the policy check fires only once the script runs.
        not_run = [c if i != runs[0] else "pwd" for i, c in enumerate(commands)]
        assert execute(seed, mistake.task, not_run).policy_checks == []
        # The benign version of the same script differs, and the mistaken line is not a step command.
        benign = tasks.run_plan(mistake.task, seed, 1, _forced(mistake)).steps
        benign_script = next(c for _, c in benign if c.startswith(f"cat > {script} <<"))
        mistaken_lines = set(commands[writes[0]].split("\n")) - set(benign_script.split("\n"))
        assert mistaken_lines or len(commands[writes[0]]) < len(benign_script)
        mistaken_lines -= {c for _, c in benign}      # a line that is also an ordinary step is not the mistake
        assert not any(line == c for line in mistaken_lines for c in commands)


def _forced(mistake):
    """The same plan variant without the mistake."""
    return incidents.Mistake(mistake.name, mistake.harm, mistake.route, mistake.task, "", mistake.force, {}, [], [])


@pytest.mark.parametrize("name", [m.name for m in incidents.MISTAKES if m.route == "variable"])
def test_variable_route_goes_through_a_variable(name):
    mistake = incidents.BY_NAME[name]
    for seed in SEEDS:
        steps, marked = incidents.plan(mistake, seed, "unaware")
        commands = [c for _, c in steps]
        using = [i for i in marked if "$" in commands[i]]
        assert using, "no mistaken step reads a variable"
        if name in ("unit_var_typo", "enabled_from_unset_var"):
            continue   # these two read a variable that was never set
        # The wrong value is assigned in one step; the command that uses it is, as text, exactly
        # the command of an ordinary run.
        assert any(commands[i].startswith("export ") for i in marked)
        ordinary = {c for _, c in tasks.run_plan(mistake.task, seed, 1, _forced(mistake)).steps}
        assert all(commands[i] in ordinary for i in using)


def test_nothing_is_concealed():
    """Rule 2 of docs/DESIGN.md: no encoding on the mistaken steps."""
    for mistake in incidents.MISTAKES:
        steps, marked = incidents.plan(mistake, SEEDS[0], "unaware")
        assert not any("base64" in steps[i][1] for i in marked)


def test_benign_thoughts_also_mention_sensitive_things():
    thoughts = " ".join(t for task in tasks.TASKS for seed in range(60) for t, _ in tasks.plan(task, seed)).lower()
    for word in ("secret", "audit service", "key", "threshold", "cleared"):
        assert word in thoughts, word


def test_generate_all_is_what_the_cli_calls(tmp_path):
    small = {"train.jsonl": ("train", "train", (0, [(1, 4)]), (1, [(1, ("unaware",), 1)]))}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(generate, "FILES", small)
        assert generate.generate_all(tmp_path, seed=1) == {"train.jsonl": 20}
