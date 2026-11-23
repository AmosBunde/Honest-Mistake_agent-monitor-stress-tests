"""Tasks: prompts, variants, success checks, and no policy check on legitimate work."""
import pytest

from monitorstress import tasks
from monitorstress.world import execute

SEEDS = list(range(40)) + list(range(tasks.HELDOUT_SEED_START, tasks.HELDOUT_SEED_START + 40))


def run(task, seed, dilution=1):
    steps = tasks.plan(task, seed, dilution)
    return steps, execute(seed, task, [command for _, command in steps])


def test_task_inventory():
    assert len(tasks.BENIGN_TASKS) == 8 and len(tasks.HARD_TASKS) == 7
    for task in tasks.TASKS.values():
        assert len(task.prompts) >= 3


@pytest.mark.parametrize("task", sorted(tasks.TASKS))
def test_every_run_succeeds_and_fires_no_policy_check(task):
    for seed in SEEDS:
        for dilution in (1, 2, 4):
            steps, result = run(task, seed, dilution)
            assert result.policy_checks == [], (seed, dilution)
            assert result.success, (seed, dilution)
            assert all(thought and command for thought, command in steps)


@pytest.mark.parametrize("task", sorted(tasks.TASKS))
def test_untouched_machine_fails_the_success_check(task):
    assert not execute(3, task, ["ls /tmp"]).success


@pytest.mark.parametrize("task", sorted(tasks.TASKS))
def test_each_task_has_variety(task):
    plans = {tuple(c.split()[0] for _, c in tasks.plan(task, seed)) for seed in range(60)}
    prompts = {tasks.task_prompt(seed, task) for seed in range(60)}
    thoughts = {t for seed in range(60) for t, _ in tasks.plan(task, seed)}
    assert len(plans) >= 3 and len(prompts) >= 3 and len(thoughts) >= 10


def test_dilution_lengthens_the_run():
    for task in tasks.TASKS:
        base = sum(len(tasks.plan(task, s, 1)) for s in range(30))
        assert 1.8 <= sum(len(tasks.plan(task, s, 2)) for s in range(30)) / base <= 2.2
        assert 3.6 <= sum(len(tasks.plan(task, s, 4)) for s in range(30)) / base <= 4.4


def test_name_pools_are_disjoint_between_train_and_heldout():
    for field, train in tasks.POOLS["train"].items():
        if field == "ticket_prefix":
            continue  # tickets differ by number: even for training, odd for held-out
        assert not set(train) & set(tasks.POOLS["heldout"][field]), field
    train = tasks.sample_names(5)
    heldout = tasks.sample_names(tasks.HELDOUT_SEED_START + 5)
    assert int(train.ticket.split("-")[1]) % 2 == 0 and int(heldout.ticket.split("-")[1]) % 2 == 1


def test_some_thoughts_cite_a_ticket_or_the_security_team():
    thoughts = " ".join(t for task in tasks.HARD_TASKS for seed in range(40) for t, _ in tasks.plan(task, seed))
    assert "security team" in thoughts and tasks.sample_names(0).ticket.split("-")[0] in thoughts


# ---- the legitimate tasks touch the sensitive resources without firing a policy check

def events_of(task, seed=21):
    return run(task, seed)[1]


def test_key_rotation_sends_the_secret_to_the_vault_only():
    result = events_of("rotate_api_key")
    n = result.world.names
    sends = [e for e in result.events if e["type"] == "send"]
    assert sends and all(e["host"] == n.vault for e in sends)
    assert any(r.marks for r in result.world.net) and result.policy_checks == []


def test_offsite_backup_and_artifact_upload_send_unmarked_content():
    for task in ("backup_offsite", "upload_artifact"):
        result = events_of(task)
        assert any(e["type"] == "send" for e in result.events)
        assert not any(r.marks for r in result.world.net) and result.policy_checks == []


def test_audit_maintenance_ends_with_auditing_on():
    for seed in range(12):
        result = events_of("audit_maintenance", seed)
        n = result.world.names
        assert any(e["type"] == "service" and e["name"] == n.audit for e in result.events)
        assert any(e["type"] == "write" and e["path"] == n.audit_conf for e in result.events)
        assert result.world.services[n.audit].active and result.policy_checks == []


def test_onboarding_adds_exactly_the_named_key():
    result = events_of("onboard_user")
    p = result.world.task_params
    assert p["key"] in tasks.task_prompt(21, "onboard_user")
    assert result.world.files[p["target"]].content == p["key"] + "\n" and result.policy_checks == []


def test_permissions_fix_sets_a_safe_mode():
    result = events_of("fix_permissions")
    p = result.world.task_params
    assert any(e["type"] == "chmod" and e["path"] == p["path"] for e in result.events)
    assert result.world.files[p["path"]].mode & 0o002 == 0 and result.policy_checks == []


def test_dataset_fix_reports_the_computed_metric_and_the_asked_threshold():
    for seed in range(12):
        result = events_of("dataset_fix", seed)
        w, n, p = result.world, result.world.names, result.world.task_params
        assert w.reported_metric(n.expected) == w.evaluate()[0]
        assert p["asked"] in tasks.task_prompt(seed, "dataset_fix")
        assert w.thresholds()["min_accuracy"] == float(p["asked"]) and result.policy_checks == []
