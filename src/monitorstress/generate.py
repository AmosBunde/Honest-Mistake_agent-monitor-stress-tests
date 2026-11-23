"""Dataset builder.

`generate_all(out_dir, seed)` writes train.jsonl, val.jsonl, test.jsonl, dilution.jsonl and
val_dilution.jsonl with the contents listed in docs/DESIGN.md. Every row is built by
`build_trajectory`, which re-executes the commands in a fresh machine made from
(seed, task) and validates the record; labels therefore come from execution.

`build_trajectory` is the one place a trajectory is made: a list of (thought, command)
steps for a task goes in, a validated Trajectory comes out.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from . import incidents, tasks
from .schema import AWARENESS, Step, Trajectory, write_jsonl
from .world import execute

SEEDS_PER_BLOCK = 1_000_000
MAX_ATTEMPTS = 60

EASY_SHARE = 0.15   # share of benign rows from tasks that never host an incident

# file name -> id prefix, split, benign (seed block, [(dilution, rows)]),
#              incidents (seed block, [(dilution, awareness levels, rows per cell and level)]),
#              file whose incidents set the benign task mix
# Seed blocks below 10 give seeds under tasks.HELDOUT_SEED_START (training name pools); val
# and test use blocks from 10 up (the shared held-out pools).
FILES = {
    "train.jsonl": ("train", "train", (0, [(1, 1200)]), (1, [(1, AWARENESS, 25)]), "train.jsonl"),
    "val.jsonl": ("val", "val", (10, [(1, 2000)]), None, "test.jsonl"),
    "test.jsonl": ("test", "test", (11, [(1, 2000)]), (14, [(1, AWARENESS, 40)]), "test.jsonl"),
    "dilution.jsonl": ("dil", "test", (12, [(2, 500), (4, 500)]), (15, [(2, ("unaware",), 30), (4, ("unaware",), 30)]),
                       "dilution.jsonl"),
    "val_dilution.jsonl": ("valdil", "val", (13, [(2, 500), (4, 500)]), None, "dilution.jsonl"),
}


def incident_plan(parts: list) -> list[tuple]:
    """(dilution, awareness, mistake) for every incident row the parts ask for, in file order."""
    out = []
    for dilution, levels, per_cell in parts:
        for harm, route in incidents.CELLS:
            realizations = incidents.cell(harm, route)
            for awareness in levels:
                out += [(dilution, awareness, realizations[k % len(realizations)]) for k in range(per_cell)]
    return out


def benign_tasks(count: int, incident_parts: list, dilution: int, key: str) -> list[str]:
    """The task of each of `count` benign rows.

    Among the tasks that host incidents, each task's share of the benign rows equals its
    share of the incidents (of the same dilution) in the reference file, so that the task
    alone says nothing about the label. The remaining EASY_SHARE of the rows is spread
    evenly over the tasks that never host an incident.
    """
    hosting = Counter(m.task for d, _, m in incident_plan(incident_parts) if d == dilution)
    easy = [name for name in tasks.TASKS if name not in incidents.HOSTING_TASKS]
    n_easy = round(count * EASY_SHARE)
    quotas = {name: n_easy / len(easy) for name in easy}
    total = sum(hosting.values())
    quotas.update({name: (count - n_easy) * hosting[name] / total for name in incidents.HOSTING_TASKS})
    counts = {name: int(q) for name, q in quotas.items()}
    by_remainder = sorted(quotas, key=lambda name: (counts[name] - quotas[name], name))
    for name in by_remainder[:count - sum(counts.values())]:
        counts[name] += 1
    order = [name for name in tasks.TASKS for _ in range(counts.get(name, 0))]
    random.Random(f"mix:{key}").shuffle(order)
    return order


def row_seed(base_seed: int, block: int, index: int) -> int:
    return block * SEEDS_PER_BLOCK + (base_seed * 1000 + index) % SEEDS_PER_BLOCK


def build_trajectory(row_id: str, split: str, seed: int, task: str, steps: list[tuple[str, str]], *,
                     kind: str, dilution: int = 1, harm: str | None = None, route: str | None = None,
                     awareness: str = "neutral", incident_steps: tuple = (),
                     require_success: bool = True) -> Trajectory:
    """Execute `steps` for (seed, task) and return the validated record.

    Outputs, events and harms_fired come from execution, never from the caller. Raises
    ValueError if the record is inconsistent (a benign row on which a policy check fires,
    an incident whose check does not fire alone) or, with require_success, if the task's
    own success check fails.
    """
    result = execute(seed, task, [command for _, command in steps])
    traj = Trajectory(
        id=row_id, split=split, seed=seed, task=task, task_prompt=tasks.task_prompt(seed, task),
        kind=kind, label=0 if kind in ("benign", "benign_hard") else 1, harm=harm, route=route,
        awareness=awareness, dilution=dilution, n_steps=len(steps),
        steps=[Step(thought, command, output) for (thought, command), output in zip(steps, result.outputs)],
        events=result.events, harms_fired=result.policy_checks, incident_steps=list(incident_steps))
    traj.validate()
    if require_success and not result.success:
        raise ValueError(f"{row_id}: task {task!r} did not pass its success check")
    return traj


def benign_row(row_id: str, split: str, seed: int, task: str, dilution: int = 1) -> Trajectory:
    return build_trajectory(row_id, split, seed, task, tasks.plan(task, seed, dilution),
                            kind=tasks.TASKS[task].kind, dilution=dilution)


def incident_row(row_id: str, split: str, seed: int, mistake: incidents.Mistake, awareness: str,
                 dilution: int = 1) -> Trajectory:
    """One incident. Raises ValueError unless execution fires exactly the mistake's policy check."""
    steps, incident_steps = incidents.plan(mistake, seed, awareness, dilution)
    return build_trajectory(row_id, split, seed, mistake.task, steps, kind="incident", dilution=dilution,
                            harm=mistake.harm, route=mistake.route, awareness=awareness,
                            incident_steps=tuple(incident_steps), require_success=False)


def benign_rows(prefix: str, split: str, block: int, parts: list, base_seed: int, mix: list) -> list[Trajectory]:
    """Benign rows whose task mix follows the incidents described by `mix` (see benign_tasks)."""
    rows = []
    for dilution, count in parts:
        for task in benign_tasks(count, mix, dilution, f"{base_seed}:{prefix}:{dilution}"):
            index = len(rows)
            rows.append(benign_row(f"{prefix}-{index:06d}", split, row_seed(base_seed, block, index), task, dilution))
    return rows


def incident_rows(prefix: str, split: str, block: int, parts: list, base_seed: int, start: int = 0) -> list[Trajectory]:
    """The same number of rows in every (violation, route, awareness) cell, cycling through realizations.

    A candidate is kept only if execution gives harms_fired == [harm]; otherwise the next
    seed is tried.
    """
    rows = []
    for dilution, awareness, mistake in incident_plan(parts):
        index = len(rows)
        for attempt in range(MAX_ATTEMPTS):
            seed = row_seed(base_seed, block, index + 7919 * attempt)
            try:
                rows.append(incident_row(f"{prefix}-{start + index:06d}", split, seed, mistake, awareness, dilution))
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(f"no valid incident for {mistake.name} after {MAX_ATTEMPTS} seeds")
    return rows


def file_rows(name: str, base_seed: int, incidents_too: bool = True) -> list[Trajectory]:
    prefix, split, benign, incident, mix_from = FILES[name]
    rows = benign_rows(prefix, split, benign[0], benign[1], base_seed, FILES[mix_from][3][1])
    if incident is not None and incidents_too:
        rows += incident_rows(prefix, split, incident[0], incident[1], base_seed, start=len(rows))
    return rows


def generate_all(out_dir: str | Path, seed: int = 2026) -> dict[str, int]:
    """Write the five data files. Returns the number of rows written per file."""
    return {name: write_jsonl(Path(out_dir) / name, file_rows(name, seed)) for name in FILES}


def generate_benign(out_dir: str | Path, seed: int = 2026) -> dict[str, int]:
    """Write the five files with their benign rows only."""
    return {name: write_jsonl(Path(out_dir) / name, file_rows(name, seed, incidents_too=False)) for name in FILES}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the trajectory files.")
    parser.add_argument("--out", default="data/")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for name, count in generate_all(args.out, args.seed).items():
        print(f"{name}: {count} rows")


if __name__ == "__main__":
    main()
