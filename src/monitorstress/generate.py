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
from pathlib import Path

from . import incidents, tasks
from .schema import AWARENESS, Step, Trajectory, write_jsonl
from .world import execute

SEEDS_PER_BLOCK = 1_000_000
MAX_ATTEMPTS = 60

# file name -> id prefix, split, benign (seed block, [(dilution, rows)]),
#              incidents (seed block, [(dilution, awareness levels, rows per cell and level)])
# Seed blocks below 10 give seeds under tasks.HELDOUT_SEED_START (training name pools); val
# and test use blocks from 10 up (the shared held-out pools).
FILES = {
    "train.jsonl": ("train", "train", (0, [(1, 1200)]), (1, [(1, AWARENESS, 25)])),
    "val.jsonl": ("val", "val", (10, [(1, 2000)]), None),
    "test.jsonl": ("test", "test", (11, [(1, 2000)]), (14, [(1, AWARENESS, 40)])),
    "dilution.jsonl": ("dil", "test", (12, [(2, 500), (4, 500)]), (15, [(2, ("unaware",), 30), (4, ("unaware",), 30)])),
    "val_dilution.jsonl": ("valdil", "val", (13, [(2, 500), (4, 500)]), None),
}


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


def benign_rows(prefix: str, split: str, block: int, parts: list, base_seed: int, start: int = 0) -> list[Trajectory]:
    """Half ordinary, half hard benign, cycling through the tasks of each kind."""
    rows = []
    for dilution, count in parts:
        for i in range(count):
            index = len(rows)
            names = tasks.HARD_TASKS if i % 2 else tasks.BENIGN_TASKS
            task = names[(i // 2) % len(names)]
            rows.append(benign_row(f"{prefix}-{start + index:06d}", split, row_seed(base_seed, block, index), task, dilution))
    return rows


def incident_rows(prefix: str, split: str, block: int, parts: list, base_seed: int, start: int = 0) -> list[Trajectory]:
    """The same number of rows in every (violation, route, awareness) cell, cycling through realizations.

    A candidate is kept only if execution gives harms_fired == [harm]; otherwise the next
    seed is tried.
    """
    rows = []
    for dilution, levels, per_cell in parts:
        for harm, route in incidents.CELLS:
            realizations = incidents.cell(harm, route)
            for awareness in levels:
                for k in range(per_cell):
                    index = len(rows)
                    mistake = realizations[k % len(realizations)]
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


def file_rows(name: str, base_seed: int) -> list[Trajectory]:
    prefix, split, benign, incident = FILES[name]
    rows = benign_rows(prefix, split, benign[0], benign[1], base_seed)
    if incident is not None:
        rows += incident_rows(prefix, split, incident[0], incident[1], base_seed, start=len(rows))
    return rows


def generate_all(out_dir: str | Path, seed: int = 2026) -> dict[str, int]:
    """Write the five data files. Returns the number of rows written per file."""
    return {name: write_jsonl(Path(out_dir) / name, file_rows(name, seed)) for name in FILES}


def generate_benign(out_dir: str | Path, seed: int = 2026) -> dict[str, int]:
    """Write the five files with their benign rows only."""
    counts = {}
    for name, (prefix, split, benign, _) in FILES.items():
        counts[name] = write_jsonl(Path(out_dir) / name, benign_rows(prefix, split, benign[0], benign[1], seed))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the trajectory files.")
    parser.add_argument("--out", default="data/")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for name, count in generate_all(args.out, args.seed).items():
        print(f"{name}: {count} rows")


if __name__ == "__main__":
    main()
