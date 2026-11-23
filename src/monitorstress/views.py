"""What a monitor is shown. Monitors read task_prompt, steps and events, nothing else."""
from __future__ import annotations

from .schema import Trajectory

TEXT_VIEWS = ("commands", "commands+outputs", "commands+thoughts")
VIEWS = TEXT_VIEWS + ("events",)
MAX_OUTPUT_CHARS = 200


def render_steps(traj: Trajectory, view: str) -> list[str]:
    """One string per step, without the task prompt."""
    if view not in TEXT_VIEWS:
        raise ValueError(f"not a text view: {view!r}")
    out = []
    for s in traj.steps:
        parts = []
        if view == "commands+thoughts" and s.thought:
            parts.append(f"# {s.thought}")
        parts.append(f"$ {s.command}")
        if view == "commands+outputs" and s.output:
            parts.append(s.output[:MAX_OUTPUT_CHARS])
        out.append("\n".join(parts))
    return out


def render(traj: Trajectory, view: str) -> str:
    """The task prompt, then the steps."""
    return f"TASK: {traj.task_prompt}\n" + "\n".join(render_steps(traj, view))
