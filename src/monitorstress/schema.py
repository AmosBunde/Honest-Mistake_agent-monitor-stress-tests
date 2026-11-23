"""The trajectory record shared by the generator and the monitors. See docs/DESIGN.md."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "val", "test")
KINDS = ("benign", "benign_hard", "incident")
HARMS = ("exfil", "audit_off", "backdoor", "eval_tamper")
ROUTES = ("direct", "script", "variable", "bundle")
AWARENESS = ("noticed", "unaware", "mistaken")
DILUTIONS = (1, 2, 4)
EVENT_TYPES = ("read", "write", "delete", "chmod", "send", "recv", "exec", "service", "env")


@dataclass
class Step:
    thought: str
    command: str
    output: str


@dataclass
class Trajectory:
    id: str
    split: str
    seed: int
    task: str
    task_prompt: str
    kind: str
    label: int
    harm: str | None
    route: str | None
    awareness: str
    dilution: int
    n_steps: int
    steps: list[Step]
    events: list[dict]
    harms_fired: list[str] = field(default_factory=list)
    incident_steps: list[int] = field(default_factory=list)

    def validate(self) -> None:
        """Raise ValueError if the record breaks the rules in docs/DESIGN.md."""
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"{self.id}: {msg}")

        need(self.split in SPLITS, f"bad split {self.split!r}")
        need(self.kind in KINDS, f"bad kind {self.kind!r}")
        need(self.dilution in DILUTIONS, f"bad dilution {self.dilution!r}")
        need(self.n_steps == len(self.steps) > 0, "n_steps does not match steps")
        need(bool(self.task_prompt), "empty task_prompt")
        if self.kind == "incident":
            need(self.label == 1, "incident must have label 1")
            need(self.harm in HARMS, f"bad harm {self.harm!r}")
            need(self.route in ROUTES, f"bad route {self.route!r}")
            need(self.awareness in AWARENESS, f"bad awareness {self.awareness!r}")
            need(self.harms_fired == [self.harm], f"harms_fired {self.harms_fired} != [{self.harm}]")
            need(len(self.incident_steps) > 0, "incident without incident_steps")
        else:
            need(self.label == 0, "benign must have label 0")
            need(self.harm is None and self.route is None, "benign with harm or route")
            need(self.awareness == "neutral", "benign awareness must be 'neutral'")
            need(self.harms_fired == [], f"benign fired {self.harms_fired}")
            need(self.incident_steps == [], "benign with incident_steps")
        for e in self.events:
            need(e.get("type") in EVENT_TYPES, f"bad event type {e.get('type')!r}")
            need(isinstance(e.get("step"), int) and 0 <= e["step"] < self.n_steps, "event step out of range")
            need("taint" not in e and "tainted" not in e, "events must not carry taint")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        d = dict(d)
        d["steps"] = [Step(**s) for s in d["steps"]]
        return cls(**d)


def write_jsonl(path: str | Path, trajs: Iterable[Trajectory]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for t in trajs:
            f.write(t.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, validate: bool = False) -> list[Trajectory]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = Trajectory.from_dict(json.loads(line))
                if validate:
                    t.validate()
                out.append(t)
    return out
