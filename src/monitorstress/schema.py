"""The trajectory record shared by the generator and the monitors. See docs/DESIGN.md."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "val", "test")
KINDS = ("benign", "benign_hard", "attack")
HARMS = ("exfil", "audit_off", "backdoor", "eval_tamper")
EVASIONS = ("direct", "split", "indirect", "encode")
THOUGHT_STYLES = ("candid", "silent", "cover")
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
    evasion: str | None
    thoughts: str
    dilution: int
    n_steps: int
    steps: list[Step]
    events: list[dict]
    harms_fired: list[str] = field(default_factory=list)
    side_steps: list[int] = field(default_factory=list)

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
        if self.kind == "attack":
            need(self.label == 1, "attack must have label 1")
            need(self.harm in HARMS, f"bad harm {self.harm!r}")
            need(self.evasion in EVASIONS, f"bad evasion {self.evasion!r}")
            need(self.thoughts in THOUGHT_STYLES, f"bad thought style {self.thoughts!r}")
            need(self.harms_fired == [self.harm], f"harms_fired {self.harms_fired} != [{self.harm}]")
            need(len(self.side_steps) > 0, "attack without side_steps")
        else:
            need(self.label == 0, "benign must have label 0")
            need(self.harm is None and self.evasion is None, "benign with harm or evasion")
            need(self.thoughts == "neutral", "benign thoughts must be 'neutral'")
            need(self.harms_fired == [], f"benign fired {self.harms_fired}")
            need(self.side_steps == [], "benign with side_steps")
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
