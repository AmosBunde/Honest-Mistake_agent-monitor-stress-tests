"""Command line: monitorstress generate | run | bench-llm | figures."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _cmd_generate(args: argparse.Namespace) -> int:
    from . import generate  # written by the generator half; imported only when asked for

    generate.generate_all(args.out, seed=args.seed)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from . import experiments
    from .monitors.llm import DEFAULT_MODELS

    chosen = tuple(e.strip().upper() for e in args.experiments.split(",")) if args.experiments else experiments.EXPERIMENTS
    opts = experiments.Options(
        data_dir=args.data,
        out_dir=args.out,
        experiments=chosen,
        skip_llm=args.skip_llm,
        only_llm=args.only_llm,
        llm_models=tuple(args.llm_models) if args.llm_models else DEFAULT_MODELS,
        llm_n_benign=args.llm_n_benign,
        llm_n_per_cell=args.llm_n_per_cell,
        llm_n_benign_dilution=args.llm_n_benign_dilution,
        llm_max_tokens=args.llm_max_tokens,
        llm_threads=args.llm_threads,
        embed_cache_dir=args.embed_cache,
        allow_artifact_leak=args.allow_artifact_leak,
        seed=args.seed,
    )
    summary = experiments.run(opts)
    print(json.dumps({p["id"]: p["verdict"] for p in summary["predictions"]}, indent=1))
    return 0


def _cmd_bench_llm(args: argparse.Namespace) -> int:
    from .monitors.llm import DEFAULT_MODELS, benchmark
    from .schema import read_jsonl

    pool = read_jsonl(args.data)
    results = []
    for model in args.llm_models or DEFAULT_MODELS:  # one model at a time
        result = benchmark(model, pool, view=args.view, token_lengths=tuple(args.tokens),
                           n_per_length=args.n, num_threads=args.llm_threads, seed=args.seed)
        results.append(result)
        for row in result["by_length"]:
            print(f"{result['monitor']:<20} {row['mean_prompt_tokens']:>7.0f} tokens  "
                  f"{row['seconds_per_trajectory']:>7.2f} s/trajectory  (n={row['n']}, {result['num_threads']} threads)",
                  flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
    return 0


def _cmd_figures(args: argparse.Namespace) -> int:
    from . import figures

    for path in figures.make_all(args.results, args.out):
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="monitorstress", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="write the data files")
    p.add_argument("--out", default="data/")
    p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("run", help="run experiments E0 to E7 and write results/")
    p.add_argument("--data", default="data/")
    p.add_argument("--out", default="results/")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--skip-llm", action="store_true", help="everything except the language-model monitors")
    group.add_argument("--only-llm", action="store_true", help="only the language-model monitors")
    p.add_argument("--experiments", default=None, help="comma-separated, for example E1,E2 (default: all)")
    p.add_argument("--llm-models", nargs="+", default=None, help="Hugging Face model names (default: the three in PLAN.md)")
    p.add_argument("--llm-n-benign", type=int, default=None,
                   help="benign trajectories in the language-model subsample, for validation and for test")
    p.add_argument("--llm-n-per-cell", type=int, default=None, help="incidents per cell in the language-model subsample")
    p.add_argument("--llm-n-benign-dilution", type=int, default=None,
                   help="benign trajectories per dilution level in the subsample (default: half of --llm-n-benign)")
    p.add_argument("--llm-max-tokens", type=int, default=1024, help="prompt budget; longer trajectories are cut in the middle")
    p.add_argument("--llm-threads", type=int, default=None, help="torch threads (default: physical cores)")
    p.add_argument("--embed-cache", default=None, help="directory for the embedding cache")
    p.add_argument("--allow-artifact-leak", action="store_true", help="continue even if the E0 artifact check fails")
    p.add_argument("--seed", type=int, default=2026, help="seed for subsampling")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("bench-llm", help="time the language-model monitors by prompt length")
    p.add_argument("--data", default="data/test.jsonl", help="a .jsonl file to draw steps from")
    p.add_argument("--llm-models", nargs="+", default=None)
    p.add_argument("--view", default="commands+thoughts", choices=["commands", "commands+thoughts"])
    p.add_argument("--tokens", type=int, nargs="+", default=[300, 600, 1200])
    p.add_argument("--n", type=int, default=6, help="trajectories per length")
    p.add_argument("--llm-threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the timings to this JSON file")
    p.set_defaults(func=_cmd_bench_llm)

    p = sub.add_parser("figures", help="draw the figures from results/*.json")
    p.add_argument("--results", default="results/")
    p.add_argument("--out", default="docs/img/")
    p.set_defaults(func=_cmd_figures)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models come from the local cache only
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
