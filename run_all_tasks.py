#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h0_alpha_pi_1x5.orchestrator import default_config, run_all


def parser() -> argparse.ArgumentParser:
    defaults = default_config()
    output = argparse.ArgumentParser(description="Run one locked 5-fold repetition of standalone H0 Alpha-Pi for all five tasks.")
    output.add_argument("--base", type=Path, default=defaults["base"])
    output.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    output.add_argument("--epochs", type=int, default=70)
    output.add_argument("--batch-size", type=int, default=32)
    output.add_argument("--device", type=str, default=None)
    output.add_argument("--print-every", type=int, default=10)
    output.add_argument("--no-resume", action="store_true")
    return output


def main() -> None:
    args = parser().parse_args()
    paths = default_config(args.base)
    paths["output_dir"] = args.output_dir
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "print_every": args.print_every,
        "resume": not args.no_resume,
    }
    if args.device:
        overrides["device"] = args.device
    run_all(paths, **overrides)


if __name__ == "__main__":
    main()
