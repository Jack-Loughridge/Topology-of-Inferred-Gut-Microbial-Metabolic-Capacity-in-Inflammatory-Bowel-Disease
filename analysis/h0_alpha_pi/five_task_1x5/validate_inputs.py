#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h0_alpha_pi_1x5.orchestrator import default_config, validate_all


def main() -> None:
    defaults = default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=defaults["base"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    args = parser.parse_args()
    paths = default_config(args.base)
    paths["output_dir"] = args.output_dir
    audit = validate_all(paths)
    print("=" * 112)
    print("STANDALONE H0 ALPHA-PI ALL-TASK 1x5 PREFLIGHT")
    print("=" * 112)
    print(audit.to_string(index=False))
    print("Existing corrected IBD output will be resumed only until repetition 1 reaches 5/5 folds.")
    print("Remaining four tasks use the exact repetition-1 task manifests shared with Ricci, benchmarks and H0+Ricci.")
    print("=" * 112)
    print("STANDALONE H0 ALPHA-PI ALL-TASK 1x5 PREFLIGHT: PASSED")
    print("=" * 112)


if __name__ == "__main__":
    main()
