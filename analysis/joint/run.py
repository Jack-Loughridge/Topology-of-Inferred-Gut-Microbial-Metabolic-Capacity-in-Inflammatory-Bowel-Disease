#!/usr/bin/env python3
"""Canonical joint entry point: execute the unchanged regularised v8 runner.

Pass --help for its actual interface. This wrapper does not change its parser,
settings, numerical algorithm, or required production provenance.
"""
from pathlib import Path
import os
import sys

if __name__ == '__main__':
    runner = Path(__file__).resolve().parent / 'regularised_v8' / 'run.py'
    os.execv(sys.executable, [sys.executable, str(runner), *sys.argv[1:]])
