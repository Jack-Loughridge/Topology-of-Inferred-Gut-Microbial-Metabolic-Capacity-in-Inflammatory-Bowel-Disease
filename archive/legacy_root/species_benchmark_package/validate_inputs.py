#!/usr/bin/env python3
from __future__ import annotations

import sys

from species_benchmarks_all_tasks import main

if __name__ == "__main__":
    main([*sys.argv[1:], "--validate-only"])
