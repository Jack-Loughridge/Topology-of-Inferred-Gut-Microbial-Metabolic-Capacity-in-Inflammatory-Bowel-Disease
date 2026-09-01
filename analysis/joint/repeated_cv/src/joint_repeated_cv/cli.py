from __future__ import annotations

from .config import build_parser, config_from_args
from .orchestrator import run


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(config_from_args(args), args)
