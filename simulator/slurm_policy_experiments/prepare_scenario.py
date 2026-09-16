#!/usr/bin/env python3
"""Copy the audited working Slurm configuration into a generated scenario."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = ROOT / "config" / "stanage_2025_assumed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()

    args.scenario.mkdir(parents=True, exist_ok=True)
    required = ["slurm.conf", "gres.conf", "sim.conf"]
    for name in required:
        source = args.template / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, args.scenario / name)
    if not (args.scenario / "sim.events").exists():
        raise FileNotFoundError(f"{args.scenario / 'sim.events'} must be generated first")
    if not (args.scenario / "users.sim").exists():
        raise FileNotFoundError(f"{args.scenario / 'users.sim'} must be generated first")
    print(f"Prepared {args.scenario.resolve()}")


if __name__ == "__main__":
    main()
