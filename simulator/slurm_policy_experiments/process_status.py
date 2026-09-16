"""Keep simulator exit evidence separate from job and timing validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def simulator_command(timeout_seconds: int) -> str:
    if timeout_seconds <= 0:
        raise ValueError('Timeout must be positive')
    return (
        'cd /opt/slurm-sim/etc || exit 125; '
        f'stdbuf -oL -eL timeout -s KILL {int(timeout_seconds)} slurmctld -D -i >/dev/null 2>&1; '
        'code=$?; printf "%s\\n" "$code" > simulator_exit_code.txt; exit "$code"'
    )


def record_process_status(directory: Path, docker_exit_code: int | None = None) -> dict:
    path = directory / 'simulator_exit_code.txt'
    simulator_exit = None
    if path.exists():
        value = path.read_text().strip()
        if not value.isdecimal() or not 0 <= int(value) <= 255:
            raise ValueError(f'Invalid simulator exit status: {path}')
        simulator_exit = int(value)
    if simulator_exit is None:
        status = 'simulator_exit_unknown'
    elif simulator_exit != 0 or docker_exit_code not in (None, 0):
        status = 'abnormal_exit'
    else:
        status = 'exited_zero'
    record = {
        'simulator_exit_code': simulator_exit,
        'docker_exit_code': docker_exit_code,
        'process_exit_status': status,
        'job_completion_status': 'see simulation_validation.json',
        'timing_status': 'see simulation_validation.json; not inferred from exit status',
    }
    (directory / 'process_status.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    command = commands.add_parser('command')
    command.add_argument('timeout_seconds', type=int)
    record = commands.add_parser('record')
    record.add_argument('directory', type=Path)
    record.add_argument('--docker-exit-code', type=int)
    args = parser.parse_args()
    if args.action == 'command':
        print(simulator_command(args.timeout_seconds))
    else:
        print(json.dumps(record_process_status(args.directory, args.docker_exit_code)))


if __name__ == '__main__':
    main()
