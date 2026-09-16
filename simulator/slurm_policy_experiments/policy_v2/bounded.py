"""One persistent wall-clock budget across builds, tests and pilots."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time


class Budget:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "budget.json"
        if not self.path.exists():
            now = time.time()
            record = {"started_epoch": now, "deadline_epoch": now + 10800,
                      "limit_seconds": 10800, "includes": "builds, tests and pilots; intervening elapsed time also counts",
                      "started_utc": dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat()}
            with self.path.open("x") as stream:
                json.dump(record, stream, indent=2)
        self.record = json.loads(self.path.read_text())

    def run(self, name, command, maximum=900, container=None):
        if (self.root / "finished.json").exists():
            raise RuntimeError("This bounded experiment is closed; do not silently restart it")
        remaining = self.record["deadline_epoch"] - time.time() - 30
        if remaining <= 0:
            raise TimeoutError("Aggregate three-hour budget exhausted")
        timeout = min(maximum, remaining)
        logfile = self.root / (name + ".log")
        if logfile.exists():
            raise FileExistsError(f"Preserve previous attempt: {logfile}")
        start = time.time()
        expired = False
        with logfile.open("w") as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except BaseException as exc:
                expired = isinstance(exc, subprocess.TimeoutExpired)
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                if not expired:
                    raise
                code = 124
            finally:
                if container:
                    subprocess.run(["docker", "--context", "colima", "rm", "-f", container],
                                   stdout=stream, stderr=stream, timeout=15, check=False)
        result = {"phase": name, "command": command, "exit_code": code, "timed_out": expired,
                  "elapsed_seconds": time.time()-start, "aggregate_elapsed_seconds": time.time()-self.record["started_epoch"],
                  "allowed_seconds": timeout}
        with (self.root / "phases.jsonl").open("a") as stream:
            stream.write(json.dumps(result)+"\n")
        print(json.dumps(result), flush=True)
        return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("name")
    parser.add_argument("--maximum", type=int, default=900)
    parser.add_argument("--container")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(Budget(args.directory).run(args.name, command, args.maximum, args.container))
