"""Bounded admission control from arrived jobs and observed Slurm state.

The fluid backlog is a heuristic, not a replacement for Slurm placement.
Carbon input in the pilot is retrospective, not an as-issued forecast.
"""
from bisect import bisect_right
from collections import defaultdict
import json
import math
import sys

try:
    from .selection import choose
except ImportError:
    from selection import choose


class Signal:
    def __init__(self, points, end):
        self.points = sorted((float(t), float(v)) for t, v in points)
        self.times = [x[0] for x in self.points]
        self.end = float(end)
        if not self.points or len(set(self.times)) != len(self.times) or self.end <= self.times[-1]:
            raise ValueError("Invalid carbon coverage")
        if any(not math.isfinite(v) or v < 0 for _, v in self.points):
            raise ValueError("Invalid intensity")

    def mean(self, start, duration):
        end = start + duration
        if duration <= 0 or start < self.times[0] or end > self.end:
            return None
        total, cursor = 0.0, start
        while cursor < end:
            i = bisect_right(self.times, cursor)-1
            stop = min(end, self.times[i+1] if i+1 < len(self.times) else self.end)
            total += (stop-cursor)*self.points[i][1]
            cursor = stop
        return total/duration


class Controller:
    def __init__(self, config):
        self.config = config
        self.signal = Signal(config["carbon_points"], config["carbon_end_s"])
        self.jobs = config["jobs"]
        self.held, self.seen, self.released = {}, set(), set()
        self.spent, self.reserved = defaultdict(float), defaultdict(float)
        self.records = []
        self.last_now = -math.inf

    def log(self, now, job, action, reason, **extra):
        self.records.append({"now_s": now, "sim_job_id": job, "action": action, "reason": reason, **extra})

    def pressure(self, job, states, now):
        """Capacity-seconds in this resource pool, using historical durations only."""
        pool = self.jobs[job]["pool"]
        work, load, pending = 0.0, 0.0, False
        for item in states:
            metadata = self.jobs[item["id"]]
            if metadata["pool"] != pool:
                continue
            fraction = metadata["fraction"]
            if item["state"] == "running":
                elapsed = max(now-item["start_s"], 0)
                remaining = max(metadata["predicted_runtime_s"]-elapsed, self.config["tick_s"])
                work += fraction*remaining
                load += fraction
            elif item["state"] == "pending":
                work += fraction*metadata["predicted_runtime_s"]
                pending = True
        return work if pending or load+self.jobs[job]["fraction"] > 1 else 0.0

    def option(self, job, now, release, backlog):
        m = self.jobs[job]
        runtime = m["predicted_runtime_s"]
        # Existing backlog drains during a hold; unseen arrivals are not predicted.
        baseline_start = now+backlog
        predicted_start = max(release, baseline_start)
        before = self.signal.mean(baseline_start, runtime)
        after = self.signal.mean(predicted_start, runtime)
        gain = (before-after)/before*100 if before and after is not None else -math.inf
        weight = m["fraction"]*runtime/3600
        proxy = (before-after)*weight if before is not None and after is not None else -math.inf
        return {"release": release, "gain_pct": gain, "saving_proxy": proxy,
                "utility": proxy/max(release-now, 1), "predicted_start_s": predicted_start,
                "incremental_wait_s": predicted_start-baseline_start}

    def release(self, job, now, reason, output):
        hold = self.held.pop(job, None)
        if hold:
            user = self.jobs[job]["user"]
            self.reserved[user] -= hold["reserved_s"]
            spent = max(now-hold["arrived_s"], 0)
            self.spent[user] += spent
        else:
            spent = 0.0
        if job in self.released:
            raise ValueError("Duplicate release")
        self.released.add(job)
        output.append(job)
        self.log(now, job, "release", reason, actual_admission_wait_s=spent)

    def step(self, request):
        now = float(request["now_s"])
        if now < self.last_now:
            raise ValueError("Clock moved backwards")
        self.last_now = now
        states = request.get("states", [])
        if any(s["id"] not in self.released for s in states):
            raise ValueError("Snapshot exposes unsubmitted job")
        output, candidates = [], []
        # Reconsider only outstanding holds. Never retract a job from Slurm.
        for job, hold in list(self.held.items()):
            if now >= hold["release_s"]:
                self.release(job, now, "planned_release_or_deadline", output)
                continue
            backlog = self.pressure(job, states, now)
            option = self.option(job, now, hold["release_s"], backlog)
            if option["gain_pct"] < self.config["minimum_saving_pct"]:
                self.release(job, now, "feedback_cancel_benefit_lost", output)
            else:
                self.log(now, job, "keep_hold", "feedback_rechecked", backlog_s=backlog, **option)
        for job, arrived in request.get("arrivals", []):
            if job in self.seen:
                raise ValueError("Duplicate arrival")
            self.seen.add(job)
            m = self.jobs[job]
            if self.config.get("baseline") or not m["flexible"]:
                self.release(job, now, "baseline_or_protected", output)
                continue
            if not m["prediction_reliable"]:
                self.release(job, now, "runtime_prediction_unreliable", output)
                continue
            allowance = min(self.config["max_delay_s"], m["predicted_runtime_s"]*self.config["runtime_delay_ratio"])
            deadline = arrived+allowance
            backlog = self.pressure(job, states, now)
            releases = [arrived+i*self.config["tick_s"] for i in range(1, int(allowance//self.config["tick_s"])+1)
                        if arrived+i*self.config["tick_s"] > now]
            options = [self.option(job, now, value, backlog) for value in releases]
            best, audit = choose(options, [("insufficient_carbon_gain", lambda x: x["gain_pct"] >= self.config["minimum_saving_pct"]),
                                          ("no_effect_on_predicted_start", lambda x: x["incremental_wait_s"] > 0)])
            for row in audit:
                self.log(now, job, "candidate", row["rejection_reason"],
                         **{key:value for key,value in row.items() if key!="rejection_reason"})
            if best is None:
                self.release(job, now, "no_feasible_candidate", output)
            else:
                feasible_indices={row["candidate_index"] for row in audit
                                  if row["accepted"] or row["rejection_reason"]=="feasible_lower_rank"}
                candidates.extend((job,arrived,option) for i,option in enumerate(options) if i in feasible_indices)
        # Same-time candidates compete for a user's remaining admission-wait budget.
        candidates.sort(key=lambda x: (-x[2]["utility"], x[2]["release"], x[0]))
        for job, arrived, option in candidates:
            if job in self.held:
                continue
            user = self.jobs[job]["user"]
            reserved = option["release"]-arrived
            if self.spent[user]+self.reserved[user]+reserved > self.config["user_budget_s"]+1e-8:
                self.log(now,job,"budget_candidate_reject","user_budget_exhausted",**option)
            else:
                self.held[job] = {"arrived_s": arrived, "release_s": option["release"], "reserved_s": reserved}
                self.reserved[user] += reserved
                self.log(now, job, "hold", "budget_ranked_accept", user=user, reserved_s=reserved, **option)
        for job in dict.fromkeys(item[0] for item in candidates):
            if job not in self.held:
                self.release(job,now,"user_budget_exhausted",output)
        wake = min((h["release_s"] for h in self.held.values()), default=now+86400)
        if self.held:
            wake = min(wake, now+self.config["tick_s"])
        return {"release": output, "next_wake_s": wake}


def serve(path):
    with open(path) as stream:
        config = json.load(stream)
    controller = Controller(config)
    with open(config["audit_path"], "w", buffering=1) as audit:
        for line in sys.stdin:
            request = json.loads(line)
            result = controller.step(request)
            for row in controller.records:
                audit.write(json.dumps(row, allow_nan=False)+"\n")
            controller.records.clear()
            numbers = [str(int(x.removeprefix("sim_"))) for x in result["release"]]
            print(int(result["next_wake_s"]*1000000), len(numbers), *numbers, flush=True)


if __name__ == "__main__":
    serve(sys.argv[1])
