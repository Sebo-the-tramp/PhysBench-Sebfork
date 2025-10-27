#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# =========================
# Config
# =========================
MODELS_FULL: List[str] = [
#    "instructblip-flan-t5-xl",
#    "instructblip-flan-t5-xxl",
#    "instructblip-vicuna-7b",
    # "instructblip-vicuna-13b",
#    "blip2-flant5xxl",
    "llava-1.5-7b-hf",
    # "llava-1.5-13b-hf",
    "llava-v1.6-mistral-7b-hf",
    "llava-v1.6-vicuna-7b-hf",
#    "deepseek1B",
#    "deepseek7B",
#    "Xinyuan-VL-2B",
#    "Aquila-VL-2B",
    # "Phi-3-vision-128k-instruct",
    "MiniCPM-V2",
    "MiniCPM-V2.5",
#    "MiniCPM-V2.6",
    "Qwen-VL-Chat",
    "InternVL-Chat-V1-5-quantable",
#    "cambrian-8b",
#    "paligemma2-3b",
#    "paligemma2-10b",
    "MolmoE-1B",
    "MolmoE-7B-O",
]
# Models that should reserve 2 GPUs. Omit for 1 GPU.
MODEL_GPUS: Dict[str, int] = {
}

# Optional per‑model extra args to pass to eval/test_benchmark.py
MODEL_EXTRA_ARGS: Dict[str, List[str]] = {
    # "LLaVA-NeXT-Video-7B-hf": ["--some_flag", "value"],
}

AVAILABLE_GPUS: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_CONCURRENT_JOBS: int = 8

PY_SCRIPT = os.path.join("eval", "test_benchmark.py")
DATASET_PATH = "/mnt/proj1/eu-25-92/tiny_vqa_creation/output"
SPLIT = "val"
APPEND_PYTHONPATH = "./"  # prepends to existing PYTHONPATH if present

# Set to a directory string to keep per‑job logs, or None to stream to console
LOG_DIR: Optional[str] = None

# Polling sleep while waiting for jobs or resources
SLEEP_SECS = 1.0


@dataclass
class Job:
    model: str
    gpus_needed: int = 1
    extra_args: List[str] = field(default_factory=list)


@dataclass
class Running:
    popen: subprocess.Popen
    job: Job
    gpus: List[int]
    start_ts: float


def make_jobs(models: List[str]) -> List[Job]:
    jobs = []
    for m in models:
        jobs.append(Job(
            model=m,
            gpus_needed=MODEL_GPUS.get(m, 1),
            extra_args=MODEL_EXTRA_ARGS.get(m, []),
        ))
    # Simple heuristic to reduce fragmentation: try bigger jobs first
    jobs.sort(key=lambda j: (-j.gpus_needed, j.model))
    return jobs


def allocate_gpus(free_q: deque, need: int) -> Optional[List[int]]:
    if len(free_q) < need:
        return None
    return [free_q.popleft() for _ in range(need)]


def free_gpus(free_q: deque, gpus: List[int]) -> None:
    for g in gpus:
        free_q.append(g)


def launch(job: Job, gpus: List[int]) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    if APPEND_PYTHONPATH:
        env["PYTHONPATH"] = (
            APPEND_PYTHONPATH
            + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        )

    cmd = [
        sys.executable,
        PY_SCRIPT,
        "--model_name", job.model,
        "--dataset_path", DATASET_PATH,
        "--split", SPLIT,
        *job.extra_args,
    ]

    if LOG_DIR:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(
            LOG_DIR, f"{job.model.replace('/', '_')}_{'-'.join(map(str, gpus))}.log"
        )
        log_fh = open(log_path, "wb")  # closed when process ends due to inheritance
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Launching {job.model} on GPU(s) {gpus}  log={log_path}", flush=True)
        return proc
    else:
        print(f"Launching {job.model} on GPU(s) {gpus}", flush=True)
        proc = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,
        )
        return proc


def main() -> int:
    free_q = deque(AVAILABLE_GPUS)
    jobs = make_jobs(MODELS_FULL)
    running: Dict[int, Running] = {}
    results = []

    shutdown = {"flag": False}

    def on_signal(sig, frame):
        print(f"Received signal {sig}. Stopping new launches and terminating children...", flush=True)
        shutdown["flag"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    start_all = time.time()

    # Main scheduler loop
    while (jobs or running) and not shutdown["flag"]:
        # Try to launch as many as we can
        launched_any = False
        while jobs and len(running) < MAX_CONCURRENT_JOBS:
            # Find the first job that fits the current free GPU pool
            fit_idx = None
            for idx, j in enumerate(jobs):
                if len(free_q) >= j.gpus_needed:
                    fit_idx = idx
                    break
            if fit_idx is None:
                break  # nothing fits right now

            job = jobs.pop(fit_idx)
            gpus = allocate_gpus(free_q, job.gpus_needed)
            assert gpus is not None
            proc = launch(job, gpus)
            running[proc.pid] = Running(
                popen=proc, job=job, gpus=gpus, start_ts=time.time()
            )
            launched_any = True

        # Harvest finished
        done_pids = []
        for pid, r in list(running.items()):
            rc = r.popen.poll()
            if rc is not None:
                elapsed = int(time.time() - r.start_ts)
                print(f"Finished {r.job.model} on GPU(s) {r.gpus}  code={rc}  time={elapsed}s", flush=True)
                free_gpus(free_q, r.gpus)
                results.append((r.job.model, r.gpus, rc, elapsed))
                done_pids.append(pid)
        for pid in done_pids:
            running.pop(pid, None)

        if not launched_any and not done_pids:
            time.sleep(SLEEP_SECS)

    # If we were interrupted, try to terminate children
    if shutdown["flag"] and running:
        for r in running.values():
            try:
                os.killpg(os.getpgid(r.popen.pid), signal.SIGTERM)
            except Exception:
                pass
        # Give them a moment, then force kill leftovers
        time.sleep(2)
        for r in running.values():
            if r.popen.poll() is None:
                try:
                    os.killpg(os.getpgid(r.popen.pid), signal.SIGKILL)
                except Exception:
                    pass

    # Final drain to collect exit codes
    for pid, r in list(running.items()):
        rc = r.popen.wait()
        elapsed = int(time.time() - r.start_ts)
        print(f"Finished {r.job.model} on GPU(s) {r.gpus}  code={rc}  time={elapsed}s", flush=True)
        free_gpus(free_q, r.gpus)
        results.append((r.job.model, r.gpus, rc, elapsed))
        running.pop(pid, None)

    end_all = time.time()
    total = int(end_all - start_all)

    # Summary
    ok = sum(1 for _, _, rc, _ in results if rc == 0)
    fail = len(results) - ok
    print("\nSummary:")
    for name, g, rc, secs in results:
        print(f"  {name:35s} gpus={g} code={rc} time={secs}s")
    print(f"All done in {total}s  ok={ok}  fail={fail}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())