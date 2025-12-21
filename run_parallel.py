#!/usr/bin/env python3
import argparse
import os
import time
import subprocess
import pathlib
import re
from itertools import combinations
from datetime import datetime, timedelta

# Defaults that can be overridden via CLI flags
DATASET = '/mnt/proj1/eu-25-92/tiny_vqa_creation/output'
SPLIT = 'val'
GPUS = list(range(8))                    # physical GPU indices to use
GPU_MB = [40960] * len(GPUS)             # per-GPU VRAM in MiB (edit if heterogeneous)

JOB_SETS = {
    'big': {
        'cpu_per_job': 24,
        'jobs': [
            {'model': 'InternVL2-26B', 'g': 2, 'mb': 40000, 'mode': 'general', 'size': 'big'},
            {'model': 'InternVL2-40B', 'g': 2, 'mb': 40000, 'mode': 'general', 'size': 'big', 'uv': ['transformers==4.57.1']},
            # {'model':'InternVL2-76B','g':3,'mb':40000,'mode':'general', 'size': 'big', 'uv':['transformers==4.57.1']},
            {'model': 'InternVL2_5-26B', 'g': 2, 'mb': 40000, 'mode': 'general', 'size': 'big'},
            {'model': 'InternVL2_5-38B', 'g': 2, 'mb': 40000, 'mode': 'general', 'size': 'big', 'uv': ['transformers==4.57.1']},
            # {'model':'InternVL2_5-78B','g':3,'mb':40000,'mode':'general', 'size': 'big', 'uv':['transformers==4.57.1']},
        ]
    },
    'small': {
        'cpu_per_job': 12,
        'jobs': [
            {'model': 'instructblip-flan-t5-xl', 'g': 1, 'mb': 10000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'instructblip-flan-t5-xxl', 'g': 1, 'mb': 30000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'instructblip-vicuna-7b', 'g': 1, 'mb': 18000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'instructblip-vicuna-13b', 'g': 2, 'mb': 30000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'blip2-flant5xxl', 'g': 1, 'mb': 30000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'llava-1.5-7b-hf', 'g': 1, 'mb': 16000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'llava-1.5-13b-hf', 'g': 1, 'mb': 30000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'llava-v1.6-mistral-7b-hf', 'g': 1, 'mb': 22000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'llava-v1.6-vicuna-7b-hf', 'g': 1, 'mb': 22000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'deepseek1B', 'g': 1, 'mb': 8000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'deepseek7B', 'g': 1, 'mb': 18000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'Xinyuan-VL-2B', 'g': 1, 'mb': 12000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'Aquila-VL-2B', 'g': 1, 'mb': 14000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'MiniCPM-V2', 'g': 1, 'mb': 10000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'MiniCPM-V2.5', 'g': 1, 'mb': 19000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'MiniCPM-V2.6', 'g': 1, 'mb': 19000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'Qwen-VL-Chat', 'g': 1, 'mb': 20000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'cambrian-8b', 'g': 1, 'mb': 26000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'paligemma2-3b', 'g': 1, 'mb': 10000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'paligemma2-10b', 'g': 1, 'mb': 24000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'InternVL-Chat-V1-5-quantable', 'g': 1, 'mb': 40000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'MolmoE-1B', 'g': 1, 'mb': 38000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'MolmoE-7B-O', 'g': 1, 'mb': 38000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'MolmoE-7B-D', 'g': 1, 'mb': 38000, 'mode': 'image-only', 'size': 'small'},
            {'model': 'Phi-3-vision-128k-instruct', 'g': 1, 'mb': 40000, 'mode': 'general', 'size': 'small'},
            {'model': 'Phi-3.5V', 'g': 1, 'mb': 20000, 'mode': 'general', 'size': 'small'},
            {'model': 'mPLUG-Owl3-1B-241014', 'g': 1, 'mb': 12000, 'mode': 'general', 'size': 'small'},
            {'model': 'mPLUG-Owl3-2B-241014', 'g': 1, 'mb': 16000, 'mode': 'general', 'size': 'small'},
            {'model': 'mPLUG-Owl3-7B-241101', 'g': 1, 'mb': 22000, 'mode': 'general', 'size': 'small'},
            {'model': 'llava-interleave-qwen-7b-hf', 'g': 1, 'mb': 23000, 'mode': 'general', 'size': 'small'},
            {'model': 'llava-interleave-qwen-7b-dpo-hf', 'g': 1, 'mb': 23000, 'mode': 'general', 'size': 'small'},
            {'model': 'vila-1.5-3b', 'g': 1, 'mb': 10000, 'mode': 'general', 'size': 'small'},
            {'model': 'vila-1.5-3b-s2', 'g': 1, 'mb': 15000, 'mode': 'general', 'size': 'small'},
            {'model': 'vila-1.5-8b', 'g': 1, 'mb': 20000, 'mode': 'general', 'size': 'small'},
            {'model': 'vila-1.5-13b', 'g': 1, 'mb': 31000, 'mode': 'general', 'size': 'small'},
            {'model': 'LLaVA-NeXT-Video-7B-DPO-hf', 'g': 1, 'mb': 20000, 'mode': 'general', 'size': 'small'},
            {'model': 'LLaVA-NeXT-Video-7B-hf', 'g': 1, 'mb': 20000, 'mode': 'general', 'size': 'small'},
            {'model': 'InternVL2-1B', 'g': 1, 'mb': 5000, 'mode': 'general', 'size': 'small'},
            {'model': 'InternVL2-2B', 'g': 1, 'mb': 13000, 'mode': 'general', 'size': 'small'},
            {'model': 'InternVL2-4B', 'g': 1, 'mb': 16000, 'mode': 'general', 'size': 'small'},
            {'model': 'InternVL2-8B', 'g': 1, 'mb': 29000, 'mode': 'general', 'size': 'small'},
            {'model': 'InternVL2_5-1B', 'g': 1, 'mb': 8000, 'mode': 'general', 'size': 'small'},
            {'model': 'InternVL2_5-2B', 'g': 1, 'mb': 13000, 'mode': 'general', 'size': 'small'},
            {'model': 'InternVL2_5-4B', 'g': 1, 'mb': 16000, 'mode': 'general', 'size': 'small'},
            {'model': 'InternVL2_5-8B', 'g': 1, 'mb': 29000, 'mode': 'general', 'size': 'small'},
            {'model': 'Mantis-8B-Idefics2', 'g': 1, 'mb': 32000, 'mode': 'general', 'size': 'small'},
            {'model': 'Mantis-llava-7b', 'g': 1, 'mb': 22000, 'mode': 'general', 'size': 'small'},
            {'model': 'Mantis-8B-siglip-llama3', 'g': 1, 'mb': 35000, 'mode': 'general', 'size': 'small'},
            {'model': 'Mantis-8B-clip-llama3', 'g': 1, 'mb': 35000, 'mode': 'general', 'size': 'small'},
        ]
    }
}


def pick_cpus(free_set, n):
    """Pick n free logical CPUs, or None if not enough."""
    if len(free_set) < n:
        return None
    chosen = sorted(list(free_set))[:n]
    return chosen


def safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def make_cmd(job, run_name, dataset, split):
    if run_name is None:
        raise ValueError("run_name must be provided to make_cmd")

    base = [
        'python', 'eval/test_benchmark.py',
        '--model_name', job['model'],
        '--dataset_path', dataset,
        '--split', split,
        '--run_name', f"{run_name}"
    ]
    if job.get('extra'):
        base += job['extra']
    if job.get('uv'):
        pref = ['uv', 'run']
        for p in job['uv']:
            pref += ['--with', p]
        return pref + base
    return base


def pick(free, k, need):
    # choose k GPUs that minimize fragmentation
    best_cost, best = None, None
    for combo in combinations(range(len(free)), k):
        if all(free[i] >= need for i in combo):
            worst_left = max(free[i] - need for i in combo)
            total_left = sum(free[i] - need for i in combo)
            cost = (worst_left, total_left)
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, combo
    return list(best) if best else None


def run_one_experiment(run_name, jobs, cpu_per_job, dataset, split):
    logs = pathlib.Path('logs')
    logs.mkdir(exist_ok=True)
    summary_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_log_path = logs / f'{summary_stamp}_{safe(run_name)}_summary.log'
    summary_log = open(summary_log_path, 'w')

    def log(msg='', end='\n'):
        text = str(msg)
        print(text, end=end, flush=True)
        summary_log.write(text)
        if end:
            summary_log.write(end)
        summary_log.flush()

    log(f"Starting experiment with run name: {run_name}")
    log(f"Summary log: {summary_log_path}")
    log()
    free = GPU_MB[:]          # remaining MiB per GPU
    cpu_ids = list(range(os.cpu_count() or 1))
    cpu_free = set(cpu_ids)   # remaining free logical CPUs
    running = []
    completed_jobs = []       # track completed jobs with timing
    overall_start_time = datetime.now()

    # largest first: big per-GPU mem first, tie break by jobs that need more GPUs
    JOBS = [job.copy() for job in jobs]
    JOBS.sort(key=lambda j: (j['mb'], j['g']), reverse=True)

    while JOBS or running:
        # reclaim finished
        for r in running[:]:
            if r['p'].poll() is not None:
                end_time = datetime.now()
                duration = end_time - r['start_time']

                # Record completion
                completed_jobs.append({
                    'model': r['job']['model'],
                    'start_time': r['start_time'],
                    'end_time': end_time,
                    'duration': duration,
                    'return_code': r['p'].returncode
                })

                status = "✓" if r['p'].returncode == 0 else "✗"
                print(f'{status} Completed: {r["job"]["model"]} in {duration} (return code: {r["p"].returncode}) {len(JOBS)} jobs remaining.')

                for d in r['devs']:
                    free[d] += r['job']['mb']
                cpu_free.update(r['cpus'])
                r['log'].close()
                running.remove(r)

        # launch what fits
        i = 0
        while i < len(JOBS):
            job = JOBS[i]
            need, k = job['mb'], job['g']
            devs = pick(free, k, need)
            if not devs:
                i += 1
                continue
            # choose CPUs for this job
            cpus = pick_cpus(cpu_free, cpu_per_job)
            if not cpus:
                i += 1
                continue
            env = os.environ.copy()
            env['PYTHONPATH'] = './'
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(GPUS[d]) for d in devs)

            cmd = make_cmd(job, run_name=run_name, dataset=dataset, split=split)
            log("running the command: " + ' '.join(cmd))
            ts = time.strftime('%Y%m%d_%H%M%S')
            logf = open(logs / f'{ts}_{safe(job["model"])}_g{k}.log', 'w')

            start_time = datetime.now()
            log(f'Starting: {job["model"]} at {start_time.strftime("%H:%M:%S")} on GPUs {env["CUDA_VISIBLE_DEVICES"]}')
            final_cmd = ['taskset', '-c', ','.join(map(str, cpus))] + cmd

            p = subprocess.Popen(final_cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            for d in devs:
                free[d] -= need
            for c in cpus:
                cpu_free.discard(c)
            running.append({'p': p, 'devs': devs, 'cpus': cpus, 'log': logf, 'job': job, 'start_time': start_time})
            JOBS.pop(i)

        time.sleep(0.3)

    # Print final summary
    overall_end_time = datetime.now()
    overall_duration = overall_end_time - overall_start_time

    log("\n" + "=" * 80)
    log("EXECUTION SUMMARY")
    log("=" * 80)

    # Sort by duration for summary
    completed_jobs.sort(key=lambda x: x['duration'], reverse=True)

    successful_jobs = [j for j in completed_jobs if j['return_code'] == 0]
    failed_jobs = [j for j in completed_jobs if j['return_code'] != 0]

    log(f"\nSUCCESSFUL MODELS ({len(successful_jobs)}):")
    log("-" * 50)
    for job in successful_jobs:
        hours, remainder = divmod(job['duration'].total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            duration_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        else:
            duration_str = f"{int(minutes):02d}m {int(seconds):02d}s"
        log(f"✓ {job['model']:<40} {duration_str}")

    if failed_jobs:
        log(f"\nFAILED MODELS ({len(failed_jobs)}):")
        log("-" * 50)
        for job in failed_jobs:
            hours, remainder = divmod(job['duration'].total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                duration_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
            else:
                duration_str = f"{int(minutes):02d}m {int(seconds):02d}s"
            log(f"✗ {job['model']:<40} {duration_str} (code: {job['return_code']})")

    # Overall statistics
    total_model_time = sum(job['duration'].total_seconds() for job in completed_jobs)
    avg_time = total_model_time / len(completed_jobs) if completed_jobs else timedelta(0)

    log(f"\nOVERALL STATISTICS:")
    log("-" * 50)

    def to_seconds(x):
        if isinstance(x, timedelta):
            return x.total_seconds()
        elif isinstance(x, (int, float)):
            return x
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    def format_duration(td):
        total_seconds = int(to_seconds(td))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        else:
            return f"{int(minutes):02d}m {int(seconds):02d}s"

    log(f"Total models completed: {len(completed_jobs)}")
    log(f"Successful: {len(successful_jobs)}")
    log(f"Failed: {len(failed_jobs)}")
    log(f"Overall wall time: {format_duration(overall_duration)}")
    log(f"Total model time: {format_duration(total_model_time)}")
    log(f"Average time per model: {format_duration(avg_time)}")
    if overall_duration.total_seconds() > 0:
        parallelization_factor = total_model_time / overall_duration.total_seconds()
        log(f"Parallelization factor: {parallelization_factor:.2f}x")

    log("=" * 80)
    log(f"Summary stored at {summary_log_path}")
    summary_log.close()


def build_run_name(base_name, quantity):
    karo = ""
    if os.path.exists("/home/it4i-thvu/seb_dev/computational_physics/PhysBench-Sebfork"):
        karo = "karo_"
    return f"{base_name}/test_{base_name}_{karo}{quantity}"


def main():
    parser = argparse.ArgumentParser(description="Parallel runner for PhysBench models.")
    parser.add_argument('--run-name', required=True, help='Base run name, e.g. run_07_general')
    parser.add_argument('--quantity', required=True, help='Quantity tag, e.g. 10K or 1K')
    parser.add_argument('--model-size', choices=JOB_SETS.keys(), default='big', help='Model group to evaluate')
    parser.add_argument('--dataset-path', default=DATASET, help='Dataset path (default matches repo setting)')
    parser.add_argument('--split', default=SPLIT, help='Dataset split (default: val)')
    args = parser.parse_args()

    config = JOB_SETS[args.model_size]
    run_name_and_path = build_run_name(args.run_name, args.quantity)

    run_one_experiment(
        run_name=run_name_and_path,
        jobs=config['jobs'],
        cpu_per_job=config['cpu_per_job'],
        dataset=args.dataset_path,
        split=args.split,
    )


if __name__ == '__main__':
    main()
