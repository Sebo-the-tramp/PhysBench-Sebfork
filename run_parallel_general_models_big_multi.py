#!/usr/bin/env python3
import os, time, subprocess, pathlib, re
from itertools import combinations
from datetime import datetime, timedelta

# config
# DATASET = '/mnt/proj1/eu-25-92/tiny_vqa_creation/output'
SPLIT = 'val'

# this should run on 8 A100-40GBs
# GPUS = list(range(8))                    # physical GPU indices to use
# GPU_MB = [40960] * len(GPUS)             # per-GPU VRAM in MiB (edit if heterogeneous)

# config
DATASET = '/mnt/proj1/eu-25-92/tiny_vqa_creation/output'
SPLIT = 'val'
GPUS = list(range(8))                    # physical GPU indices to use
GPU_MB = [40960] * len(GPUS)             # per-GPU VRAM in MiB (edit if heterogeneous)
RUN_NAME = 'run_05_10K'

# jobs: model, g = number of GPUs, mb = per-GPU VRAM needed (MiB)
# optional: uv = ['pkg==ver', ...], extra = ['--flag','value', ...]
JOBS = [
    # All these models are 'general' models
    {'model':'InternVL2-26B','g':2,'mb':40000,'mode':'general', 'size': 'big'},
    {'model':'InternVL2-40B','g':2,'mb':40000,'mode':'general', 'size': 'big', 'uv':['transformers==4.57.1']},
    # {'model':'InternVL2-76B','g':3,'mb':40000,'mode':'general', 'size': 'big', 'uv':['transformers==4.57.1']},
    {'model':'InternVL2_5-26B','g':2,'mb':40000,'mode':'general', 'size': 'big'},
    {'model':'InternVL2_5-38B','g':2,'mb':40000,'mode':'general', 'size': 'big', 'uv':['transformers==4.57.1']},
    # {'model':'InternVL2_5-78B','g':3,'mb':40000,'mode':'general', 'size': 'big', 'uv':['transformers==4.57.1']},
]
# CPU limiting config
CPU_PER_JOB = 24  # same number of logical CPUs per process
CPU_IDS = list(range(os.cpu_count() or 1))

def safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]+','_', name)

def make_cmd(job, run_name):
    if run_name is None:
        raise ValueError("run_name must be provided to make_cmd")
    
    base = [
        'python','eval/test_benchmark.py',
        '--model_name', job['model'],
        '--dataset_path', DATASET,
        '--split', SPLIT,
        '--run_name', f"{run_name}"
    ]
    print("run_name in make_cmd:", base)
    if job.get('extra'):
        base += job['extra']
    if job.get('uv'):
        pref = ['uv','run']
        for p in job['uv']:
            pref += ['--with', p]
        return pref + base
    return base

def pick(free, k, need):
    # choose k GPUs that minimize fragmentation: first minimize worst leftover, then total leftover
    best_cost, best = None, None
    for combo in combinations(range(len(free)), k):
        if all(free[i] >= need for i in combo):
            worst_left = max(free[i] - need for i in combo)
            total_left = sum(free[i] - need for i in combo)
            cost = (worst_left, total_left)
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, combo
    return list(best) if best else None

def run_one_experiment(run_name='default_run'):

    logs = pathlib.Path('logs'); logs.mkdir(exist_ok=True)
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
    running = []
    completed_jobs = []       # track completed jobs with timing
    overall_start_time = datetime.now()

    # largest first: big per-GPU mem first, tie break by jobs that need more GPUs
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
                
                # Print completion message
                status = "✓" if r['p'].returncode == 0 else "✗"
                log(f'{status} Completed: {r["job"]["model"]} in {duration} (return code: {r["p"].returncode}) {len(JOBS)} jobs remaining.')
                
                for d in r['devs']:
                    free[d] += r['job']['mb']
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

            env = os.environ.copy()
            env['PYTHONPATH'] = './'
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(GPUS[d]) for d in devs)

            cmd = make_cmd(job, run_name=run_name)
            log("running the command: " + ' '.join(cmd))
            ts = time.strftime('%Y%m%d_%H%M%S')
            logf = open(logs / f'{ts}_{safe(job["model"])}_g{k}.log', 'w')

            start_time = datetime.now()
            log(f'Starting: {job["model"]} at {start_time.strftime("%H:%M:%S")} on GPUs {env["CUDA_VISIBLE_DEVICES"]}')

            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            for d in devs:
                free[d] -= need
            running.append({'p': p, 'devs': devs, 'log': logf, 'job': job, 'start_time': start_time})
            JOBS.pop(i)

        time.sleep(0.3)

    # Print final summary
    overall_end_time = datetime.now()
    overall_duration = overall_end_time - overall_start_time
    
    log("\n" + "="*80)
    log("EXECUTION SUMMARY")
    log("="*80)
    
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
    
    log("="*80)
    log(f"Summary stored at {summary_log_path}")
    summary_log.close()

GENERAL_RUN_COUNT = 6

def main():

    # we have a list of experiments with different run names

    runs_config = {
        "10K_general":{
            "run_name": "run_06_general",
            "quantity": "10K"
        },
        # "1K_soft":{
        #     "run_name": "run_06_1K_soft"
        # },
        # "1K_medium":{
        #     "run_name": "run_06_1K_medium"
        # },
        # "1K_stiff":{
        #     "run_name": "run_06_1K_stiff"
        # },
        # "1K_roi_circling":{
        #     "run_name": "run_06_1K_roi_circling"
        # },
        # "1K_masking":{
        #     "run_name": "run_06_1K_masking"
        # },
        # "1K_scene_context":{
        #     "run_name": "run_06_1K_scene_context"
        # },
        # "1K_textual_context":{
        #     "run_name": "run_06_1K_textual_context"
        # }
    }

    for run_name, config in runs_config.items():
        # config["run_name"] = f"run_{str(GENERAL_RUN_COUNT).zfill(2)}_{run_name}"
        run_name_and_path = f"{config['run_name']}/test_{config['run_name']}_{config['quantity']}"
        run_one_experiment(run_name=run_name_and_path)

if __name__ == '__main__':    
    main()
