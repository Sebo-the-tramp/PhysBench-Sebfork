#!/usr/bin/env python3
import os, time, subprocess, pathlib, re
from itertools import combinations
from datetime import datetime, timedelta

# config
DATASET = '/mnt/proj1/eu-25-92/tiny_vqa_creation/output'
SPLIT = 'val'
GPUS = list(range(8))                    # physical GPU indices to use
GPU_MB = [40960] * len(GPUS)             # per-GPU VRAM in MiB (edit if heterogeneous)

# jobs: model, g = number of GPUs, mb = per-GPU VRAM needed (MiB)
# optional: uv = ['pkg==ver', ...], extra = ['--flag','value', ...]
JOBS = [
    {'model':'Phi-3-vision-128k-instruct','g':1,'mb':40000,'mode':'general', 'size': 'small'},
    {'model':'Phi-3.5V','g':1,'mb':20000,'mode':'general', 'size': 'small'},
    {'model':'mPLUG-Owl3-1B-241014','g':1,'mb':12000,'mode':'general', 'size': 'small'},
    {'model':'mPLUG-Owl3-2B-241014','g':1,'mb':16000,'mode':'general', 'size': 'small'},
    {'model':'mPLUG-Owl3-7B-241101','g':1,'mb':22000,'mode':'general', 'size': 'small'},
    {'model':'llava-interleave-qwen-7b-hf','g':1,'mb':23000,'mode':'general', 'size': 'small'},
    {'model':'llava-interleave-qwen-7b-dpo-hf','g':1,'mb':23000,'mode':'general', 'size': 'small'},
    {'model':'vila-1.5-3b','g':1,'mb':10000,'mode':'general', 'size': 'small'},
    {'model':'vila-1.5-3b-s2','g':1,'mb':15000,'mode':'general', 'size': 'small'},
    {'model':'vila-1.5-8b','g':1,'mb':20000,'mode':'general', 'size': 'small'},
    {'model':'vila-1.5-13b','g':1,'mb':31000,'mode':'general', 'size': 'small'},
    {'model':'LLaVA-NeXT-Video-7B-DPO-hf','g':1,'mb':20000,'mode':'general', 'size': 'small'}, # bug -> FIXFIXFIX
    {'model':'LLaVA-NeXT-Video-7B-hf','g':1,'mb':20000,'mode':'general', 'size': 'small'}, # bug -> FIXFIXFIX
    {'model':'InternVL2-1B','g':1,'mb':5000,'mode':'general', 'size': 'small'},
    {'model':'InternVL2-2B','g':1,'mb':13000,'mode':'general', 'size': 'small'},
    {'model':'InternVL2-4B','g':1,'mb':16000,'mode':'general', 'size': 'small'},
    {'model':'InternVL2-8B','g':1,'mb':29000,'mode':'general', 'size': 'small'},
    {'model':'InternVL2_5-1B','g':1,'mb':8000,'mode':'general', 'size': 'small'},
    {'model':'InternVL2_5-2B','g':1,'mb':13000,'mode':'general', 'size': 'small'},
    {'model':'InternVL2_5-4B','g':1,'mb':16000,'mode':'general', 'size': 'small'},
    {'model':'InternVL2_5-8B','g':1,'mb':29000,'mode':'general', 'size': 'small'},
    {'model':'Mantis-8B-Idefics2','g':1,'mb':32000,'mode':'general', 'size': 'small'},
    {'model':'Mantis-llava-7b','g':1,'mb':22000,'mode':'general', 'size': 'small'},
    {'model':'Mantis-8B-siglip-llama3','g':1,'mb':35000,'mode':'general', 'size': 'small'},
    {'model':'Mantis-8B-clip-llama3','g':1,'mb':35000,'mode':'general', 'size': 'small'},
]

def safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]+','_', name)

def make_cmd(job):
    base = [
        'python','eval/test_benchmark.py',
        '--model_name', job['model'],
        '--dataset_path', DATASET,
        '--split', SPLIT
    ]
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

def main():
    logs = pathlib.Path('logs'); logs.mkdir(exist_ok=True)
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
                print(f'{status} Completed: {r["job"]["model"]} in {duration} (return code: {r["p"].returncode}) {len(JOBS)} jobs remaining.')
                
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

            cmd = make_cmd(job)
            ts = time.strftime('%Y%m%d_%H%M%S')
            logf = open(logs / f'{ts}_{safe(job["model"])}_g{k}.log', 'w')

            start_time = datetime.now()
            print(f'Starting: {job["model"]} at {start_time.strftime("%H:%M:%S")} on GPUs {env["CUDA_VISIBLE_DEVICES"]}')

            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            for d in devs:
                free[d] -= need
            running.append({'p': p, 'devs': devs, 'log': logf, 'job': job, 'start_time': start_time})
            JOBS.pop(i)

        time.sleep(0.3)

    # Print final summary
    overall_end_time = datetime.now()
    overall_duration = overall_end_time - overall_start_time
    
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    # Sort by duration for summary
    completed_jobs.sort(key=lambda x: x['duration'], reverse=True)
    
    successful_jobs = [j for j in completed_jobs if j['return_code'] == 0]
    failed_jobs = [j for j in completed_jobs if j['return_code'] != 0]
    
    print(f"\nSUCCESSFUL MODELS ({len(successful_jobs)}):")
    print("-" * 50)
    for job in successful_jobs:
        hours, remainder = divmod(job['duration'].total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            duration_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        else:
            duration_str = f"{int(minutes):02d}m {int(seconds):02d}s"
        print(f"✓ {job['model']:<40} {duration_str}")
    
    if failed_jobs:
        print(f"\nFAILED MODELS ({len(failed_jobs)}):")
        print("-" * 50)
        for job in failed_jobs:
            hours, remainder = divmod(job['duration'].total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                duration_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
            else:
                duration_str = f"{int(minutes):02d}m {int(seconds):02d}s"
            print(f"✗ {job['model']:<40} {duration_str} (code: {job['return_code']})")
    
    # Overall statistics
    total_model_time = sum(job['duration'].total_seconds() for job in completed_jobs)
    avg_time = total_model_time / len(completed_jobs) if completed_jobs else timedelta(0)
    
    print(f"\nOVERALL STATISTICS:")
    print("-" * 50)
    
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
    
    print(f"Total models completed: {len(completed_jobs)}")
    print(f"Successful: {len(successful_jobs)}")
    print(f"Failed: {len(failed_jobs)}")
    print(f"Overall wall time: {format_duration(overall_duration)}")
    print(f"Total model time: {format_duration(total_model_time)}")
    print(f"Average time per model: {format_duration(avg_time)}")
    if overall_duration.total_seconds() > 0:
        parallelization_factor = total_model_time / overall_duration.total_seconds()
        print(f"Parallelization factor: {parallelization_factor:.2f}x")
    
    print("="*80)

if __name__ == '__main__':
    main()
