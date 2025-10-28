#!/usr/bin/env python3
import os, time, subprocess, pathlib, re
from itertools import combinations

# config
DATASET = '/mnt/proj1/eu-25-92/tiny_vqa_creation/output'
SPLIT = 'val'
GPUS = list(range(8))                    # physical GPU indices to use
GPU_MB = [24576] * len(GPUS)             # per-GPU VRAM in MiB (edit if heterogeneous)

# jobs: model, g = number of GPUs, mb = per-GPU VRAM needed (MiB)
# optional: uv = ['pkg==ver', ...], extra = ['--flag','value', ...]
JOBS = [
    {'model':'instructblip-flan-t5-xl','g':1,'mb':10000,'mode':'image-only', 'size': 'small'},
    {'model':'instructblip-flan-t5-xxl','g':1,'mb':30000,'mode':'image-only', 'size': 'small'},
    {'model':'instructblip-vicuna-7b','g':1,'mb':18000,'mode':'image-only', 'size': 'small'},
    {'model':'instructblip-vicuna-13b','g':2,'mb':30000,'mode':'image-only', 'size': 'small'},
    {'model':'blip2-flant5xxl','g':1,'mb':30000,'mode':'image-only', 'size': 'small'},
    {'model':'llava-1.5-7b-hf','g':1,'mb':16000,'mode':'image-only', 'size': 'small'},
    {'model':'llava-1.5-13b-hf','g':1,'mb':30000,'mode':'image-only', 'size': 'small'},
    {'model':'llava-v1.6-mistral-7b-hf','g':1,'mb':22000,'mode':'image-only', 'size': 'small'},
    {'model':'llava-v1.6-vicuna-7b-hf','g':1,'mb':22000,'mode':'image-only', 'size': 'small'},
    {'model':'deepseek1B','g':1,'mb':8000,'mode':'image-only', 'size': 'small'},
    {'model':'deepseek7B','g':1,'mb':18000,'mode':'image-only', 'size': 'small'},
    {'model':'Xinyuan-VL-2B','g':1,'mb':12000,'mode':'image-only', 'size': 'small'},
    {'model':'Aquila-VL-2B','g':1,'mb':14000,'mode':'image-only', 'size': 'small'},
    {'model':'MiniCPM-V2','g':1,'mb':10000,'mode':'image-only', 'size': 'small'},
    {'model':'MiniCPM-V2.5','g':1,'mb':19000,'mode':'image-only', 'size': 'small'},
    {'model':'MiniCPM-V2.6','g':1,'mb':19000,'mode':'image-only', 'size': 'small'},    
    {'model':'InternVL-Chat-V1-5-quantable','g':1,'mb':40000,'mode':'image-only', 'size': 'small'},
    {'model':'cambrian-8b','g':1,'mb':26000,'mode':'image-only', 'size': 'small', 'uv':['peft==0.17.0']},
    {'model':'MolmoE-1B','g':1,'mb':38000,'mode':'image-only', 'size': 'small'},
    {'model':'MolmoE-7B-O','g':1,'mb':38000,'mode':'image-only', 'size': 'small'},
    {'model':'MolmoE-7B-D','g':1,'mb':38000,'mode':'image-only', 'size': 'small'},
    {'model':'paligemma2-3b','g':1,'mb':10000,'mode':'image-only', 'size': 'small'},
    {'model':'paligemma2-10b','g':1,'mb':24000,'mode':'image-only', 'size': 'small'},
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

    # largest first: big per-GPU mem first, tie break by jobs that need more GPUs
    JOBS.sort(key=lambda j: (j['mb'], j['g']), reverse=True)

    while JOBS or running:
        # reclaim finished
        for r in running[:]:
            if r['p'].poll() is not None:
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

            print(f'Starting: {" ".join(cmd)} on GPUs {env["CUDA_VISIBLE_DEVICES"]}')

            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            for d in devs:
                free[d] -= need
            running.append({'p': p, 'devs': devs, 'log': logf, 'job': job})
            JOBS.pop(i)

        time.sleep(0.3)

if __name__ == '__main__':
    main()
