#!/usr/bin/env python3
import os, time, subprocess, pathlib

DATASET = '/mnt/proj1/eu-25-92/tiny_vqa_creation/output'
SPLIT = 'val'
GPUS = list(range(8))          # physical GPU indices
GPU_MEM_MB = 40960             # set to your GPU VRAM in MiB

# Jobs: model name, g = number of GPUs, mb = per‑GPU VRAM budget (MiB)
# Optional: uv = list of uv extras, extra = list of extra args for your script
JOBS = [
    {'model':'vlm_a','g':1,'mb':20000},
    {'model':'vlm_b','g':2,'mb':22000,'uv':['flash-attn==2.6.3']},
    {'model':'vlm_c','g':1,'mb':12000,'extra':['--some_flag','1']},
]

def pick(free, k, need):
    ids = sorted(range(len(free)), key=lambda i: free[i], reverse=True)
    sel = [i for i in ids if free[i] >= need][:k]
    return sel if len(sel) == k else None

def make_cmd(job):
    base = [
        'python','eval/test_benchmark.py',
        '--model_name', job['model'],
        '--dataset_path', DATASET,
        '--split', SPLIT
    ]
    if job.get('extra'): base += job['extra']
    if job.get('uv'):
        withs = []
        for p in job['uv']: withs += ['--with', p]
        return ['uv','run'] + withs + base
    return base

def main():
    logs = pathlib.Path('logs'); logs.mkdir(exist_ok=True)
    free = [GPU_MEM_MB] * len(GPUS)
    running = []

    while JOBS or running:
        # reclaim finished
        for r in running[:]:
            if r['p'].poll() is not None:
                for d in r['devs']: free[d] += r['job']['mb']
                r['log'].close()
                running.remove(r)

        # launch what fits
        i = 0
        while i < len(JOBS):
            job = JOBS[i]; need = job['mb']
            devs = pick(free, job['g'], need)
            if not devs: i += 1; continue

            env = os.environ.copy()
            env['PYTHONPATH'] = './'
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(GPUS[d]) for d in devs)

            cmd = make_cmd(job)
            ts = time.strftime('%Y%m%d_%H%M%S')
            name = ''.join(c if c.isalnum() else '_' for c in job['model'])
            logf = open(logs / f'{ts}_{name}_g{job["g"]}.log', 'w')

            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            for d in devs: free[d] -= need
            running.append({'p': p, 'devs': devs, 'log': logf, 'job': job})
            JOBS.pop(i)
        time.sleep(0.5)

if __name__ == '__main__':
    main()
