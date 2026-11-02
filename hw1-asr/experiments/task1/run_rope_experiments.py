#!/usr/bin/env python3

"""
This script is intended ofr the experiments additional questions for task1_rope 
this is intended to be run inside the hw1-asr gpu allocated GPU node.

1. Change ROPE_THREADS to 32, 64, 128, 512 and 1024 and compare the GB/s of the audio encoder case. Which sizes are slow, and why might very small blocks hurt?
2. The tiny cases reach almost none of the peak, and their time barely changes with the size. What is that time made of?
3. Build with fast but less accurate maths: make clean && make test1 NVCCFLAGS="-O3 -std=c++17 --use_fast_math". cosf and sinf become fast hardware approximations. 
- Which case fails, and why that one? (Look at the largest angle each case uses.) Run make clean afterwards to go back to the normal flags.

Task 1 RoPE experiment runner. Run inside hw1-asr on an allocated GPU node.
Restores csrc/task1_rope.cu even when interrupted.
"""
import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

ROOT = Path.cwd()
SOURCE = ROOT / 'csrc' / 'task1_rope.cu' # source for run 
PATTERN = re.compile(rb'(constexpr\s+int\s+ROPE_THREADS\s*=\s*)\d+(\s*;)')
CASES = ('tiny', 'odd sizes', 'audio encoder', 'text decoder') # different experimnets flags we are going to pass into this script recursively 

def run(cmd, log_file):
    with log_file.open('w') as f:
        f.write('$ ' + ' '.join(cmd) + '\n\n')
        f.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
        f.write(p.stdout)
        f.write(f'\nEXIT_CODE={p.returncode}\n')
    return p.returncode, p.stdout

def parse_rows(output, threads, mode, gpu, trial):
    rows=[]
    for line in output.splitlines():
        if 'GB/s' not in line or not any(s in line for s in CASES):
            continue
        case=next(s for s in CASES if s in line)
        bw=re.search(r'([\d.]+)\s*GB/s', line)
        peak=re.search(r'\(\s*([\d.]+)%\s*of peak\)', line)
        # Test output format is controlled by tests/test_util.h; keep full line for auditing.
        rows.append({'mode':mode,'threads':threads,'trial':trial,'gpu':gpu,'case':case,
                     'status':'FAIL' if 'FAIL' in line else 'PASS' if 'PASS' in line else 'UNKNOWN',
                     'GB_per_s':bw.group(1) if bw else '',
                     'pct_peak':peak.group(1) if peak else '', 'raw_result':line.strip()})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trials',type=int,default=3)
    ap.add_argument('--sizes',type=int,nargs='+',default=[32,64,128,256,512,1024]) # differnet flags we are passing into the script for cuda block sizes 
    ap.add_argument('--skip-fast-math',action='store_true') # --use fast maths nvidia operation expected to dramatically speed up. 
    args=ap.parse_args()
    if not SOURCE.exists():
        sys.exit('Run from hw1-asr (missing csrc/task1_rope.cu)') # run check that this is actually a file we caun run in parallel 
        
    if args.trials<1 or any(n<32 or n>1024 or n%32 for n in args.sizes):
        sys.exit('Need trials >= 1 and block sizes that are warp multiples from 32 to 1024') # ensure the warp sizes can contain the threads? 

    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT / 'experiments' / 'task1' / 'results' / stamp
    out.mkdir(parents=True,exist_ok=True)
    original=SOURCE.read_bytes()
    if len(PATTERN.findall(original))!=1:
        sys.exit('Could not uniquely find ROPE_THREADS assignment')
    (out/'source_original.cu').write_bytes(original)
    for name,cmd in [('gpu.txt',['nvidia-smi']),('cuda_version.txt',['nvcc','--version']),('git_status.txt',['git','status','--short'])]:
        run(cmd,out/name)
    gpu_res=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True)
    gpu=gpu_res.stdout.strip().splitlines()[0] if gpu_res.returncode==0 and gpu_res.stdout.strip() else 'unknown'
    records=[]
    failed=[]
    try:
        configurations=[('normal',size) for size in args.sizes]
        if not args.skip_fast_math:
            configurations += [('fast_math',256)]
        for mode,size in configurations:
            print(f'Building {mode}, ROPE_THREADS={size}',flush=True)
            modified=PATTERN.sub(lambda m:m.group(1)+str(size).encode()+m.group(2),original)
            SOURCE.write_bytes(modified)
            # Always clean between different flags to ensure all objects are recompiled.
            rc,_=run(['make','clean'],out/f'{mode}_{size}_clean.log')
            if rc:
                failed.append((mode,size,'clean'));continue
            make=['make','test1']
            if mode=='fast_math':
                make.append('NVCCFLAGS=-O3 -std=c++17 --use_fast_math')
            for trial in range(1,args.trials+1):
                # First repetition builds; subsequent repetitions just run test executable.
                command=make if trial==1 else ['make','test1'] + (make[2:] if mode=='fast_math' else [])
                rc,output=run(command,out/f'{mode}_{size}_trial{trial}.log')
                records.extend(parse_rows(output,size,mode,gpu,trial))
                if rc:
                    failed.append((mode,size,trial))
                    print(f'  FAIL trial {trial}: see log',flush=True)
                else:
                    print(f'  PASS trial {trial}',flush=True)
    finally:
        SOURCE.write_bytes(original)
       
        # Remove modified-flag build products to prevent accidentally reusing fast-math objects.
        rc,_=run(['make','clean'],out/'final_clean.log')
        columns=['mode','threads','trial','gpu','case','status','GB_per_s','pct_peak','raw_result']
        
        with (out/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader();writer.writerows(records)
        
        (out/'readme.txt').write_text('Results are from repeated existing make test1 runs.\nEach raw log records build output, timing and correctness.\n'
                                   'The source file has been restored; the build directory was cleaned.\n'
                                   'Failures: '+repr(failed)+'\n')
        print(f'\nSaved results: {out}\nSource restored. Build directory cleaned.',flush=True)
    if failed:
        sys.exit(f'{len(failed)} run(s) failed; inspect logs, including fast-math precision failures')

if __name__=='__main__':
    main()
