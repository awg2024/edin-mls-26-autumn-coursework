"""
Task 2 experiment runner to answers posed in the readme file, leaving original csrc untouched.

Change REDUCE_THREADS to 32, 64, 128, 512 and 1024. Watch the "layernorm, audio encoder" and "softmax, vocabulary row" cases. Why do they react differently?
The vocabulary row is 59,264 floats, but its GB/s is far below that of the other large cases. How many blocks does the launch have, and how many of the GPU's SMs can that keep busy? Sketch (or write) a version that splits one long row over many blocks. MaxSum makes the second step easy.
Delete the final __syncthreads() (step 4) in block_reduce_sum and run make test2. Then run compute-sanitizer --tool racecheck build/sm_XX/test_task2_reduce small (small runs only the small cases, without timing, so that the slow sanitizer finishes in seconds). What do the test and the sanitizer each tell you? Put the barrier back afterwards.
In rmsnorm_kernel, replace the block reduction with atomicAdd on one __shared__ float (every thread adds its partial sum). Compare the time with your shuffle version.

"""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path.cwd()
ORIGINAL = ROOT / 'csrc' # root dir 
TEST = ROOT / 'experiments' / 'task2' # test root dir 



# execute helepr 
def execute(command, log):
    with log.open('w') as f:
        p = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f'\nEXIT_CODE={p.returncode}\n')
    return p.returncode


def copy_sources(folder):
    target = folder / 'csrc'
    shutil.copytree(ORIGINAL, target) # dir helper 
    return target


def substitute_once(path, old, new, label):
    content = path.read_text()
    occurrences = content.count(old)
    if occurrences != 1:
        raise RuntimeError(f'{label}: expected one match, found {occurrences}; inspect {path}')
    path.write_text(content.replace(old, new, 1))


def build_and_test(src, build, logs, prefix, small=False):
    makeargs = ['make', f'SRC={src.relative_to(ROOT)}', f'BUILD={build.relative_to(ROOT)}', 'test2']
    status = execute(makeargs, logs / f'{prefix}_test.log')
    if small and status == 0:
        # Direct executable small-mode instead of timed full test
        status = execute([str(build / 'test_task2_reduce'), 'small'], logs / f'{prefix}_small.log')
    return status


def parse_log(path, variant, threads, writer): # helper function for read out of csv 
    for line in path.read_text(errors='replace').splitlines():
        match = re.match(r'^\s*(rmsnorm|layernorm|softmax),\s*(.*?)\s+(PASS|FAIL)\s+max_err=([\deE+.-]+)(?:\s+([\d.]+)\s+ms\s+([\d.]+)\s+GB/s\s+\(\s*(\d+)% of peak\))?', line)
        if match:
            op, name, status, error, latency, bandwidth, peak = match.groups()
            writer.writerow([variant, threads, op + ', ' + name.strip(), status, error, latency or '', bandwidth or '', peak or ''])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', nargs='+', choices=['sweep','race','atomic'], default=['sweep','race','atomic']) # different experiment alteraations we are applying 
    parser.add_argument('--sizes', nargs='+', type=int, default=[32,64,128,256,512,1024]) # REDUCE_thread configurations we are applying 
    args = parser.parse_args()
    
    if not (ROOT / 'Makefile').exists() or not (ORIGINAL / 'task2_reduce.cu').exists(): # checker if task2 reduce exsits 
        sys.exit('Run this script from the hw1-asr directory.')


    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = TEST / 'results' / stamp
    work = out / 'sources'
    logs = out / 'logs'
    logs.mkdir(parents=True)
    sm = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True, text=True)
    gpu_sm = sm.stdout.strip().splitlines()[0].replace('.', '') if sm.returncode == 0 and sm.stdout.strip() else None
    if not gpu_sm:
        sys.exit('No CUDA GPU detected. Obtain a Slurm GPU allocation first.')
    execute(['nvidia-smi'], logs / 'gpu.log')
    csvpath = out / 'results.csv'
    
     
    with csvpath.open('w',newline='') as f: # logs saving for report writeup 
        writer = csv.writer(f)
        writer.writerow(['variant','threads','case','status','max_err','latency_ms','effective_GBps','pct_peak'])
        
        if 'sweep' in args.modes: # Benchmark performance across different REDUCE thread block sizes seeing the performance 
            for threads in args.sizes:
                if threads < 32 or threads > 1024 or threads % 32:
                    raise ValueError('Threads must be a multiple of 32, between 32 and 1024')
                name = f'normal_{threads}'
                src = copy_sources(work / name)
                substitute_once(src / 'task2_reduce.cu', 'constexpr int REDUCE_THREADS = 256;', f'constexpr int REDUCE_THREADS = {threads};', name)
                build = ROOT / 'build' / 'task2_experiments' / name / ('sm_'+gpu_sm)
                print('Running',name,flush=True)
                status = build_and_test(src, build, logs, name)
                parse_log(logs / f'{name}_test.log',name,threads,writer)
                print('  ', 'PASS' if status == 0 else f'FAIL (inspect logs/{name}_test.log)',flush=True)
                f.flush()
        if 'race' in args.modes: # deliberately removes critical sync thread barriers that guards shared memory between warps / writes            
            name = 'race_missing_barrier_256'
            src = copy_sources(work / name)
            header = src / 'task2_reduce.cuh'
            text = header.read_text()
            needle = '  __syncthreads();  // SECOND CALL'
            substitute_once(header,needle,'  // EXPERIMENT: final barrier deliberately removed',name)
            build = ROOT / 'build' / 'task2_experiments' / name / ('sm_'+gpu_sm)
            print('Running',name,flush=True)
            status = build_and_test(src,build,logs,name)
            parse_log(logs / f'{name}_test.log',name,256,writer)
            sanitizer=shutil.which('compute-sanitizer')
            if sanitizer:
                rc = execute([sanitizer,'--tool','racecheck',str(build / 'test_task2_reduce'),'small'], logs / f'{name}_racecheck.log')
                print('  racecheck exit:',rc,flush=True)
            else:
                print('  compute-sanitizer not on PATH; racecheck skipped',flush=True)
            print('  test2 exit:',status,flush=True)
            f.flush()
        if 'atomic' in args.modes: # Compare standard tree reduction performance against shared-memory atomic additions in RMSNorm.
            name='atomic_rmsnorm_256'
            src=copy_sources(work/name)
            file=src/'task2_reduce.cu'
            original=file.read_text()
            # Limit replacement to RMSNorm body; never replace LayerNorm reductions.
            start=original.index('__global__ void rmsnorm_kernel(')
            end=original.index('// Softmax over each row',start)
            region=original[start:end]
            pat=re.compile(r'(?m)^\s*(?:const\s+)?float\s+(\w+)\s*=\s*block_reduce_sum\(sq\);')
            matches=list(pat.finditer(region))
            if len(matches)!=1:
                raise RuntimeError('Atomic experiment needs one `float name = block_reduce_sum(sq);` in RMSNorm; found '+str(len(matches))+'; edit source template accordingly.')
            var=matches[0].group(1)
            atomic=f'''\n  // EXPERIMENT: shared-memory atomic sum of partial squares.\n  __shared__ float shared_sq;\n  if (threadIdx.x == 0) shared_sq = 0.0f;\n  __syncthreads();\n  atomicAdd(&shared_sq, sq);\n  __syncthreads();\n  float {var} = shared_sq;'''
            region=pat.sub(atomic,region,count=1)
            file.write_text(original[:start]+region+original[end:])
            build=ROOT/'build'/'task2_experiments'/name/('sm_'+gpu_sm)
            print('Running',name,flush=True)
            status=build_and_test(src,build,logs,name)
            parse_log(logs/f'{name}_test.log',name,256,writer)
            print('  ', 'PASS' if status==0 else 'FAIL; inspect log',flush=True)
            f.flush()

    print('\nResults:',out)
    print('Original csrc/ files never changed.')

if __name__=='__main__':
    main()
