"""Schedule at most two experiment clients against ONE model-serving replica."""
import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--stage',choices=['fit','dev','test'],required=True)
    p.add_argument('--jobs-per-gpu',type=int,default=2);p.add_argument('--limit',type=int)
    p.add_argument('--config',type=Path,default=ROOT/'configs/study.json');p.add_argument('--lock',type=Path,default=ROOT/'protocol.lock.json')
    p.add_argument('--methods',nargs='+',help='Explicit partial method selection; omitted means the full configured suite')
    p.add_argument('--datasets',nargs='+',help='Explicit dataset selection; omitted means every configured dataset')
    p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if a.jobs_per_gpu not in [1,2]:raise ValueError('Use one isolated client or two shared clients per replica')
    c=json.loads(a.config.read_text())
    if a.model not in c['models']+c.get('optional_models',[]):p.error('Model is not enabled in this package')
    slug=a.model.split('/')[-1]
    allowed=['safesieve','card','agentnet'] if a.stage=='fit' else c['controls'] if a.stage=='controls' else c['methods']
    if a.methods and (len(set(a.methods))!=len(a.methods) or any(m not in allowed for m in a.methods)):p.error('Invalid or duplicate methods for this stage')
    selected=a.methods or allowed
    if a.datasets and (len(set(a.datasets))!=len(a.datasets) or not set(a.datasets)<=set(c['datasets'])):
        p.error('Invalid or duplicate datasets')
    datasets=a.datasets or c['datasets']
    if a.stage in ['fit','dev','controls'] and datasets==['humaneval']:
        p.error('HumanEval has no training/development split; use frozen held-out evaluation')
    if a.methods:print('EXPLICIT PARTIAL SUITE:', ', '.join(selected), flush=True)
    if not a.plan_only:
        sys.path.insert(0,str(ROOT))
        from agentroute.jobs import verify_upstream
        for method in selected:
            if method in ['safesieve','card','agentnet','selforg']:verify_upstream(method,ROOT/'upstream')
        from agentroute.server import wait_ready
        headers={}
        if c.get('api_key_env'):headers['Authorization']='Bearer '+os.environ[c['api_key_env']]
        wait_ready(a.base_url,a.model,headers=headers)
    jobs=[];checkpoints=ROOT/'checkpoints'/slug
    for dataset in datasets:
        if a.stage in ['fit','dev','controls'] and dataset=='humaneval':continue
        methods=selected
        for method in methods:
            if a.stage=='fit':
                directory=checkpoints/dataset/method
                marker=directory/('checkpoint.pkl' if method=='agentnet' else 'checkpoint.pt')
                if marker.exists():continue
                command=[sys.executable,str(ROOT/'scripts/fit.py'),'--method',method,'--dataset',dataset,
                    '--model',a.model,'--base-url',a.base_url,'--config',str(a.config),'--output',str(directory)]
                jobs.append((dataset,method,'fit',command));continue
            split='test' if a.stage=='test' else 'dev';limit=a.limit if a.limit is not None else 80 if split=='test' else 16 if a.stage=='controls' else 50
            for seed in c['seeds'] if split=='test' else [c['seeds'][0]]:
                command=[sys.executable,str(ROOT/'scripts/run.py'),'--method',method,'--dataset',dataset,
                    '--model',a.model,'--base-url',a.base_url,'--config',str(a.config),'--split',split,
                    '--limit',str(limit),'--seed',str(seed)]
                if split=='test':command+=['--lock',str(a.lock)]
                if method in ['safesieve','card','agentnet']:
                    fit_dataset='mmlu' if dataset=='humaneval' else dataset
                    checkpoint=checkpoints/fit_dataset/method/('checkpoint.pkl' if method=='agentnet' else 'checkpoint.pt')
                    if not checkpoint.exists() and not a.plan_only:raise RuntimeError(f'Run fitting first: {checkpoint}')
                    command+=['--checkpoint',str(checkpoint)]
                jobs.append((dataset,method,seed,command))
    import random
    random.Random(97).shuffle(jobs)
    if a.plan_only:
        import shlex
        for _,_,_,command in jobs:print(shlex.join(command))
        print('JOBS:',len(jobs));return
    logs=ROOT/'logs'/slug/a.stage;logs.mkdir(parents=True,exist_ok=True)
    def run(job):
        dataset,method,seed,command=job
        path=logs/f'{dataset}_{method}_{seed}.log'
        with path.open('a') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Job failed; inspect {path}')
        print('completed',dataset,method,seed,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs_per_gpu) as pool:
        active={};pending=iter(jobs)
        for _ in range(a.jobs_per_gpu):
            job=next(pending,None)
            if job:active[pool.submit(run,job)]=job
        while active:
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                del active[future];future.result()
                job=next(pending,None)
                if job:active[pool.submit(run,job)]=job

if __name__=='__main__':main()
