"""Trusted CPU VM runner. Original AgentRoute scripts and Docker evaluator are unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modal_runner.plan import commands
from modal_runner.proxy import start_proxy

WORK=Path('/work');ROOT=WORK/'project';PY='/opt/client/bin/python'

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def run(cmd):
    # Print commands only from our fixed command builders, which contain no credentials.
    print('RUN:',json.dumps([str(x) for x in cmd]),flush=True)
    child=subprocess.Popen([str(x) for x in cmd],cwd=ROOT,start_new_session=True)
    save(WORK/'active_process.json',{'pid':child.pid})
    try:
        code=child.wait()
        if code:raise subprocess.CalledProcessError(code,[str(x) for x in cmd])
    finally:(WORK/'active_process.json').unlink(missing_ok=True)

def cancel():
    path=WORK/'active_process.json'
    if path.exists():
        pid=json.loads(path.read_text())['pid']
        try:os.killpg(pid,signal.SIGTERM)
        except ProcessLookupError:pass

def initialize(plan):
    if not ROOT.exists():shutil.copytree('/opt/package',ROOT)
    source_manifest=json.loads(Path('/opt/package/MANIFEST.sha256.json').read_text())
    for rel,expected in source_manifest.items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Persisted package has changed: {rel}. Use a new campaign for a changed revision.')
    fixed={k:plan[k] for k in ['model','package_sha256','revision','gpu_count','tensor_parallel_size','max_model_len',
                              'clients_per_replica','cpu','memory_mib','client_runtime','server_runtime','launcher_sha256']}
    path=WORK/'deployment.json'
    if path.exists() and json.loads(path.read_text())!=fixed:
        raise RuntimeError('Deployment settings changed within this campaign. Choose a new --campaign.')
    if not path.exists():save(path,fixed)
    (ROOT/'logs').mkdir(exist_ok=True)
    (ROOT/'setup').mkdir(exist_ok=True)

def ensure_docker():
    if subprocess.run(['docker','info'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        log=(ROOT/'logs/dockerd.log').open('a')
        subprocess.Popen(['dockerd','--host=unix:///var/run/docker.sock','--storage-driver=vfs'],
                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close()
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            if subprocess.run(['docker','info'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:break
            time.sleep(1)
        else:raise RuntimeError('Docker daemon failed in the CPU VM; inspect logs/dockerd.log')
    image_file=WORK/'evaluator.json'
    if image_file.exists():image=json.loads(image_file.read_text())['image']
    else:image='python:3.11-slim'
    run(['docker','pull',image])
    if '@sha256:' not in image:
        digests=json.loads(subprocess.check_output(['docker','image','inspect','--format','{{json .RepoDigests}}',image],text=True))
        image=next(x for x in digests if x.startswith('python@sha256:'))
        save(image_file,{'image':image,'resolved_from':'python:3.11-slim'})
    # Infrastructure test only; no model-generated code and no benchmark score.
    run(['docker','run','--rm','--network','none','--read-only','--cap-drop','ALL',
         '--security-opt','no-new-privileges','--pids-limit','64','--memory','512m','--cpus','1',
         '--user','65534:65534',image,'python','-I','-B','-c','print("ISOLATED_EVALUATOR_READY")'])
    return image

def prepare(plan):
    if plan['stage']=='export':
        if not ROOT.is_dir():raise RuntimeError('No project exists in this campaign to export')
        return
    initialize(plan)
    run([PY,ROOT/'scripts/prepare_data.py'])
    sources=plan['required_sources']
    if sources:
        cmd=[PY,ROOT/'scripts/bootstrap.py','--methods',*sources]
        for name in sources:
            source=Path('/opt/provided_upstream')/name
            if (source/'.git').is_dir():cmd+=['--source',f'{name}={source}']
        run(cmd)
    if plan['stage'] in ['freeze','test']:
        # Freeze only a complete comparison; test selections remain explicitly recorded.
        methods=['safesieve','card','agentnet'] if plan['stage']=='freeze' else set(plan['methods'])&{'safesieve','card','agentnet'}
        for method in methods:
            for dataset in ['mmlu','svamp','gsm8k','bbh']:
                suffix='checkpoint.pkl' if method=='agentnet' else 'checkpoint.pt'
                p=ROOT/'checkpoints'/plan['model'].split('/')[-1]/dataset/method/suffix
                if not p.is_file():raise RuntimeError(f'Missing fitted checkpoint: {p}. Run fitting first.')
        if plan['stage']=='test' and not (ROOT/'protocol.lock.json').is_file():
            raise RuntimeError('Run the separate freeze stage after reviewing development results.')
    if plan['stage'] in ['dev','test']:
        for method in set(plan['methods'])&{'safesieve','card','agentnet'}:
            for dataset in dict.fromkeys('mmlu' if d=='humaneval' else d for d in plan['datasets']):
                suffix='checkpoint.pkl' if method=='agentnet' else 'checkpoint.pt'
                if not (ROOT/'checkpoints'/plan['model'].split('/')[-1]/dataset/method/suffix).is_file():
                    raise RuntimeError(f'Run fitting first: {dataset}/{method}')
    if plan['needs_docker']:ensure_docker()
    save(WORK/'prepared.json',{'status':'passed','stage':plan['stage'],'gpu_requested_yet':False})

def execute(plan):
    initialize(plan);proxy=None
    try:
        if plan['gpu']:
            endpoint=os.environ['AGENTROUTE_REMOTE_ORIGIN'];key=os.environ['AGENTROUTE_REMOTE_KEY']
            proxy=start_proxy(endpoint,key)
            run([PY,ROOT/'scripts/check_server.py','--model',plan['model']])
        if plan['needs_docker']:
            # The same VM from prepare is still running; its daemon and pinned image are reused.
            os.environ['AGENTROUTE_EVAL_IMAGE']=json.loads((WORK/'evaluator.json').read_text())['image']
        for command in commands(plan,ROOT,PY):run(command)
    finally:
        if proxy:proxy.shutdown();proxy.server_close()

def export(plan):
    directory=WORK/'exports'/plan['operation_id'];directory.mkdir(parents=True,exist_ok=True)
    target=directory/'AgentRoute_results.zip';temporary=target.with_suffix('.zip.tmp')
    skip={'upstream','experiment_env','serving_env','__pycache__','.pytest_cache','.git'}
    manifest={}
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in ROOT.rglob('*'):
            rel=path.relative_to(ROOT)
            if path.is_file() and not any(x in skip for x in rel.parts) and path.suffix!='.pyc':
                name='project/'+rel.as_posix();z.write(path,name)
                manifest[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*WORK.glob('*.json'),*(WORK/'operations').glob('*.json')]:
            if path.name in {'latest_export.json','active_process.json'}:continue
            name='deployment/'+path.relative_to(WORK).as_posix();z.write(path,name)
            manifest[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        z.writestr('EXPORT_MANIFEST.sha256.json',json.dumps(manifest,indent=2)+'\n')
    temporary.replace(target)
    metadata={'operation_id':plan['operation_id'],'path':target.relative_to(WORK).as_posix(),
              'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'files':len(manifest),'status':'ready'}
    save(directory/'export.json',metadata)
    print('EXPORT_READY:',target,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','execute','export','cancel']);p.add_argument('--plan',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    if a.action=='cancel':cancel();return
    if a.action=='export':export(plan);return
    operation=WORK/'operations'/f'{plan["operation_id"]}-{a.action}.json'
    record={'plan':plan,'action':a.action,'status':'running'};save(operation,record)
    try:
        (prepare if a.action=='prepare' else execute)(plan);record['status']='passed'
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:save(operation,record)

if __name__=='__main__':main()
