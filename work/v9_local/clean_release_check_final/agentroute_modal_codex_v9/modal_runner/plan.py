"""Pure local validation and planning. Importing this module creates no cloud resources."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
MODELS={'1.7b': ('1.7B','1_7b',1,'e5c50d47e97925820c9bad820284e5c15ff304bc0671deb480b9f69cb8000141')}
GPU_STAGES={'smoke','fit','dev','test','compare'}
STAGES=sorted(GPU_STAGES|{'freeze','analyze','export'})
EXTERNAL={'safesieve','card','agentnet','selforg'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unpack(model):
    label,slug,_,expected=MODELS[model]
    archive=BASE/'packages'/f'AgentRoute_Qwen3_{label}_v9.zip'
    if sha(archive)!=expected:raise RuntimeError('Bundled package hash differs from the verified v9 release')
    dirname=f'agentroute_qwen3_{slug}_v9'
    parent=BASE/'.prepared';root=parent/dirname
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read(dirname+'/MANIFEST.sha256.json'))
        if not root.exists():
            for member in z.infolist():
                dest=(parent/member.filename).resolve()
                if not dest.is_relative_to(parent.resolve()) or Path(member.filename).parts[0]!=dirname:
                    raise RuntimeError('Unsafe ZIP member')
                if (member.external_attr >> 16) & 0o170000 == 0o120000:raise RuntimeError('ZIP symlink refused')
            z.extractall(parent)
    if json.loads((root/'MANIFEST.sha256.json').read_text())!=manifest:
        raise RuntimeError('Prepared cache belongs to an older package. Remove .prepared to regenerate it; preserve runs and campaign data.')
    for rel,digest in manifest.items():
        if sha(root/rel)!=digest:raise RuntimeError(f'Prepared package changed: {rel}')
    info=json.loads((root/'PACKAGE_INFO.json').read_text())
    if info['setup_fix']!='2026-10-28-local-vote-v9':raise RuntimeError('Wrong package generation')
    return root

def make_plan(model,stage,methods=None,gpus=None,hours=2,limit=None,campaign='v9-local',datasets=None):
    if model not in MODELS:raise ValueError('Only the 1.7b model is enabled in v9')
    if stage not in STAGES:raise ValueError('Unknown stage')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}',campaign):raise ValueError('Campaign must be 1-32 lowercase letters/digits/hyphens')
    if not 0<hours<=24:raise ValueError('Hours must be greater than 0 and at most 24')
    root=unpack(model);c=json.loads((root/'configs/study.json').read_text())
    allowed=['cascade'] if stage=='smoke' else ['safesieve','card','agentnet'] if stage=='fit' else c['controls'] if stage=='controls' else c['methods']
    if methods and stage in {'smoke','freeze','analyze','export'}:raise ValueError('--methods is only for fit/dev/test/controls/compare')
    if methods and (len(set(methods))!=len(methods) or not set(methods)<=set(allowed)):raise ValueError('Invalid/duplicate methods')
    if limit is not None and (limit<0 or stage in {'fit','analyze','export'}):raise ValueError('Invalid --limit for stage')
    selected=methods or allowed
    selected_datasets=datasets or (['mmlu'] if stage in {'smoke','compare'} else c['datasets'])
    if datasets and stage in {'smoke','freeze','analyze','export'}:raise ValueError('--datasets is only for fit/dev/test/controls/compare')
    if len(set(selected_datasets))!=len(selected_datasets) or not set(selected_datasets)<=set(c['datasets']):raise ValueError('Invalid/duplicate datasets')
    if stage=='compare' and 'humaneval' in selected_datasets:raise ValueError('HumanEval has no development set')
    count=gpus if gpus is not None else MODELS[model][2]
    if count not in [1,2]:raise ValueError('Use 1 or 2 GPUs')
    if model=='14b' and count!=2:raise ValueError('This 14B deployment uses two A100 40GB GPUs')
    use_gpu=stage in GPU_STAGES
    if limit is None:limit=2 if stage=='smoke' else 50 if stage in {'compare','dev'} else 80 if stage in {'freeze','test'} else 16
    if stage=='compare' and limit<=0:raise ValueError('Compare requires a positive question limit')
    return {'model_key':model,'model':c['models'][0],'package_sha256':MODELS[model][3],
        'revision':json.loads((root/'data/manifest.json').read_text())['models'][c['models'][0]],
        'stage':stage,'methods':selected,'explicit_partial_suite':bool(methods),'limit':limit,
        'datasets':selected_datasets,'explicit_dataset_selection':bool(datasets),
        'gpu':f'A100-40GB:{count}' if use_gpu else None,'gpu_count':count,'tensor_parallel_size':count,
        'max_model_len':32768,'clients_per_replica':2,'timeout_s':int(hours*3600),'campaign':campaign,
        'volume_name':f'agentroute-codex-{model.replace(".","-")}-{campaign}',
        'cpu':8,'memory_mib':32768,'client_runtime':'vm','server_runtime':'gvisor',
        'required_sources':sorted(set(selected)&EXTERNAL) if stage not in {'analyze','export'} else [],
        'needs_docker':False,'cloud_resources_created':False}

def commands(plan,root,python):
    """Only commands that invoke the original package. No scientific code is changed."""
    root=Path(root);model=plan['model'];stage=plan['stage']
    if stage=='smoke':return [[python,str(root/'scripts/run.py'),'--model',model,'--dataset','mmlu',
        '--method','cascade','--split','dev','--limit',str(plan['limit']),'--seed','17',
        '--output',str(root/'setup/smoke_results'/plan.get('operation_id','preview')),'--require-format']]
    if stage in {'fit','dev','test','controls'}:
        cmd=[python,str(root/'scripts/suite.py'),'--model',model,'--stage',stage,'--jobs-per-gpu','2']
        if stage!='fit':cmd+=['--limit',str(plan['limit'])]
        if plan['explicit_partial_suite']:cmd+=['--methods',*plan['methods']]
        if plan['explicit_dataset_selection']:cmd+=['--datasets',*plan['datasets']]
        return [cmd]
    if stage=='compare':
        cmd=[python,str(root/'scripts/compare.py'),'--model',model,'--limit',str(plan['limit']),
             '--datasets',*plan['datasets']]
        if plan['explicit_partial_suite']:cmd+=['--methods',*plan['methods']]
        return [cmd]
    if stage=='freeze':return [[python,str(root/'scripts/freeze.py'),'--limit',str(plan['limit'])]]
    if stage=='analyze':return [[python,str(root/'scripts/analyze.py')],
        [python,str(root/'scripts/plot_results.py'),'--analysis','analysis.json','--output','figures']]
    return []
