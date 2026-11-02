"""Immutable data selection, model resources, and per-job reproducibility."""
import json, random, subprocess, sys
from pathlib import Path
import numpy as np
from .core import digest,file_hash,read_rows

ROOT=Path(__file__).resolve().parents[1]

def select_rows(data_dir,dataset,split,limit):
    manifest=json.loads((data_dir/'manifest.json').read_text())
    relative=f'{dataset}/{split}.jsonl';path=data_dir/relative
    if file_hash(path)!=manifest['files'][relative]:raise ValueError('Dataset hash mismatch')
    rows=read_rows(path)
    groups={}
    for row in rows:groups.setdefault(row.get('stratum','all'),[]).append(row)
    for group in groups.values():group.sort(key=lambda r:digest([97,r['id']]))
    ordered=[]
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:ordered.append(groups[key].pop(0))
    return ordered if limit==0 else ordered[:limit]

def set_seed(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def resources(config,manifest,model,with_embedding=True):
    if model not in config['models']+config.get('optional_models',[]):raise ValueError('Model is not enabled in this package')
    from transformers import AutoTokenizer
    if model not in manifest['models']:raise ValueError('Unpinned model')
    tokenizer=AutoTokenizer.from_pretrained(model,revision=manifest['models'][model],trust_remote_code=False)
    if not with_embedding:return tokenizer,None
    from sentence_transformers import SentenceTransformer
    emb=config['embedding']['model']
    embedding=SentenceTransformer(emb,revision=manifest['models'][emb],device=config['embedding']['device'],trust_remote_code=False)
    return tokenizer,embedding

def verify_upstream(method,path):
    spec=json.loads((ROOT/'configs/upstream.json').read_text())[method]
    repo=path/method
    if not (repo/'.git').exists():
        raise RuntimeError(f'Missing upstream {method}: {repo}. Run scripts/bootstrap.py and inspect setup/upstream_status.json. The full suite cannot proceed without its pinned source.')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if head!=spec['commit']:raise RuntimeError('Wrong upstream commit')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip():
        raise RuntimeError('Modified upstream files; record a new adapter version')
    return spec

def code_hashes():
    return {str(p.relative_to(ROOT)):file_hash(p) for directory in ['agentroute','scripts','configs']
            for p in sorted((ROOT/directory).rglob('*')) if p.suffix in ['.py','.json']}

def stable_job_identity(identity):
    # Measured initialization durations change on every process start; they are
    # provenance, not the experiment cell key. Including them prevented resume.
    return {k:v for k,v in identity.items() if k!='initialization_embedding_events'}

def checkpoint_load(path,method,model,config,manifest):
    import pickle,torch
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    if meta['checkpoint_sha256']!=file_hash(path):raise ValueError('Changed checkpoint')
    if meta['method']!=method or meta['model']!=model or meta['model_revision']!=manifest['models'][model]:raise ValueError('Checkpoint model/method mismatch')
    if meta['config_hash']!=digest(config):raise ValueError('Checkpoint settings differ')
    if meta['source_hashes']!=code_hashes():raise ValueError('Code changed after fitting')
    for relative,sha in meta['data_files'].items():
        if manifest['files'].get(relative)!=sha:raise ValueError('Checkpoint training data changed')
    # AgentNet contains upstream dataclasses. Load only a locally produced,
    # hash-verified checkpoint, never a downloaded arbitrary pickle.
    if method=='agentnet':
        with path.open('rb') as f:state=pickle.load(f)
    else:state=torch.load(path,map_location='cpu',weights_only=True)
    return state,meta
