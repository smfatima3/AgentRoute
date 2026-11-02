"""Explicit compatibility attestation for existing, locally generated fitted state."""
import json,pickle
from .core import file_hash
from .jobs import ROOT

def load(method,config,manifest):
    import torch
    approval=json.loads((ROOT/'checkpoint_reuse.json').read_text(encoding='utf-8'))
    for name in approval['critical_adapter_sources_byte_identical']:
        if file_hash(ROOT/'agentroute'/name)!=file_hash(ROOT/'imported_v9/agentroute'/name):raise RuntimeError('Critical adapter changed after compatibility audit')
    for key in approval['matching_config_fields']:
        if config[key]!=approval['original_config'][key]:raise RuntimeError('Checkpoint-compatible field changed: '+key)
    path=ROOT/'imported_v9/checkpoints/Qwen3-1.7B/mmlu'/method/('checkpoint.pkl' if method=='agentnet' else 'checkpoint.pt')
    meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if file_hash(path)!=meta['checkpoint_sha256'] or meta['model_revision']!=manifest['models']['Qwen/Qwen3-1.7B']:
        raise RuntimeError('Checkpoint/model integrity mismatch')
    if meta['dataset']!='mmlu' or meta['method']!=method:raise RuntimeError('Wrong source checkpoint')
    oldtrain=ROOT/'imported_v9/data/mmlu/train.jsonl'
    if file_hash(oldtrain)!=meta['data_files']['mmlu/train.jsonl']:raise RuntimeError('Original fitting inputs changed')
    if method=='agentnet':
        with path.open('rb') as f:state=pickle.load(f)
    else:state=torch.load(path,map_location='cpu',weights_only=True)
    return state,meta
