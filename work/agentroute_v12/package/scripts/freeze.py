"""Freeze settings and evaluation IDs before examining any held-out predictions."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentroute.core import digest,file_hash,write_json
from agentroute.jobs import ROOT,code_hashes,select_rows,verify_upstream

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/study.json')
    p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--limit',type=int,default=80)
    p.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints');p.add_argument('--output',type=Path,default=ROOT/'protocol.lock.json')
    p.add_argument('--upstream',type=Path,default=ROOT/'upstream')
    a=p.parse_args()
    if a.output.exists():raise RuntimeError('Protocol already frozen; do not overwrite it')
    config=json.loads(a.config.read_text());manifest=json.loads((a.data/'manifest.json').read_text())
    hashes=code_hashes();files=[]
    for method in set(config['methods'])&{'safesieve','card','agentnet','selforg'}:
        verify_upstream(method,a.upstream)
    for model in config['models']:
        for dataset in [d for d in config['datasets'] if d!='humaneval']:
            for method in set(config['methods'])&{'safesieve','card','agentnet'}:
                path=a.checkpoints/model.split('/')[-1]/dataset/method/('checkpoint.pkl' if method=='agentnet' else 'checkpoint.pt')
                if not path.is_file() or not path.with_suffix('.json').is_file():
                    raise RuntimeError(f'Missing fitted checkpoint/provenance: {dataset}/{method}. Finish fitting before freeze.')
                meta=json.loads(path.with_suffix('.json').read_text())
                if (meta['checkpoint_sha256']!=file_hash(path) or meta['config_hash']!=digest(config) or
                    meta['source_hashes']!=hashes or meta['method']!=method or meta['dataset']!=dataset or
                    meta['model']!=model or meta['model_revision']!=manifest['models'][model] or
                    meta['data_files']!={f'{dataset}/train.jsonl':manifest['files'][f'{dataset}/train.jsonl']}):
                    raise RuntimeError(f'Changed or mismatched fitted checkpoint: {dataset}/{method}')
                files.append(path)
    write_json(a.output,{'config_hash':digest(config),'code_hashes':code_hashes(),'data_hash':file_hash(a.data/'manifest.json'),
        'test_ids':{d:[r['id'] for r in select_rows(a.data,d,'test',a.limit)] for d in config['datasets']},
        'checkpoint_hashes':[file_hash(f) for f in files],
        'noninferiority_margin':config['noninferiority_margin'],'statistical_unit':'question, seeds averaged within question',
        'native_baseline_parity':'not_yet_demonstrated','claim_status':'exploratory_until_native_parity_and_dev_selection'} )
    print(a.output)
