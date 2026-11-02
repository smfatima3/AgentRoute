"""Read-only audit of v11, with immutable v12 copies and independent rescoring."""
import collections, hashlib, json, shutil, sys, statistics, functools
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OLD=BASE/'outputs/agentroute_compact_v11/runs/20261030T163201-finish/raw_export/project'
OUT=BASE/'outputs/agentroute_v12_completion'
sys.path.insert(0,str(OLD))
from agentroute.evaluation import score_text

@functools.cache
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(o,indent=2,ensure_ascii=False)+'\n').encode()
    if p.exists() and p.read_bytes()==raw:return
    temp=p.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(p)
def copy(p):
    dest=OUT/'reuse/v11'/p.relative_to(OLD);dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copyfile(p,dest)
    if sha(p)!=sha(dest):raise RuntimeError('Immutable reuse content mismatch')
    return dict(path=dest.relative_to(OUT).as_posix(),sha256=sha(dest))

def main():
    manifest=read(OLD/'compact_data/manifest.json');config=read(OLD/'configs/compact.json')
    datasets={}
    for rel,h in manifest['files'].items():
        p=OLD/'compact_data'/rel
        assert sha(p)==h,rel
        datasets[rel]={r['id']:r for r in map(json.loads,p.read_text(encoding='utf8').splitlines())}
        copy(p)
    copy(OLD/'compact_data/manifest.json');copy(OLD/'configs/compact.json')
    accepted=[];rejected=[];seen=set();timing=collections.defaultdict(list)
    for p in sorted((OLD/'results').glob('*/*.json')):
        r=read(p)
        if not {'id','status','attempt_id','dataset','method'}<=r.keys():continue
        evidence=copy(p);job=read(p.parent/'job.json');jobref=copy(p.parent/'job.json')
        if r['status']!='ok' or not r.get('usage_complete'):
            rejected.append(dict(attempt=r['attempt_id'],reason='infrastructure/incomplete usage',evidence=evidence));continue
        row=datasets[f"{r['dataset']}/{r['split']}.jsonl"][r['id']]
        rescored=score_text(r['final_text'],row)
        assert all(rescored[k]==r[k] for k in ['quality','format_ok','prediction']),p
        assert job['config_sha256']==sha(OLD/'configs/compact.json'),p
        assert job['data_sha256']==manifest['files'][f"{r['dataset']}/{r['split']}.jsonl"],p
        assert job['model_revision']==manifest['models'][config['models'][0]],p
        for name,h in job['source_hashes'].items():assert sha(OLD/name)==h,(p,name)
        assert all(c['status']=='ok' and isinstance(c['prompt_tokens'],int) and isinstance(c['completion_tokens'],int) and c.get('response_id') for c in r['calls']),p
        assert r['total_online_tokens']==sum(c['prompt_tokens']+c['completion_tokens'] for c in r['calls']),p
        identity=(r['dataset'],r['split'],r['method'],r['seed'],r['id'])
        assert identity not in seen,identity
        seen.add(identity);timing[(r['dataset'],r['method'])].append(r)
        accepted.append(dict(identity=list(identity),attempt_id=r['attempt_id'],phase=r['phase'],score=rescored,raw=evidence,job=jobref,
          source_hashes_verified=True,usage_reconciled=True,generation_reuse_eligible=True,
          telemetry_reuse={'quality_tokens_calls_logical_deliveries':True,'new_device_live_latency':False,
            'physical_link_bytes':False,'per_request_serving_queue':False,'per_stage_routing_cost':False},
          scientific_note='Original policy unchanged; historical selected-panel evidence, not a fresh confirmatory rerun.'))
        journal=p.with_name(p.stem+'.journal.jsonl')
        if journal.exists():copy(journal)
    fits=[]
    test_ids={k for rel,rows in datasets.items() if rel.endswith('/test.jsonl') for k in rows}
    for method in ['safesieve','card','agentnet']:
        root=OLD/'imported_v9/checkpoints/Qwen3-1.7B/mmlu'/method;meta=read(root/'checkpoint.json')
        checkpoint=root/('checkpoint.pkl' if method=='agentnet' else 'checkpoint.pt')
        assert sha(checkpoint)==meta['checkpoint_sha256']
        train=[read(p) for p in sorted(root.glob('train_*.json'))]
        iterations=[read(p) for p in sorted(root.glob('iteration_*.json'))]
        assert len(train)==120 and len(iterations)==30
        assert len({r['id'] for r in train})==120 and not test_ids.intersection(r['id'] for r in train)
        assert all(r['status']=='ok' and all(c['status']=='ok' for c in r['calls']) for r in train)
        assert collections.Counter(r['iteration'] for r in train)=={i:4 for i in range(30)}
        if method=='card':assert all(r['optimizer_diagnostics'].get('weights_changed') and r['optimizer_diagnostics'].get('gradient_l1',0)>0 for r in iterations)
        refs=[copy(p) for p in root.iterdir() if p.is_file()]
        fits.append(dict(method=method,train_questions=120,iterations=30,batch_size=4,test_ids_disjoint=True,
          binary_hash_verified=True,checkpoint=copy(checkpoint),metadata=meta,
          initialization=read(root/'initialization.json') if (root/'initialization.json').exists() else None,
          iteration_records=iterations,evidence=refs,
          successful_trajectory_tokens=sum(c['prompt_tokens']+c['completion_tokens'] for r in train for c in r['calls']),
          status='raw_recipe_verified_requires_adapter_load_validation'))
    estimate=[]
    for (d,m),rows in sorted(timing.items()):
        estimate.append(dict(dataset=d,method=m,n=len(rows),mean_question_wall_s=statistics.mean(r['wall_s'] for r in rows),
          mean_tokens=statistics.mean(r['total_online_tokens'] for r in rows)))
    report=dict(status='AUDITED_REUSE_CANDIDATES_NOT_YET_CAMPAIGN_RECEIPTS',generation_count=len(accepted),
      by_phase=dict(collections.Counter(r['phase'] for r in accepted)),generation=accepted,rejected_attempts=rejected,
      fitting=fits,observed_throughput=estimate,transport_accepted=0,
      transport_reason='v11 did not satisfy v12 measured physical shaping and >=1000 scheduled arrivals per repeat.',
      source_note='All successful raw rows independently rescored; exact old model/config/data/source hashes verified. No test-answer policy tuning performed.')
    save(OUT/'REUSE_AUDIT.json',report)
    print(json.dumps({k:report[k] for k in ['status','generation_count','by_phase','transport_accepted','observed_throughput']}))
if __name__=='__main__':main()
