"""Prepare prespecified untouched panels and verify historical fitting provenance."""
import collections,hashlib,importlib.util,json,sys,urllib.request,zipfile
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent.parent
sys.path.insert(0,str(ROOT))
from agentroute.core import digest,file_hash,read_json,read_rows,write_json,write_rows
SEED=20261029

def fetch(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'AgentRoute-compact'}),timeout=90) as r:b=r.read()
        path.write_bytes(b)
    return path

def question_key(q):return digest(' '.join(q.split()).casefold())

def pick(rows,n,stratum='stratum'):
    groups=collections.defaultdict(list)
    for r in rows:groups[str(r.get(stratum,'all'))].append(r)
    for key in groups:groups[key].sort(key=lambda r:digest([SEED,r['id']]))
    keys=sorted(groups,key=lambda k:digest([SEED,k]));selected=[]
    while len(selected)<n and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected)<n:selected.append(groups[key].pop(0))
    if len(selected)!=n:raise ValueError('Insufficient uncontaminated candidates')
    return selected

def main():
    out=ROOT/'compact_data'
    if out.exists():raise RuntimeError('Prepared inputs are immutable; inspect existing output')
    prior_ids=set();prior_questions=set();evidence=[]
    for prefix in ['agentroute_modal_codex','updated_v8','v9_local','v10_builds','agentroute_v10_evidence']:
        for manifest_path in (BASE/'work'/prefix).rglob('manifest.json'):
            try:m=read_json(manifest_path)
            except (ValueError,OSError):continue
            if not isinstance(m,dict) or not isinstance(m.get('files'),dict):continue
            for rel in m['files']:
                if rel.split('/')[0] not in ['mmlu','svamp','math','bbh'] or Path(rel).stem not in ['train','dev','calibration']:continue
                file=manifest_path.parent/rel
                if not file.is_file():continue
                for row in read_rows(file):
                    prior_ids.add(row['id']);prior_questions.add(question_key(row['question']))
                evidence.append({'path':str(file),'sha256':file_hash(file)})
    for directory in (BASE/'outputs').iterdir():
        if directory.name=='agentroute_compact_v11' or not directory.is_dir():continue
        for path in directory.rglob('*.json'):
            if path.name in ['job.json','manifest.json'] or path.stat().st_size>5000000:continue
            try:r=read_json(path)
            except (ValueError,OSError,UnicodeError):continue
            if isinstance(r,dict) and r.get('dataset') in ['mmlu','svamp','math','bbh'] and isinstance(r.get('id'),str):
                prior_ids.add(r['id'])
                if r.get('question'):prior_questions.add(question_key(r['question']))
    def untouched(r):return r['id'] not in prior_ids and question_key(r['question']) not in prior_questions
    revision=read_json(ROOT/'source_metadata/mmlu.json')['sha']
    path=fetch(f'https://huggingface.co/datasets/cais/mmlu/resolve/{revision}/all/test-00000-of-00001.parquet',ROOT/'source_data/mmlu_test.parquet')
    mmlu=[]
    for i,r in enumerate(pq.read_table(path).to_pylist()):
        q=r['question']+'\n'+'\n'.join(f'{chr(65+j)}. {v}' for j,v in enumerate(r['choices']))
        mmlu.append(dict(id=f'mmlu/test/{i}',dataset='mmlu',question=q,answers=[chr(65+int(r['answer']))],metric='mcq',stratum=r['subject'],
            context=[],source={'dataset':'cais/mmlu','revision':revision,'split':'test','row':i}))
    svrev=read_json(ROOT/'source_metadata/svamp.json')['sha']
    path=fetch(f'https://raw.githubusercontent.com/arkilpatel/SVAMP/{svrev}/SVAMP.json',ROOT/'source_data/SVAMP.json')
    svamp=[]
    for i,r in enumerate(read_json(path)):
        svamp.append(dict(id='svamp/challenge/'+str(r['ID']),dataset='svamp',question=str(r['Body']).strip()+' '+str(r['Question']).strip(),
            answers=[str(r['Answer'])],metric='numeric',stratum=r.get('Type','all'),context=[],
            source={'dataset':'arkilpatel/SVAMP','revision':svrev,'split':'official_challenge_set','row':i,'source_id':r['ID']}))
    math=read_rows(ROOT/'evidence_data/math/test.jsonl');bbh=read_rows(ROOT/'evidence_data/bbh/test.jsonl')
    for r in math:r['panel_stratum']=r['stratum']+'/'+r['level']
    panels={};pool_counts={};seen=set(prior_questions)
    for name,pool,n,field in [('mmlu',mmlu,100,'stratum'),('svamp',svamp,60,'stratum'),('math',math,50,'panel_stratum'),('bbh',bbh,50,'stratum')]:
        eligible=[r for r in pool if untouched(r) and question_key(r['question']) not in seen]
        unique={}
        for r in eligible:unique.setdefault(question_key(r['question']),r)
        eligible=list(unique.values());pool_counts[name]=len(eligible)
        panels[name]=pick(eligible,n,field)
        for r in panels[name]:r['split']='test';seen.add(question_key(r['question']))
    math_dev=read_rows(ROOT/'evidence_data/math/dev.jsonl')
    known=next(r for r in math_dev if r['id']=='math/train/number_theory/418')
    development={'math':[known]+pick([r for r in math_dev if r['id']!=known['id']],19),
        'bbh':pick(read_rows(ROOT/'evidence_data/bbh/dev.jsonl'),10)}
    files={};counts={}
    for dataset,rows in panels.items():
        rel=dataset+'/test.jsonl';write_rows(out/rel,rows);files[rel]=file_hash(out/rel);counts[rel]=len(rows)
    for dataset,rows in development.items():
        assert not {question_key(r['question']) for r in rows}&{question_key(v['question']) for values in panels.values() for v in values}
        rel=dataset+'/dev.jsonl';write_rows(out/rel,rows);files[rel]=file_hash(out/rel);counts[rel]=len(rows)
    ablation={d:[r['id'] for r in pick(panels[d],20)] for d in ['mmlu','math']}
    old=read_json(ROOT/'evidence_data/manifest.json')
    manifest={'schema':3,'sample_seed':SEED,'models':old['models'],'files':files,'counts':counts,
        'panel_ids':{d:[r['id'] for r in rows] for d,rows in panels.items()},'ablation_ids':ablation,
        'coverage':{d:dict(collections.Counter(r.get('panel_stratum',r.get('stratum','all')) for r in rows)) for d,rows in panels.items()},
        'weighting':'Equal weight per selected question within each dataset; no pooled cross-dataset significance claim',
        'eligible_pool_counts':pool_counts,'prior_exclusion_ids':sorted(prior_ids),'prior_question_hash_count':len(prior_questions),
        'prior_input_sources':evidence,'svamp_source_revision':svrev,'svamp_split_note':'Custom untouched panel from official challenge release; not an official test split',
        'mmlu_source_revision':revision,'raw_sources':{p.relative_to(ROOT).as_posix():file_hash(p) for p in (ROOT/'source_data').glob('*') if p.is_file()},
        'test_inference_performed':False,'selection_precedes_predictions':True}
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'counts':counts,'eligible_pool_counts':pool_counts,'coverage_counts':{d:len(v) for d,v in manifest['coverage'].items()}},indent=2))

if __name__=='__main__':main()
