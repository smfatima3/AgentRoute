"""Real datasets, immutable revisions, deterministic disjoint splits, no model calls."""
import argparse, collections, hashlib, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentroute.core import digest, file_hash, write_json, write_rows

ROOT = Path(__file__).resolve().parents[1]
REPOS = {'mmlu':'cais/mmlu','humaneval':'openai/openai_humaneval',
         'svamp':'ChilleD/SVAMP','gsm8k':'openai/gsm8k'}

def rank(rows, seed):
    return sorted(rows, key=lambda r: digest([seed,r['id']]))

def stratified(rows, n, seed):
    groups = collections.defaultdict(list)
    for row in rows: groups[row.get('stratum','all')].append(row)
    groups = {k:rank(v,seed) for k,v in sorted(groups.items())}
    result=[]
    while len(result)<n and any(groups.values()):
        for group in groups.values():
            if group and len(result)<n: result.append(group.pop(0))
    return result

def normalize(name, ds, revision, split):
    result=[]
    for i,r in enumerate(ds):
        source_id = str(r.get('task_id',r.get('ID',i)))
        base={'id':f'{name}/{split}/{source_id}', 'dataset':name,
              'source':{'dataset':REPOS[name],'revision':revision,'split':split,'row':i,'source_id':source_id}}
        if name=='mmlu':
            q=r['question']+'\n'+'\n'.join(f'{chr(65+j)}. {v}' for j,v in enumerate(r['choices']))
            base.update(question=q,answers=[chr(65+int(r['answer']))],metric='mcq',stratum=r['subject'])
        elif name=='humaneval':
            base.update(question=r['prompt'], answers=[], metric='code',
                        test=r['test'],entry_point=r['entry_point'])
        elif name=='svamp':
            q=str(r['Body']).strip()+' '+str(r['Question']).strip()
            base.update(question=q,answers=[str(r['Answer'])],metric='numeric')
        else:
            base.update(question=r['question'],answers=[r['answer'].split('####')[-1].strip()],metric='numeric')
        result.append(base)
    return result

def prepare(out, bbh_dir):
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        for path, sha in manifest['files'].items():
            if file_hash(out/path)!=sha: raise RuntimeError(f'Changed frozen data: {path}')
        print('Existing immutable data verified:',out); return
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Partial output exists; choose a fresh --output to preserve evidence')
    out.mkdir(parents=True,exist_ok=True)
    from huggingface_hub import HfApi
    from datasets import load_dataset
    api=HfApi(); revisions={}; partitions={}
    for name,repo in REPOS.items():
        info=api.dataset_info(repo); revisions[name]={'repo':repo,'revision':info.sha,
          'license':(getattr(info,'card_data',None) or {}).get('license','unspecified')}
        ds=load_dataset(repo,'all' if name=='mmlu' else 'main' if name=='gsm8k' else None,revision=info.sha)
        splits={s:normalize(name,d,info.sha,s) for s,d in ds.items() if s!='auxiliary_train'}
        if name=='mmlu':
            partitions[name]={'train':splits['dev'],'dev':stratified(splits['validation'],285,97),'test':stratified(splits['test'],570,97)}
        elif name=='humaneval': partitions[name]={'train':[],'dev':[],'test':splits['test']}
        else:
            training=rank(splits['train'],97)
            ntrain=120 if name=='gsm8k' else min(600,len(training)-100)
            partitions[name]={'train':training[:ntrain],'dev':training[ntrain:ntrain+100],
                              'test':stratified(splits['test'],500 if name=='gsm8k' else len(splits['test']),97)}
        print('loaded',name,{k:len(v) for k,v in partitions[name].items()},flush=True)
    if not bbh_dir.exists():
        subprocess.run(['git','clone','--depth','1','https://github.com/suzgunmirac/BIG-Bench-Hard.git',str(bbh_dir)],check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=bbh_dir,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=bbh_dir,text=True).strip():
        raise RuntimeError('BBH source has local modifications')
    revisions['bbh']={'repo':'https://github.com/suzgunmirac/BIG-Bench-Hard','revision':revision,
        'license':'MIT (repository; underlying BIG-bench task licenses also apply)',
        'split_note':'Custom disjoint 10 train / 5 dev / 20 test per task from official examples; not an official training split.'}
    partitions['bbh']={'train':[],'dev':[],'test':[]}
    for path in sorted((bbh_dir/'bbh').glob('*.json')):
        raw=json.loads(path.read_text())['examples']; records=[]
        for i,r in enumerate(raw):
            records.append({'id':f'bbh/{path.stem}/{i}','dataset':'bbh','question':r['input'],
              'answers':[str(r['target'])],'metric':'exact','stratum':path.stem,
              'source':{'dataset':revisions['bbh']['repo'],'revision':revision,'file':f'bbh/{path.name}',
                        'sha256':file_hash(path),'row':i,'split':'official_examples'}})
        ordered=rank(records,97)
        for split,slc in [('train',slice(0,10)),('dev',slice(10,15)),('test',slice(15,35))]:
            partitions['bbh'][split].extend(ordered[slc])
    # Resolve exact duplicates globally in train -> dev -> test order BEFORE freezing.
    # Exclusions are recorded; never replace a removed row after observing outcomes.
    seen={}; exclusions=[]; files={}; counts={}; hashes={}
    for split in ['train','dev','test']:
        for name in partitions:
            rows=[]
            for row in partitions[name][split]:
                h=digest(' '.join(row['question'].split()).casefold())
                if h in seen:
                    exclusions.append({'id':row['id'],'duplicate_of':seen[h]}); continue
                seen[h]=row['id']; row['split']=split; rows.append(row)
            partitions[name][split]=rows
            relative=f'{name}/{split}.jsonl';write_rows(out/relative,rows)
            files[relative]=file_hash(out/relative); counts[relative]=len(rows)
            hashes[relative]=digest([r['id'] for r in rows])
    models={}
    config=json.loads((ROOT/'configs/study.json').read_text())
    for repo in dict.fromkeys(config['models']+config.get('optional_models',[])+[config['embedding']['model'],'BAAI/bge-large-en-v1.5']):
        models[repo]=api.model_info(repo).sha
    write_json(out/'manifest.json',{'schema':1,'sample_seed':97,'sources':revisions,'files':files,
        'counts':counts,'ordered_id_hashes':hashes,'exact_duplicate_exclusions':exclusions,
        'models':models,'status':'data_prepared_no_inference'})
    print(json.dumps(counts,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data')
    p.add_argument('--bbh-dir',type=Path,default=ROOT/'upstream/bbh')
    a=p.parse_args();prepare(a.output,a.bbh_dir)
