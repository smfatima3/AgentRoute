"""Pin real MATH/BBH data. MATH's entire original test split stays held out."""
import collections, hashlib, json, re, sys, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentroute.core import file_hash, digest, write_json, write_rows

MATH_REV = '21a5633873b6a120296cce3e2df9d5550074f4a3'
BBH_REV = '9ee07bd481feebf959a6b59d61ea57bdcf30964d'
SUBJECTS = ['algebra','counting_and_probability','geometry','intermediate_algebra',
            'number_theory','prealgebra','precalculus']


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        req = urllib.request.Request(url, headers={'User-Agent':'AgentRoute-evidence-study'})
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_bytes(data); temporary.replace(path)
    return path


def boxed(solution):
    matches=list(re.finditer(r'\\(?:boxed|fbox)\s*',solution))
    if not matches:return None
    left=matches[-1].end()
    if left>=len(solution):return None
    # TeX without braces boxes its next single token (two source rows use this).
    if solution[left]!='{':
        match=re.match(r'(?:\\[A-Za-z]+|[^\s$])',solution[left:])
        return match.group() if match else None
    depth = 1; end = left + 1
    while end < len(solution) and depth:
        if solution[end] == '{': depth += 1
        elif solution[end] == '}': depth -= 1
        end += 1
    return solution[left+1:end-1] if depth == 0 else None


def prepare():
    import pyarrow.parquet as pq
    target = ROOT / 'evidence_data'; raw = ROOT / 'source_data'
    if (target/'manifest.json').exists():
        m = json.loads((target/'manifest.json').read_text())
        assert all(file_hash(target/k)==v for k,v in m['files'].items())
        print(json.dumps(m['counts'])); return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError('Incomplete preparation is preserved; inspect before retry')
    partitions = {d:{s:[] for s in ['train','dev','calibration','test']} for d in ['math','bbh']}
    raw_manifest = {}; unresolved = []
    for subject in SUBJECTS:
        for split in ['train','test']:
            name=f'{subject}/{split}-00000-of-00001.parquet'
            path=fetch(f'https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/{MATH_REV}/{name}',raw/'math'/name)
            raw_manifest['math/'+name]=file_hash(path)
            rows=[]
            for i,r in enumerate(pq.read_table(path).to_pylist()):
                answer=boxed(r['solution'])
                row={'id':f'math/{split}/{subject}/{i}','dataset':'math','question':r['problem'],
                     'answers':[answer] if answer is not None else [],'metric':'math',
                     'stratum':subject,'level':r['level'],'context':[],
                     'source':{'dataset':'EleutherAI/hendrycks_math','revision':MATH_REV,'split':split,'row':i,'file':name}}
                if answer is None:
                    unresolved.append({'id':row['id'],'reason':'no_balanced_box','split':split})
                    row['gold_solution_for_offline_scoring_only']=r['solution']
                rows.append(row)
            if split=='test': partitions['math']['test'].extend(rows)
            else:
                rows.sort(key=lambda r:digest([97,r['id']]))
                partitions['math']['dev'].extend(rows[:50])
                partitions['math']['calibration'].extend(rows[50:150])
                partitions['math']['train'].extend(rows[150:])
    tree_path=fetch(f'https://api.github.com/repos/suzgunmirac/BIG-Bench-Hard/git/trees/{BBH_REV}?recursive=1',raw/'bbh_tree.json')
    tree=json.loads(tree_path.read_text())
    for item in tree['tree']:
        name=item['path']
        if not name.startswith('bbh/') or not name.endswith('.json'): continue
        path=fetch(f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{BBH_REV}/{name}',raw/name)
        raw_manifest[name]=file_hash(path); task=path.stem
        rows=[{'id':f'bbh/{task}/{i}','dataset':'bbh','question':r['input'],
               'answers':[str(r['target'])],'metric':'exact','stratum':task,'context':[],
               'source':{'dataset':'suzgunmirac/BIG-Bench-Hard','revision':BBH_REV,'file':name,'row':i,'split':'official_examples'}}
              for i,r in enumerate(json.loads(path.read_text(encoding='utf-8'))['examples'])]
        rows.sort(key=lambda r:digest([97,r['id']]))
        for split,part in [('train',rows[:20]),('dev',rows[20:30]),('calibration',rows[30:40]),('test',rows[40:])]:
            partitions['bbh'][split].extend(part)
    assert len(partitions['math']['test'])==5000
    assert sum(len(partitions['math'][s]) for s in ['train','dev','calibration'])==7500
    # Test rows have priority: exclude overlapping training/development rows, never shrink MATH test.
    seen={}; duplicates=[]; files={}; counts={}; ids=set()
    for split in ['test','calibration','dev','train']:
        for dataset in ['math','bbh']:
            kept=[]
            for row in partitions[dataset][split]:
                h=digest(' '.join(row['question'].split()).casefold())
                if h in seen and seen[h]['split']!=split:
                    duplicates.append({'excluded':row['id'],'retained':seen[h]['id'],'reason':'cross_partition_exact_question_overlap'})
                    continue
                seen.setdefault(h,{'id':row['id'],'split':split})
                assert row['id'] not in ids; ids.add(row['id'])
                row['split']=split; kept.append(row)
            relative=f'{dataset}/{split}.jsonl'; write_rows(target/relative,kept)
            files[relative]=file_hash(target/relative); counts[relative]=len(kept)
    old=json.loads((ROOT/'data/manifest.json').read_text())
    write_json(target/'manifest.json',{'schema':2,'files':files,'counts':counts,'models':old['models'],
        'sample_seed':97,'sources':{'math':{'revision':MATH_REV,'source':'EleutherAI mirror of MATH with original train/test splits'},
        'bbh':{'revision':BBH_REV,'split_note':'Custom disjoint per-file split: first 20 fitting, next 10 development, next 10 calibration, rest test after hash ordering. Not an official train/test split.'}},
        'raw_files':raw_manifest,'cross_partition_exclusions':duplicates,'unresolved_gold':unresolved,
        'test_inference_performed':False,'protocol_frozen':False})
    print(json.dumps({'counts':counts,'unresolved_gold':len(unresolved),'cross_partition_exclusions':len(duplicates)},indent=2))


if __name__=='__main__': prepare()
