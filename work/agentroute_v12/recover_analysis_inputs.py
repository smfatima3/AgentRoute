"""Read-only recovery of calibration and hash-verified accepted quality records."""
import hashlib,json,time
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parents[1]/'outputs/agentroute_v12_completion'
def main():
 c=json.loads((OUT/'PHYSICAL_RUN.json').read_text());sb=modal.Sandbox.from_id(c['id'])
 # Quality receipts are closed and immutable; read-only download may overlap transport.
 v=modal.Volume.from_name(c['volume_name']);prefix='outputs/agentroute_v12_completion/'
 latest=json.loads(b''.join(v.read_file(prefix+'PHYSICAL_TRANSPORT_LATEST.json')))
 folder=latest['path'][latest['path'].index('outputs/'):]
 dest=OUT/'analysis_inputs';dest.mkdir(exist_ok=True)
 for name in ['RATES_FROZEN.json','SOURCE_FREEZE.json','REPORT.json','SHAPING_VALIDATION.json']:
  raw=b''.join(v.read_file(folder+'/'+name));(dest/name).write_bytes(raw)
 rates=json.loads((dest/'RATES_FROZEN.json').read_text());print(json.dumps({'rates':{k:{n:x for n,x in r.items() if n!='ramp'} for k,r in rates.items()},'report':latest['report']}),flush=True)
 for name in ['campaign.lock.json','results_index.jsonl','protocol.json','resource_registry_live.json','ACTIVE_SCOPE.json']:
  (dest/name).write_bytes(b''.join(v.read_file(prefix+name)))
 rows=[json.loads(s) for s in (dest/'results_index.jsonl').read_text().splitlines()]
 manifest=json.loads((dest/'campaign.lock.json').read_text());specs={t['task_id']:t for t in manifest['tasks']}
 selected=[r for r in rows if specs[r['task_id']].get('dataset') in ['mmlu','bbh'] and specs[r['task_id']].get('split')=='test']
 paths={e['path']:e['sha256'] for r in selected for e in r['evidence'] if e['role']=='raw'}
 import concurrent.futures
 def fetch(item):
  path,sha=item;p=(dest/'raw'/(sha+Path(path).suffix)).resolve()
  if not p.is_relative_to(dest.resolve()):raise RuntimeError('Unsafe evidence path')
  if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==sha:return
  raw=b''.join(v.read_file(prefix+path))
  if hashlib.sha256(raw).hexdigest()!=sha:raise RuntimeError('Evidence hash mismatch '+path)
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,paths.items()))
 (dest/'DOWNLOAD_VERIFICATION.json').write_text(json.dumps({'verified_unix_s':time.time(),'raw_files':len(paths),'accepted_selected_tasks':len(selected),'hashes':paths},indent=2))
 print(json.dumps({'verified_raw_files':len(paths),'selected_task_count':len(selected)}),flush=True)
if __name__=='__main__':main()
