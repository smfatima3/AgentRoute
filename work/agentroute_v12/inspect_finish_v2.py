import json,time
from pathlib import Path
import modal
OUT=Path(__file__).resolve().parents[2]/'outputs/agentroute_v12_completion'
REMOTE=r'''
import json,pathlib,sqlite3,time
p=pathlib.Path('/workspace/outputs/agentroute_v12_completion');d={'observed_unix_s':time.time()}
db=sqlite3.connect(p/'coordinator.sqlite')
d['counts']=list(db.execute('SELECT group_name,state,count(*) FROM tasks GROUP BY group_name,state'))
d['pending']=list(db.execute("SELECT group_name,json_extract(spec,'$.method'),state,count(*) FROM tasks WHERE group_name IN ('short_bbh','seed_sensitivity') GROUP BY group_name,json_extract(spec,'$.method'),state"))
d['resources']=[r for r in json.loads((p/'resource_registry_live.json').read_text())['resources'] if r['state']!='terminated']
f=sorted((p/'short_windows').iterdir())[-1];d['window']=str(f)
for name in ['ERROR.json','REPORT.json','QUALITY_REMAINDER.json','TRANSPORT_DEPENDENCY.json','transport_preflight/REPORT.json']:
 if (f/name).exists():d[name]=json.loads((f/name).read_text())
if (p/'checkpoints/LATEST.json').exists():d['checkpoint']=json.loads((p/'checkpoints/LATEST.json').read_text())
print(json.dumps(d))
'''
def main():
 c=json.loads((OUT/'FOCUS_RUN.json').read_text());sb=modal.Sandbox.from_id(c['id']);code=sb.poll()
 if code is None:
  p=sb.exec('python','-c',REMOTE,timeout=30);raw=p.stdout.read();err=p.stderr.read()
  if p.wait():raise RuntimeError(err[-1000:])
  d=json.loads(raw)
 else:
  v=modal.Volume.from_name(c['volume_name']);prefix='outputs/agentroute_v12_completion/'
  def read(n):return json.loads(b''.join(v.read_file(prefix+n)))
  d=read('STATUS.json');d['resources']=[r for r in d['resources'] if r['state']!='terminated']
  paths=[e.path for e in v.iterdir(prefix+'short_windows',recursive=True) if e.path.endswith('/REPORT.json')]
  latest=sorted(x for x in paths if '/transport_preflight/' not in x)[-1];d['report']=json.loads(b''.join(v.read_file(latest)))
  for name in ['transport_preflight/REPORT.json','QUALITY_REMAINDER.json','TRANSPORT_DEPENDENCY.json']:
   try:d[name]=json.loads(b''.join(v.read_file(latest.rsplit('/',1)[0]+'/'+name)))
   except FileNotFoundError:pass
  d['checkpoint']=read('checkpoints/LATEST.json')
 d['coordinator_id']=c['id'];d['coordinator_exit_code']=code
 for r in d['resources']:r['provider_exit_code_now']=modal.Sandbox.from_id(r['id']).poll()
 (OUT/'FINISH_V2_OBSERVATION.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
if __name__=='__main__':main()
