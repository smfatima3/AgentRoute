import json,time
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parents[1]/'outputs/agentroute_v12_completion'
REMOTE="""import pathlib,json,sqlite3,time
p=pathlib.Path('/workspace/outputs/agentroute_v12_completion');d={'observed_unix_s':time.time()}
r=json.loads((p/'resource_registry_live.json').read_text());d['active_resources']=[x for x in r['resources'] if x['state']!='terminated']
db=sqlite3.connect('file:'+str(p/'coordinator.sqlite')+'?mode=ro',uri=True);d['counts']=[list(x) for x in db.execute('select group_name,state,count(*) from tasks group by group_name,state')];db.close()
f=sorted((p/'short_windows').glob('*'))[-1]
for n in ['MEASUREMENT_WINDOW.json','WORKERS_DISPATCHED.json','ERROR.json','REPORT.json']:
 if (f/n).exists():d[n]=json.loads((f/n).read_text())
print(json.dumps(d))
"""
def main():
 c=json.loads((OUT/'FIVE_MINUTE_RUN.json').read_text());sb=modal.Sandbox.from_id(c['id']);code=sb.poll()
 if code is not None:
  from read_five_minute import main as read
  read();return
 proc=sb.exec('python','-c',REMOTE,timeout=30);raw=proc.stdout.read();err=proc.stderr.read()
 if proc.wait():raise RuntimeError(err[-1500:])
 data=json.loads(raw);data['coordinator_id']=c['id'];data['coordinator_exit_code']=code
 for r in data['active_resources']:r['observed_provider_exit_code']=modal.Sandbox.from_id(r['id']).poll()
 (OUT/'SHORT_LIVE_STATUS.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
if __name__=='__main__':main()
