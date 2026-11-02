"""Manual finite status/export commands. No polling timer, experiment launch or monitoring loop."""
import json,time
from pathlib import Path
import modal
OUT=Path(__file__).resolve().parents[2]/'outputs/agentroute_v12_completion'
REMOTE=r'''
import pathlib,json,time,sqlite3
p=pathlib.Path('/workspace/outputs/agentroute_v12_completion');d={'observed_unix_s':time.time()}
reg=json.loads((p/'resource_registry_live.json').read_text());d['resources']=[r for r in reg['resources'] if r['state']!='terminated']
f=sorted((p/'runs').glob('*/physical_transport'))[-1];d['folder']=str(f)
for name in ['REPORT.json','HOSTS.json','SHAPING_VALIDATION.json','EGRESS_RESERVATION.json','CHECKPOINT_ERROR.json']:
 if (f/name).exists():d[name]=json.loads((f/name).read_text())
for name in ['CALIBRATION_IN_PROGRESS.json','RATES_FROZEN.json']:
 if (f/name).exists():
  rates=json.loads((f/name).read_text());d[name]={k:{a:b for a,b in v.items() if a!='ramp'} for k,v in rates.items()}
db=sqlite3.connect(p/'coordinator.sqlite');d['counts']=list(db.execute('SELECT group_name,state,count(*) FROM tasks GROUP BY group_name,state'));db.close()
if (p/'checkpoints/LATEST.json').exists():d['checkpoint']=json.loads((p/'checkpoints/LATEST.json').read_text())
print(json.dumps(d))
'''
def main():
 c=json.loads((OUT/'PHYSICAL_RUN.json').read_text());sb=modal.Sandbox.from_id(c['id']);v=modal.Volume.from_name(c['volume_name']);prefix='outputs/agentroute_v12_completion/'
 print('Manual console ready: status, quit',flush=True)
 while True:
  command=input().strip()
  if command=='quit':return
  if command!='status':continue
  code=sb.poll()
  if code is None:
   p=sb.exec('python','-c',REMOTE,timeout=45);raw=p.stdout.read();err=p.stderr.read()
   if p.wait():raise RuntimeError(err[-2000:])
   d=json.loads(raw)
  else:
   d={'observed_unix_s':time.time()}
   for n in ['PHYSICAL_TRANSPORT_LATEST.json','STATUS.json','checkpoints/LATEST.json']:
    try:d[n]=json.loads(b''.join(v.read_file(prefix+n)))
    except FileNotFoundError:pass
   latest=d.get('PHYSICAL_TRANSPORT_LATEST.json',{}).get('path')
   if latest:
    path=latest[latest.index('outputs/'):]
    for n in ['SHAPING_VALIDATION.json','HOSTS.json','REPORT.json']:
     try:d[n]=json.loads(b''.join(v.read_file(path+'/'+n)))
     except FileNotFoundError:pass
   status=d.pop('STATUS.json',{})
   d['counts']=status.get('counts');d['resources']=[r for r in status.get('resources',[]) if r['state']!='terminated']
  for r in d.get('resources',[]):r['provider_exit_code_now']=modal.Sandbox.from_id(r['id']).poll()
  d.update(coordinator_id=c['id'],coordinator_exit_code=code)
  (OUT/'PHYSICAL_STATUS_LATEST.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
if __name__=='__main__':main()
