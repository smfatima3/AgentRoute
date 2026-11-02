import json,time
from pathlib import Path
import modal
OUT=Path(__file__).resolve().parents[2]/'outputs/agentroute_v12_completion'
def main():
 c=json.loads((OUT/'PHYSICAL_RUN.json').read_text());v=modal.Volume.from_name(c['volume_name']);prefix='outputs/agentroute_v12_completion/'
 def read(name):return json.loads(b''.join(v.read_file(prefix+name)))
 status=read('STATUS.json');reg=read('resource_registry_live.json');current=[r for r in reg['resources'] if r['created_unix_s']>=c['created_unix_s']-1]
 for r in current:
  if r['id'].startswith('sb-'):r['provider_exit_code_now']=modal.Sandbox.from_id(r['id']).poll()
 operations=sorted({r['operation'] for r in current if r.get('operation')})
 d={'observed_unix_s':time.time(),'coordinator_id':c['id'],'coordinator_exit_code':modal.Sandbox.from_id(c['id']).poll(),'counts':status['counts'],'cost_estimate_usd':status.get('resource_cost_estimate_usd'),'resources':current,'checkpoint':read('checkpoints/LATEST.json')}
 if operations:
  path='runs/'+operations[-1]+'/physical_transport/'
  for name in ['REPORT.json','EGRESS_RESERVATION.json','SHAPING_VALIDATION.json']:
   try:d[name]=read(path+name)
   except FileNotFoundError:pass
 (OUT/'PHYSICAL_VOLUME_STATUS.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:x for k,x in d.items() if k!='SHAPING_VALIDATION.json'}),flush=True)
if __name__=='__main__':main()
