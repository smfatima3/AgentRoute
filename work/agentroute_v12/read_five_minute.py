import json,time
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parents[1]/'outputs/agentroute_v12_completion'
def main():
 c=json.loads((OUT/'FIVE_MINUTE_RUN.json').read_text());sb=modal.Sandbox.from_id(c['id']);v=modal.Volume.from_name(c['volume_name']);out={'observed_unix_s':time.time(),'coordinator_exit_code':sb.poll()}
 prefix='outputs/agentroute_v12_completion/'
 for e in v.iterdir(prefix+'short_windows',recursive=True):
  if e.type==modal.volume.FileEntryType.FILE and e.path.endswith(('REPORT.json','ERROR.json','PLAN.json')):
   out[e.path]=json.loads(b''.join(v.read_file(e.path)))
 for name in ['STATUS.json','resource_registry_live.json','ACTIVE_SCOPE.json']:
  try:out[name]=json.loads(b''.join(v.read_file(prefix+name)))
  except FileNotFoundError:pass
 out['active_resources']=[]
 for r in out.get('resource_registry_live.json',{}).get('resources',[]):
  if r['state']!='terminated':out['active_resources'].append({'id':r['id'],'role':r['role'],'exit_code':modal.Sandbox.from_id(r['id']).poll()})
 (OUT/'FIVE_MINUTE_OBSERVATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
