"""One explicit read-only inspection/download; never an automation or watcher."""
import hashlib,json,time
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/agentroute_v12_completion'
def main():
    c=json.loads((OUT/'CLOUD_COORDINATOR.json').read_text());volume=modal.Volume.from_name(c['volume_name']);sb=modal.Sandbox.from_id(c['sandbox_id'])
    data={'observed_unix_s':time.time(),'coordinator_id':sb.object_id,'provider_exit_code':sb.poll()};stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());dest=OUT/'cloud_observations'/stamp;dest.mkdir(parents=True)
    for name in ['CLOUD_STATUS.json','STATUS.json','resource_registry_live.json','checkpoints/LATEST.json']:
        try:
            raw=b''.join(volume.read_file('outputs/agentroute_v12_completion/'+name));value=json.loads(raw);data[name]=value
            p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        except FileNotFoundError:data[name]=None
    try:
        raw=b''.join(volume.read_file('outputs/agentroute_v12_completion/cloud_controller.log'));(dest/'cloud_controller.log').write_bytes(raw);data['log_tail']=raw.decode(errors='replace')[-3500:]
    except FileNotFoundError:pass
    if data['provider_exit_code'] is not None:
        data['coordinator_stderr_tail']=sb.stderr.read()[-3500:]
    resources=data.get('resource_registry_live.json') or {}
    data['provider_resource_states']=[]
    for r in resources.get('resources',[]):
        s=modal.Sandbox.from_id(r['id']);data['provider_resource_states'].append({'id':r['id'],'role':r['role'],'exit_code':s.poll()})
    (dest/'observation.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
if __name__=='__main__':main()
