"""One-shot recovery of stopped cloud evidence, including GPU stderr. No launches."""
import hashlib,json,time
from pathlib import Path
import modal
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/agentroute_v12_completion'
def main():
    c=json.loads((OUT/'CLOUD_COORDINATOR.json').read_text());sb=modal.Sandbox.from_id(c['sandbox_id'])
    if sb.poll() is None:raise RuntimeError('Coordinator running; stopped-state export refused')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());dest=OUT/'cloud_recovery'/stamp;dest.mkdir(parents=True)
    vol=modal.Volume.from_name(c['volume_name']);root='outputs/agentroute_v12_completion'
    for entry in vol.iterdir(root,recursive=True):
        if entry.type!=modal.volume.FileEntryType.FILE:continue
        rel=entry.path.lstrip('/');sub=rel[len(root)+1:]
        if sub.startswith(('reuse/','protocol_history/')):continue
        p=(dest/sub).resolve()
        if not p.is_relative_to(dest.resolve()):raise RuntimeError('Export path traversal')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b''.join(vol.read_file(rel)))
    registry=json.loads((dest/'resource_registry_live.json').read_text())
    for r in registry['resources']:
        remote=modal.Sandbox.from_id(r['id']);code=remote.poll()
        print(json.dumps({'id':r['id'],'role':r['role'],'exit_code':code}),flush=True)
        if code is None:raise RuntimeError('Owned resource still running; do not relaunch')
        if r['state']!='terminated':r.update(state='terminated',provider_exit_code=code,stopped_unix_s=time.time())
        if r['role'].startswith('GPU'):
            for kind in ['stdout','stderr']:
                raw=getattr(remote,kind).read();(dest/(r['id']+'-'+kind+'.log')).write_text(raw)
                print(r['role']+' '+kind+' '+raw[-4000:],flush=True)
    (dest/'provider_registry.json').write_text(json.dumps(registry,indent=2))
    (OUT/'CLOUD_RECOVERY_LATEST.json').write_text(json.dumps({'path':str(dest),'all_resources_stopped':True},indent=2))
    print('Saved stopped-state evidence: '+str(dest),flush=True)
if __name__=='__main__':main()
