"""Launch once, verify cloud bootstrap, then exit. Secrets stay in process/Modal memory."""
import hashlib,json,os,time,sys
from pathlib import Path
import modal
BASE=Path(__file__).resolve().parents[2];ROOT=Path(__file__).resolve().parent;OUT=BASE/'outputs/agentroute_v12_completion'
def save(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def main():
    resume='--resume' in sys.argv
    old=None
    if (OUT/'CLOUD_COORDINATOR.json').exists():
        old=json.loads((OUT/'CLOUD_COORDINATOR.json').read_text());sb=modal.Sandbox.from_id(old['sandbox_id'])
        if sb.poll() is None:raise RuntimeError('Existing cloud coordinator still running; do not duplicate')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());volume_name=old['volume_name'] if resume and old else 'agentroute-v12-coordinator-'+stamp.lower()
    if resume and not old:raise RuntimeError('No registered campaign to resume')
    volume=modal.Volume.from_name(volume_name,create_if_missing=True,version=2)
    volume.hydrate()
    app=modal.App.lookup('agentroute-v12-completion',create_if_missing=True)
    image=modal.Image.debian_slim(python_version='3.12').pip_install('modal==1.6.2','filelock==3.32.3')
    image=image.add_local_file(ROOT/'cloud_boot.py','/opt/cloud_boot.py',copy=True)
    if (OUT/'PENDING_SCOPE_CHANGE.json').exists():
        image=image.add_local_file(OUT/'PENDING_SCOPE_CHANGE.json','/opt/scope_change.json',copy=True)
    if resume:
        recovery=json.loads((OUT/'CLOUD_RECOVERY_LATEST.json').read_text())
        if not recovery['all_resources_stopped']:raise RuntimeError('All previous resources must be provider-confirmed stopped')
        folder=OUT/'deployments'/stamp;folder.mkdir(parents=True)
        patch={'reason':'Supply the pinned compact model manifest at the unchanged serving-preflight compatibility path; preserve all attempts, scientific policies, task IDs and budget history.',
          'files':{name:{'text':(ROOT/name).read_text(),'sha256':hashlib.sha256((ROOT/name).read_text().encode()).hexdigest()} for name in ['run_campaign.py','shard.py','gpu_entry.py','serve_v12.py']}}
        save(folder/'runtime_patch.json',patch)
        image=image.add_local_file(folder/'runtime_patch.json','/opt/runtime_patch.json',copy=True).add_local_file(Path(recovery['path'])/'provider_registry.json','/opt/provider_registry.json',copy=True)
        save(folder/'previous_registration.json',old)
    else:image=image.add_local_file(ROOT/'campaign_seed.zip','/opt/campaign_seed.zip',copy=True)
    start=time.time()
    sb=modal.Sandbox.create('python','/opt/cloud_boot.py',app=app,image=image,cpu=(.5,1),memory=(2048,2048),timeout=43200,
      volumes={'/workspace':volume},secrets=[modal.Secret.from_dict({'MODAL_TOKEN_ID':os.environ['MODAL_TOKEN_ID'],'MODAL_TOKEN_SECRET':os.environ['MODAL_TOKEN_SECRET'],
      'V12_COORDINATOR_STARTED':str(start),'V12_COORDINATOR_VOLUME':volume.object_id})],tags={'campaign':'agentroute-v12-completion-20261031','role':'coordinator','operation':stamp})
    state={'sandbox_id':sb.object_id,'volume_name':volume_name,'volume_id':volume.object_id,'app_id':app.app_id,'launched_unix_s':start,
      'checkpoint_interval_s':1200,'status_interval_s':300,'cloud_resident':True,'no_codex_automation':True,'provider_timeout_s':43200,
      'seed_sha256':hashlib.sha256((ROOT/'campaign_seed.zip').read_bytes()).hexdigest()}
    save(OUT/'CLOUD_COORDINATOR.json',state)
    print('Coordinator registered: '+sb.object_id,flush=True)
    # Register locally before further RPCs; use explicit identity rather than undocumented environment variables.
    sb.filesystem.write_text(json.dumps(state),'/workspace/COORDINATOR_ASSIGNED.json')
    for _ in range(45):
        try:
            raw=b''.join(volume.read_file('outputs/agentroute_v12_completion/CLOUD_STATUS.json'));remote=json.loads(raw)
            if remote.get('coordinator_id')==sb.object_id and remote.get('state')=='RUNNING':
                state['verified_bootstrap']=remote;save(OUT/'CLOUD_COORDINATOR.json',state)
                print(json.dumps(state),flush=True);return
        except FileNotFoundError:pass
        if sb.poll() is not None:
            print(sb.stdout.read()[-3000:]+sb.stderr.read()[-3000:],flush=True);raise RuntimeError('Cloud coordinator failed before durable bootstrap')
        time.sleep(2)
    raise RuntimeError('Cloud coordinator bootstrap not yet visible; exact ID retained, do not duplicate')
if __name__=='__main__':main()
