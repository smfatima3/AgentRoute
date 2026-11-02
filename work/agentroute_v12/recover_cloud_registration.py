"""Recover only the campaign-tagged cloud coordinator after local registration failure."""
import json,time,hashlib,asyncio
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/agentroute_v12_completion'
def main():
    app=modal.App.lookup('agentroute-v12-completion',create_if_missing=False)
    rows=list(modal.Sandbox.list(app_id=app.app_id,tags={'campaign':'agentroute-v12-completion-20261031','role':'coordinator'}))
    if not rows:
        from modal.client import _Client
        from modal_proto import api_pb2
        async def history():
            client=await _Client.from_env()
            request=api_pb2.SandboxListRequest(app_id=app.app_id,include_finished=True,
              tags=[api_pb2.SandboxTag(tag_name=k,tag_value=v) for k,v in {'campaign':'agentroute-v12-completion-20261031','role':'coordinator'}.items()])
            token=await client._auth_token_manager.get_token()
            result=await asyncio.wait_for(client.stub.SandboxListV2(request,metadata=[('x-modal-auth-token',token)]),30)
            return [x.id for x in result.sandboxes]
        ids=asyncio.run(history());rows=[modal.Sandbox.from_id(id) for id in ids]
    print('Matching coordinators: '+str([(r.object_id,r.poll()) for r in rows]),flush=True)
    if len(rows)!=1:raise RuntimeError('Expected exactly one owned coordinator; no resource mutation performed')
    sb=rows[0];tags=sb.get_tags();stamp=tags['operation'];name='agentroute-v12-coordinator-'+stamp.lower()
    vol=modal.Volume.from_name(name);vol.hydrate()
    state={'sandbox_id':sb.object_id,'volume_name':name,'volume_id':vol.object_id,'app_id':app.app_id,'registration_recovered_unix_s':time.time(),
      'checkpoint_interval_s':1200,'status_interval_s':300,'cloud_resident':True,'no_codex_automation':True,'provider_timeout_s':43200,
      'seed_sha256':hashlib.sha256((ROOT/'work/agentroute_v12/campaign_seed.zip').read_bytes()).hexdigest()}
    (OUT/'CLOUD_COORDINATOR.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
    from inspect_cloud import main as inspect
    inspect()
if __name__=='__main__':main()
