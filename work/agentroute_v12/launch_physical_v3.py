"""One-shot launcher; waits for no workers, reserves all coordinator time, no GPU allocation."""
import hashlib,json,os,time,uuid
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parents[1]/'outputs/agentroute_v12_completion'
def main():
 current=json.loads((OUT/'FOCUS_RUN.json').read_text())
 if modal.Sandbox.from_id(current['id']).poll() is None:raise RuntimeError('Quality coordinator still active; transport must wait')
 if (OUT/'PHYSICAL_RUN.json').exists():
  old=json.loads((OUT/'PHYSICAL_RUN.json').read_text())
  if modal.Sandbox.from_id(old['id']).poll() is None:raise RuntimeError('Physical controller already active')
 volume=modal.Volume.from_name(current['volume_name']);prefix='outputs/agentroute_v12_completion/'
 status=json.loads(b''.join(volume.read_file(prefix+'STATUS.json')))
 counts=status['counts']
 if any(r['group_name'] in ['short_bbh','seed_sensitivity'] and r['state'] not in ['accepted','cancelled_by_user'] for r in counts):raise RuntimeError('Quality still incomplete')
 registry=json.loads(b''.join(volume.read_file(prefix+'resource_registry_live.json')));cost=0
 for r in registry['resources']:
  if r['state']!='terminated' and modal.Sandbox.from_id(r['id']).poll() is None:raise RuntimeError('Owned resource still running')
  end=r.get('stopped_unix_s',min(time.time(),r['created_unix_s']+r['timeout_s']))
  cost+=max(0,end-r['created_unix_s'])*r['rate_usd_per_hour']/3600
 # Reserve half a dollar for network traffic and bound three CPU allocations collectively.
 network_reserve=1.0
 previous=json.loads((OUT/'PHYSICAL_RUN.json').read_text()) if (OUT/'PHYSICAL_RUN.json').exists() else {}
 ceiling=previous.get('campaign_ceiling_usd',cost+10)
 timeout=min(57600,int((ceiling-cost-network_reserve-.2)/(.25*.141912+.024012+2*(.141912+.024012)*1.75)*3600))
 if timeout<900:raise RuntimeError('Insufficient safe credit for transport validation and experiment')
 seed={'nonce':uuid.uuid4().hex,'timeout_s':timeout,'prior_cost_upper_usd':cost,'additional_cap_usd':10,'campaign_ceiling_usd':ceiling,'network_reserve_usd':network_reserve,'authorization':'User: Up to $10 additional, 2026-11-02; cumulative ceiling preserved across retries','source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['physical_campaign_v3.py','physical_node.py']}}
 (OUT/'physical_identity_seed.json').write_text(json.dumps(seed,indent=2))
 image=modal.Image.debian_slim(python_version='3.12').pip_install('modal==1.6.2','filelock==3.32.3')
 for name in ['physical_campaign_v3.py','physical_node.py']:image=image.add_local_file(ROOT/name,'/opt/'+name.replace('_v3',''),copy=True)
 image=image.add_local_file(OUT/'analysis_inputs/RATES_FROZEN.json','/opt/rates_frozen.json',copy=True)
 image=image.add_local_file(OUT/'physical_identity_seed.json','/opt/transport_identity.json',copy=True)
 app=modal.App.lookup('agentroute-v12-completion',create_if_missing=True);start=time.time()
 sb=modal.Sandbox.create('python','/opt/physical_campaign.py',app=app,image=image,cpu=(.25,.25),memory=(1024,1024),timeout=timeout,volumes={'/workspace':volume},secrets=[modal.Secret.from_dict({'MODAL_TOKEN_ID':os.environ['MODAL_TOKEN_ID'],'MODAL_TOKEN_SECRET':os.environ['MODAL_TOKEN_SECRET']})],tags={'campaign':'agentroute-v12-completion-20261031','role':'physical-coordinator'})
 state={**seed,'id':sb.object_id,'created_unix_s':start,'volume_name':current['volume_name']}
 history=OUT/'physical_run_history';history.mkdir(exist_ok=True)
 if (OUT/'PHYSICAL_RUN.json').exists():(history/(str(int(time.time()))+'.json')).write_text((OUT/'PHYSICAL_RUN.json').read_text())
 (OUT/'PHYSICAL_RUN.json').write_text(json.dumps(state,indent=2))
 sb.filesystem.write_text(json.dumps(state),'/workspace/outputs/agentroute_v12_completion/TRANSPORT_ASSIGNED.json')
 print(json.dumps(state),flush=True)
if __name__=='__main__':main()
