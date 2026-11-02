import json,os,time,sys
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parents[1]/'outputs/agentroute_v12_completion'
def main():
    if (OUT/'FOCUS_RUN.json').exists():
        previous=json.loads((OUT/'FOCUS_RUN.json').read_text())
        if modal.Sandbox.from_id(previous['id']).poll() is None:raise RuntimeError('Focus coordinator already running')
    from_ready=False
    deadline=time.time()+5400
    if time.time()>deadline-60:raise RuntimeError('Five-minute reporting window too short to launch; no GPU allocated')
    c=json.loads((OUT/'CLOUD_COORDINATOR.json').read_text())
    if modal.Sandbox.from_id(c['sandbox_id']).poll() is None:raise RuntimeError('Existing coordinator running')
    prior=[]
    for id,start,timeout in [('sb-01M6AXXR1GM1YBWF6JQNQMXW3Z',1793515774.0501997,600),('sb-01M6AYK6M1RJD5YS7EJWCM7GXB',1793516477.18816,1200)]:
        code=modal.Sandbox.from_id(id).poll()
        if code is None:raise RuntimeError('Previous short coordinator is still active; do not duplicate')
        prior.append({'id':id,'role':'coordinator','owned':True,'state':'terminated','provider_exit_code':code,'created_unix_s':start,'stopped_unix_s':min(time.time(),start+timeout),'timeout_s':timeout,'rate_usd_per_hour':.1899,'cost_mode':'conservative provider-timeout upper bound'})
    # Reserve this coordinator's full timeout before allocating any GPUs.
    prior.append({'id':'focus-coordinator-reservation-'+str(int(time.time())),'role':'coordinator','owned':True,'state':'running','created_unix_s':time.time(),'timeout_s':6000,'rate_usd_per_hour':.1899,'cost_mode':'reserved maximum coordinator lifetime; actual ID in FOCUS_RUN.json'})
    (OUT/'focus_prior_coordinators.json').write_text(json.dumps(prior,indent=2))
    v=modal.Volume.from_name(c['volume_name']);app=modal.App.lookup('agentroute-v12-completion',create_if_missing=True)
    image=modal.Image.debian_slim(python_version='3.12').pip_install('modal==1.6.2','filelock==3.32.3').add_local_file(ROOT/'focus_cloud.py','/opt/five_minute_cloud.py',copy=True).add_local_file(OUT/'focus_prior_coordinators.json','/opt/prior_coordinators.json',copy=True)
    sb=modal.Sandbox.create('python','/opt/five_minute_cloud.py',app=app,image=image,cpu=(.5,1),memory=(2048,2048),timeout=6000,volumes={'/workspace':v},secrets=[modal.Secret.from_dict({'MODAL_TOKEN_ID':os.environ['MODAL_TOKEN_ID'],'MODAL_TOKEN_SECRET':os.environ['MODAL_TOKEN_SECRET'],'SHORT_DEADLINE':str(deadline),'SHORT_FROM_READY':'0'})],tags={'campaign':'agentroute-v12-completion-20261031','role':'bbh-ablation-coordinator'})
    state={'id':sb.object_id,'volume_name':c['volume_name'],'deadline_unix_s':deadline,'created_unix_s':time.time(),'timeout_s':6000,'role':'bbh-ablation-coordinator','rate_usd_per_hour':.1899,'excluded_datasets':['math','svamp']}
    (OUT/'FOCUS_RUN.json').write_text(json.dumps(state,indent=2));(OUT/'FIVE_MINUTE_RUN.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
    sb.filesystem.write_text(json.dumps(state),'/workspace/FOCUS_ASSIGNED.json')
if __name__=='__main__':main()
