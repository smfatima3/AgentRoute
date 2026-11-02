import json,os,time,sys
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parents[1]/'outputs/agentroute_v12_completion'
def main():
    from_ready='--from-ready' in sys.argv
    deadline=time.time()+900 if from_ready else 1793515872.0
    if time.time()>deadline-60:raise RuntimeError('Five-minute reporting window too short to launch; no GPU allocated')
    c=json.loads((OUT/'CLOUD_COORDINATOR.json').read_text())
    if modal.Sandbox.from_id(c['sandbox_id']).poll() is None:raise RuntimeError('Existing coordinator running')
    v=modal.Volume.from_name(c['volume_name']);app=modal.App.lookup('agentroute-v12-completion',create_if_missing=True)
    image=modal.Image.debian_slim(python_version='3.12').pip_install('modal==1.6.2','filelock==3.32.3').add_local_file(ROOT/'five_minute_cloud.py','/opt/five_minute_cloud.py',copy=True)
    sb=modal.Sandbox.create('python','/opt/five_minute_cloud.py',app=app,image=image,cpu=(.5,1),memory=(2048,2048),timeout=1200,volumes={'/workspace':v},secrets=[modal.Secret.from_dict({'MODAL_TOKEN_ID':os.environ['MODAL_TOKEN_ID'],'MODAL_TOKEN_SECRET':os.environ['MODAL_TOKEN_SECRET'],'SHORT_DEADLINE':str(deadline),'SHORT_FROM_READY':'1' if from_ready else '0'})],tags={'campaign':'agentroute-v12-completion-20261031','role':'five-minute-coordinator'})
    state={'id':sb.object_id,'volume_name':c['volume_name'],'deadline_unix_s':deadline,'created_unix_s':time.time(),'timeout_s':1200,'role':'five-minute-coordinator','rate_usd_per_hour':.1899,'measurement_from_model_ready':from_ready}
    (OUT/'FIVE_MINUTE_RUN.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
if __name__=='__main__':main()
