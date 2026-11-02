"""Bounded CPU-only capability and persistence probe; no benchmark/model calls."""
import argparse,concurrent.futures,hashlib,json,secrets,threading,time
from datetime import datetime,timezone
from pathlib import Path
import modal
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/agentroute_v12_completion';LOCK=threading.Lock()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)
def command(sb,*args):
    p=sb.exec(*args,timeout=30);out=p.stdout.read();err=p.stderr.read();code=p.wait()
    return {'returncode':code,'stdout':out,'stderr':err}
PROBE='''import json,os,pathlib,socket,subprocess
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=10);return dict(argv=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def read(path):
 try:return pathlib.Path(path).read_text().strip()
 except OSError:return None
data=dict(hostname=socket.gethostname(),boot_id=read('/proc/sys/kernel/random/boot_id'),region=os.getenv('MODAL_REGION'),
 location=os.getenv('MODAL_CLOUD_PROVIDER'),network_namespace=os.readlink('/proc/self/ns/net'),
 status=[s for s in read('/proc/self/status').splitlines() if s.startswith(('CapEff','CapBnd'))],
 qdisc_before=run(['tc','-s','qdisc','show','dev','eth0']),
 shaping=run(['tc','qdisc','replace','dev','eth0','root','netem','delay','25ms','rate','10mbit']),
 qdisc_after=run(['tc','-s','qdisc','show','dev','eth0']))
if data['shaping']['returncode']==0:data['cleanup']=run(['tc','qdisc','del','dev','eth0','root'])
print(json.dumps(data))
'''
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',choices=['gvisor','vm'],default='gvisor');args=parser.parse_args()
    OUT.mkdir(exist_ok=True); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');folder=OUT/'preflight'/stamp;folder.mkdir(parents=True)
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'mode':'CPU capability/persistence probe','runtime':args.runtime,'gpu_allocated':False,'resources':[],'result':'RUNNING','modal_sdk':modal.__version__}
    save(folder/'state.json',state);save(OUT/'preflight_latest.json',{'path':str(folder)})
    app=modal.App.lookup('agentroute-v12-completion',create_if_missing=True)
    image=modal.Image.debian_slim(python_version='3.12').apt_install('iproute2')
    hosts=[]
    def make(i):
        vol=modal.Volume.from_name('agentroute-v12-preflight-'+stamp.lower()+'-'+str(i),create_if_missing=True,version=2)
        sb=modal.Sandbox.create('sleep','infinity',app=app,image=image,cpu=1 if args.runtime=='vm' else .25,memory=1024 if args.runtime=='vm' else 512,runtime=args.runtime,timeout=300,region=['us-east','us-west'][i],volumes={'/evidence':vol})
        resource={'id':sb.object_id,'role':'CPU-NET-'+str(i),'requested_region':['us-east','us-west'][i],'volume_id':vol.object_id,'volume_version_requested':2,'created_unix_s':time.time(),'state':'running'}
        with LOCK:state['resources'].append(resource);hosts.append((sb,vol,resource));save(folder/'state.json',state)
        probe=command(sb,'python','-c',PROBE)
        if probe['returncode']!=0:raise RuntimeError('Capability probe failed')
        resource['capabilities']=json.loads(probe['stdout']);marker=secrets.token_hex(24)
        sb.filesystem.write_text(marker,'/evidence/persistence.txt');sync=command(sb,'sync','/evidence')
        if sync['returncode']!=0:raise RuntimeError('Explicit v2 commit failed')
        visible=b''.join(vol.read_file('persistence.txt')).decode()
        if visible!=marker:raise RuntimeError('Committed data not visible through provider')
        resource['commit_read_verified']=True;resource['marker_sha256']=hashlib.sha256(marker.encode()).hexdigest()
        save(folder/('host'+str(i)+'.json'),resource)
        return marker
    try:
        with modal.enable_output():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(make,i) for i in range(2)]
                for f in futures:f.result()
        state['result']='PERSISTENCE_PASSED'
    except Exception as exc:state.update(result='FAILED',error_type=type(exc).__name__)
    finally:
        for sb,vol,r in hosts:
            try:
                sb.terminate(wait=True);r['exit_code']=sb.poll();r['state']='terminated' if r['exit_code'] is not None else 'unknown';r['stopped_unix_s']=time.time()
                raw=b''.join(vol.read_file('persistence.txt'));r['after_shutdown_persistence_verified']=hashlib.sha256(raw).hexdigest()==r.get('marker_sha256')
            except Exception as exc:r['cleanup_error_type']=type(exc).__name__
        state['physical_shaping_permitted']=len(hosts)==2 and all(r.get('capabilities',{}).get('shaping',{}).get('returncode')==0 for _,_,r in hosts)
        state['physical_network_experiment_ready']=False
        state['network_note']='Permission alone does not validate achieved bandwidth/one-way delay or physical host identity; those remain required before physical transport.'
        save(folder/'state.json',state);save(OUT/'MODAL_PREFLIGHT.json',state)
        print(json.dumps(state),flush=True)
    if state['result']=='FAILED':raise RuntimeError('Modal CPU preflight failed; see persisted evidence')
if __name__=='__main__':main()
