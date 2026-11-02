"""Finite, operator-driven cloud inspection. No timers, launch, checkpoint or monitoring loop."""
import concurrent.futures,json,time
from pathlib import Path
import modal
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/agentroute_v12_completion'
REMOTE="""import pathlib,json,time
p=pathlib.Path('/workspace/outputs/agentroute_v12_completion');x={'observed_unix_s':time.time()}
for n in ['CLOUD_STATUS.json','STATUS.json','resource_registry_live.json','checkpoints/LATEST.json','RUNTIME_LINEAGE.json']:
 f=p/n;x[n]=json.loads(f.read_text()) if f.exists() else None
f=p/'cloud_controller.log';x['log_tail']=f.read_text()[-5000:] if f.exists() else ''
print(json.dumps(x))
"""
def main():
    c=json.loads((OUT/'CLOUD_COORDINATOR.json').read_text());sb=modal.Sandbox.from_id(c['sandbox_id']);vol=modal.Volume.from_name(c['volume_name'])
    print('Ready for a one-shot command: status, logs, quit',flush=True)
    while True:
        command=input().strip()
        if command=='quit':return
        if command not in ['status','logs']:print('Unknown command',flush=True);continue
        code=sb.poll()
        if code is None:
            proc=sb.exec('python','-c',REMOTE,timeout=30);raw=proc.stdout.read();err=proc.stderr.read()
            if proc.wait():raise RuntimeError(err[-2000:])
            data=json.loads(raw)
        else:
            data={'observed_unix_s':time.time()}
            for name in ['CLOUD_STATUS.json','STATUS.json','resource_registry_live.json','checkpoints/LATEST.json']:
                try:data[name]=json.loads(b''.join(vol.read_file('outputs/agentroute_v12_completion/'+name)))
                except FileNotFoundError:data[name]=None
            try:data['log_tail']=b''.join(vol.read_file('outputs/agentroute_v12_completion/cloud_controller.log')).decode()[-5000:]
            except FileNotFoundError:pass
        data.update(coordinator_id=sb.object_id,provider_exit_code=code)
        resources=(data.get('resource_registry_live.json') or {}).get('resources',[])
        active=[r for r in resources if r['state']!='terminated']
        def check(r):
            remote=modal.Sandbox.from_id(r['id']);exit_code=remote.poll();result={'id':r['id'],'role':r['role'],'exit_code':exit_code}
            if command=='logs' and r['role'].startswith('GPU'):
                try:
                    v=modal.Volume.from_id(r['volume_id']);result['server_log_tail']=b''.join(v.read_file(r['role']+'/server.log')).decode()[-4500:]
                except FileNotFoundError:result['server_log_tail']=None
                if exit_code is not None:result['stderr_tail']=remote.stderr.read()[-2500:]
            return result
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:data['provider_active_resource_states']=list(pool.map(check,active))
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());folder=OUT/'cloud_observations'/stamp;folder.mkdir(parents=True,exist_ok=True)
        (folder/'observation.json').write_text(json.dumps(data,indent=2))
        summary={'provider_exit_code':code,'cloud_state':(data.get('CLOUD_STATUS.json') or {}).get('state'),'checkpoint':data.get('checkpoints/LATEST.json'),
          'resources':data['provider_active_resource_states'],'counts':(data.get('STATUS.json') or {}).get('counts'),
          'resource_cost_estimate_usd':(data.get('STATUS.json') or {}).get('resource_cost_estimate_usd'),'log_tail':data.get('log_tail'),'observation':str(folder/'observation.json')}
        print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
