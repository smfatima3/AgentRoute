"""Bounded CPU-only image/checkpoint validation and model-cache preparation."""
import hashlib,json,sys,time,threading
from pathlib import Path
import modal
BASE=Path(__file__).resolve().parents[2];ROOT=Path(__file__).resolve().parent;OUT=BASE/'outputs/agentroute_v12_completion'
sys.path.insert(0,str(BASE/'work/v9_local/clean_release_check_final/agentroute_modal_codex_v9'))
from modal_runner.images import images
from preflight_modal import save

def execute(sb,*args,timeout=900):
    p=sb.exec(*args,timeout=timeout);out=p.stdout.read();err=p.stderr.read();code=p.wait()
    if code:raise RuntimeError('Validation command failed: '+out[-3000:]+err[-3000:])
    return out

def main():
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());folder=OUT/'preparation'/stamp;folder.mkdir(parents=True)
    state={'status':'building_cpu_image','created_unix_s':time.time(),'gpu_allocated':False};save(folder/'state.json',state)
    save(OUT/'preparation_latest.json',{'path':str(folder)})
    sb=None;vol=None
    try:
        with modal.enable_output():
            ci,gi=images(ROOT/'package')
            ci=ci.run_commands('/opt/client/bin/python -m pip install -r /opt/package/requirements/evidence-math.lock','/opt/client/bin/python -m pip check')
            ci=ci.add_local_file(ROOT/'validate_fit.py','/opt/validate_fit.py',copy=True)
            ci=ci.add_local_file(BASE/'work/compact_recovery/restore_git_metadata.py','/opt/restore_git_metadata.py',copy=True)
            app=modal.App.lookup('agentroute-v12-completion',create_if_missing=True)
            vol=modal.Volume.from_name('agentroute-v12-prepare-'+stamp.lower(),create_if_missing=True,version=2)
            cache=modal.Volume.from_name('agentroute-v12-client-cache',create_if_missing=True,version=2)
            sb=modal.Sandbox.create('sleep','infinity',app=app,image=ci,cpu=2,memory=8192,timeout=1800,volumes={'/work':vol,'/hf':cache})
            state.update(status='validating_fits',id=sb.object_id,volume_id=vol.object_id,volume_name='agentroute-v12-prepare-'+stamp.lower(),created_sandbox_unix_s=time.time(),cpu_image_id=ci.object_id);save(folder/'state.json',state)
            execute(sb,'cp','-a','/opt/package','/work/project',timeout=120)
            execute(sb,'mkdir','-p','/work/project/recovery')
            execute(sb,'/opt/client/bin/python','/opt/restore_git_metadata.py',timeout=120)
            for method in ['safesieve','card','agentnet']:
                execute(sb,'/opt/client/bin/python','/opt/validate_fit.py','--method',method,timeout=900)
                execute(sb,'sync','/work',timeout=60);execute(sb,'sync','/hf',timeout=60)
                raw=b''.join(vol.read_file(f'project/validation/{method}.json'));(folder/(method+'.json')).write_bytes(raw)
                state.setdefault('verified_fits',[]).append({'method':method,'sha256':hashlib.sha256(raw).hexdigest(),'local_path':str(folder/(method+'.json'))});save(folder/'state.json',state)
                print('Checkpoint restored and durable export verified: '+method,flush=True)
            state['status']='CPU_PREPARATION_PASSED'
    except Exception as exc:
        state.update(status='FAILED',error_type=type(exc).__name__)
        print(str(exc)[:3500],flush=True)
    finally:
        if sb:
            try:
                sb.terminate(wait=True);state.update(provider_exit_code=sb.poll(),stopped_unix_s=time.time(),state='terminated')
            except Exception as exc:state['shutdown_error']=type(exc).__name__
        save(folder/'state.json',state)
        print(json.dumps(state),flush=True)
    if state['status']=='FAILED':raise RuntimeError('CPU preparation failed; evidence retained')
if __name__=='__main__':main()
