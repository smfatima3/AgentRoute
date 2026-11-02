"""Pinned V1 server with per-request observations and independent durable logging."""
import argparse,json,os,re,subprocess,sys,threading,time,urllib.request,socket
from pathlib import Path
sys.path.insert(0,'/opt/control')
from modal_runner.gpu_server import command

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-json',required=True);a=p.parse_args();plan=json.loads(a.plan_json)
    root=Path('/gpu-logs')/plan['operation_id'];root.mkdir(parents=True,exist_ok=True)
    os.environ['V12_GPU_LOG_DIR']=str(root)
    meta={'hostname':socket.gethostname(),'region':os.getenv('MODAL_REGION'),'started_unix_s':time.time(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    r=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True);meta['gpu_identity']=r.stdout
    (root/'identity.json').write_text(json.dumps(meta,indent=2))
    subprocess.run(['/opt/gpu/bin/python','/opt/package/scripts/check_serving_env.py','--model',plan['model'],'--require-cuda','--tensor-parallel-size','1','--report',str(root/'serving_preflight.json')],check=True)
    stop=threading.Event();activity={'count':None,'changed':time.monotonic(),'ready':False}
    def sample():
        last=0.
        while not stop.is_set():
            row={'unix_s':time.time(),'monotonic_s':time.monotonic()}
            try:
                r=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5);row['nvidia_smi_csv']=r.stdout.strip()
                req=urllib.request.Request('http://127.0.0.1:8000/metrics',headers={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']})
                with urllib.request.urlopen(req,timeout=2) as response:row['vllm_prometheus']=response.read().decode()
                count=sum(float(v) for v in re.findall(r'^vllm:request_success_total\{[^\n]*\}\s+([\d.e+]+)',row['vllm_prometheus'],re.M))
                if count!=activity['count']:activity.update(count=count,changed=time.monotonic())
                activity['ready']=True
            except Exception as exc:row['sample_error']=type(exc).__name__
            with (root/'resource_samples.jsonl').open('a',buffering=1) as f:f.write(json.dumps(row)+'\n')
            if time.monotonic()-last>=60:subprocess.run(['sync','/gpu-logs'],timeout=30);last=time.monotonic()
            stop.wait(5)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:
        with (root/'server.log').open('a',buffering=1) as log:
            old=command(plan);cmd=[old[0],'/opt/v12/gpu_entry.py']+old[3:]+['--no-enable-prefix-caching','--disable-log-requests']
            (root/'effective_command.json').write_text(json.dumps(cmd,indent=2))
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
            while proc.poll() is None:
                if time.monotonic()-start>=plan['gpu_timeout_s']-30 or (activity['ready'] and time.monotonic()-activity['changed']>900):
                    proc.terminate()
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:proc.kill()
                    break
                time.sleep(2)
            raise SystemExit(proc.wait())
    finally:stop.set();thread.join(timeout=35);subprocess.run(['sync','/gpu-logs'],timeout=60)
if __name__=='__main__':main()
