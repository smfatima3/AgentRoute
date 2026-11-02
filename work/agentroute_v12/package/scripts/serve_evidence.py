"""Pinned vLLM process plus real GPU samples and raw server metrics."""
import argparse,json,os,re,subprocess,sys,threading,time,urllib.request
from pathlib import Path
sys.path.insert(0,'/opt/control')
from modal_runner.gpu_server import command

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-json',required=True);a=p.parse_args();plan=json.loads(a.plan_json)
    if not os.environ.get('VLLM_API_KEY'):raise RuntimeError('Inference key required')
    directory=Path('/gpu-logs')/plan['operation_id'];directory.mkdir(parents=True,exist_ok=True)
    subprocess.run(['/opt/gpu/bin/python','/opt/package/scripts/check_serving_env.py','--model',plan['model'],
        '--require-cuda','--tensor-parallel-size','1','--report',str(directory/'serving_preflight.json')],check=True)
    stop=threading.Event()
    activity={'count':None,'changed':time.monotonic(),'ready':False}
    def sample():
        with (directory/'resource_samples.jsonl').open('a',buffering=1) as f:
            while not stop.is_set():
                row={'unix_s':time.time()}
                try:
                    r=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                    row['nvidia_smi_csv']=r.stdout.strip();row['nvidia_smi_status']=r.returncode
                except Exception as exc:row['gpu_sample_error']=type(exc).__name__
                try:
                    req=urllib.request.Request('http://127.0.0.1:8000/metrics',headers={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']})
                    with urllib.request.urlopen(req,timeout=2) as response:row['vllm_prometheus']=response.read().decode()
                    count=sum(float(v) for v in re.findall(r'^vllm:request_success_total\{[^\n]*\}\s+([\d.e+]+)',row['vllm_prometheus'],re.M))
                    if count!=activity['count']:activity.update(count=count,changed=time.monotonic())
                    activity['ready']=True
                except Exception as exc:row['server_metrics_error']=type(exc).__name__
                f.write(json.dumps(row)+'\n');stop.wait(5)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:
        with (directory/'server.log').open('a',buffering=1) as log:
            cmd=command(plan)+['--no-enable-prefix-caching','--disable-log-requests']
            (directory/'effective_command.json').write_text(json.dumps(cmd,indent=2))
            run=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);started=time.monotonic()
            while run.poll() is None:
                if time.monotonic()-started>=plan['gpu_timeout_s']-30 or (activity['ready'] and time.monotonic()-activity['changed']>900):
                    log.write('Campaign deadline or 900-second request-idle watchdog reached\n');run.terminate()
                    try:run.wait(timeout=20)
                    except subprocess.TimeoutExpired:run.kill()
                    break
                time.sleep(2)
            raise SystemExit(run.wait())
    finally:stop.set();thread.join(timeout=10)
if __name__=='__main__':main()
