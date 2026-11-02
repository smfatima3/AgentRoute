"""Run the pinned vLLM server with authenticated HTTPS supplied by the Modal tunnel."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def command(plan):
    return ['/opt/gpu/bin/python','-m','vllm.entrypoints.openai.api_server',
        '--model',plan['model'],'--served-model-name',plan['model'],'--revision',plan['revision'],
        '--tokenizer-revision',plan['revision'],'--host','0.0.0.0','--port','8000',
        '--dtype','bfloat16','--generation-config','vllm','--seed','97',
        '--max-model-len',str(plan['max_model_len']),'--gpu-memory-utilization','0.9',
        '--tensor-parallel-size',str(plan['tensor_parallel_size']),'--max-num-seqs','2']

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-json',required=True);a=p.parse_args();plan=json.loads(a.plan_json)
    if not os.environ.get('VLLM_API_KEY'):raise RuntimeError('Missing per-run inference authentication key')
    root=Path('/opt/package');info=json.loads((root/'PACKAGE_INFO.json').read_text())
    if (info['model'],info['model_revision'])!=(plan['model'],plan['revision']):raise RuntimeError('Model or revision mismatch')
    logs=Path('/gpu-logs')/plan['operation_id'];logs.mkdir(parents=True,exist_ok=True)
    log=open(logs/'server.log','a',buffering=1)
    os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    subprocess.run(['/opt/gpu/bin/python',str(root/'scripts/check_serving_env.py'),'--model',plan['model'],
        '--require-cuda','--tensor-parallel-size',str(plan['tensor_parallel_size']),
        '--report',str(logs/'serving_preflight.json')],check=True)
    # vLLM 0.8.5 reads VLLM_API_KEY from the injected Secret; never put it in arguments or logs.
    cmd=command(plan);os.execv(cmd[0],cmd)

if __name__=='__main__':main()
