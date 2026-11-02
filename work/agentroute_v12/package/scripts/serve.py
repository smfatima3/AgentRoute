"""Launch one vLLM replica with an exact model revision. Run in a serving environment."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--port',type=int,default=8000)
    p.add_argument('--gpu',default='0');p.add_argument('--tensor-parallel-size',type=int,default=1)
    p.add_argument('--max-model-len',type=int,default=32768);p.add_argument('--gpu-memory-utilization',type=float,default=.9)
    p.add_argument('--print-only',action='store_true');a=p.parse_args()
    config=json.loads((ROOT/'configs/study.json').read_text())
    if a.model not in config['models']+config.get('optional_models',[]):p.error('Model is not enabled in this package')
    mf=json.loads((ROOT/'data/manifest.json').read_text());revision=mf['models'][a.model]
    command=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',a.model,'--served-model-name',a.model,'--revision',revision,
        '--tokenizer-revision',revision,'--host','127.0.0.1','--port',str(a.port),
        '--dtype','bfloat16','--generation-config','vllm','--seed','97',
        '--max-model-len',str(a.max_model_len),'--gpu-memory-utilization',str(a.gpu_memory_utilization),
        '--tensor-parallel-size',str(a.tensor_parallel_size),'--max-num-seqs','2']
    import shlex
    print(shlex.join(command),flush=True)
    if not a.print_only:
        subprocess.run([sys.executable,str(ROOT/"scripts/check_serving_env.py"),"--versions-only"],check=True)
        env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=a.gpu
        subprocess.run(command,env=env,check=True)
