"""Run actual model inference; preserve every attempt and stop on infrastructure errors."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentroute.core import validate_config,digest,write_json,file_hash
from agentroute.jobs import ROOT,select_rows,set_seed,resources,verify_upstream,checkpoint_load,code_hashes,stable_job_identity
from agentroute.runtime import Runtime,public_task,run_cascade
from agentroute.evaluation import score_text,score_code
from agentroute.diagnostics import answer_health

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=ROOT/'configs/study.json')
    p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--dataset',required=True)
    p.add_argument('--model',required=True);p.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--method',required=True);p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--limit',type=int,default=80);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--output',type=Path,default=ROOT/'results');p.add_argument('--upstream',type=Path,default=ROOT/'upstream')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--lock',type=Path)
    p.add_argument('--evaluator-image',default=os.getenv('AGENTROUTE_EVAL_IMAGE'))
    p.add_argument('--require-format',action='store_true',help='Fail the setup stage if any saved answer has invalid formatting')
    a=p.parse_args();a.upstream=a.upstream.resolve()
    c=json.loads(a.config.read_text());validate_config(c);manifest=json.loads((a.data/'manifest.json').read_text())
    if a.method not in c['methods']+c['controls']:p.error('Unknown method')
    if a.model not in c['models']+c['optional_models']:p.error('Unknown model')
    if a.limit<0:p.error('limit must be >= 0')
    rows=select_rows(a.data,a.dataset,a.split,a.limit)
    if not rows:raise RuntimeError('No examples in selected split; HumanEval has no development set')
    if any(not r['answers'] for r in rows):raise RuntimeError('Resolve gold extraction before inference')
    hashes=code_hashes();ckmeta=None
    if a.split=='test':
        if not a.lock:raise RuntimeError('Freeze the protocol before held-out evaluation')
        lock=json.loads(a.lock.read_text())
        if lock['config_hash']!=digest(c) or lock['code_hashes']!=hashes or lock['data_hash']!=file_hash(a.data/'manifest.json'):
            raise RuntimeError('Protocol changed after freeze')
        if [r['id'] for r in rows]!=lock['test_ids'][a.dataset]:raise RuntimeError('Test IDs differ from freeze')
        if a.seed not in c['seeds']:raise ValueError('Seed not in frozen protocol')
        if a.method in ['safesieve','card','agentnet']:
            if not a.checkpoint or file_hash(a.checkpoint) not in lock['checkpoint_hashes']:raise ValueError('Checkpoint absent from freeze')
    if a.dataset=='humaneval':
        import shutil
        if not shutil.which('docker') or not a.evaluator_image or '@sha256:' not in a.evaluator_image:
            raise RuntimeError('HumanEval requires Docker and AGENTROUTE_EVAL_IMAGE pinned by digest')
    # Validate endpoint availability before downloading model resources or attempting work.
    import urllib.request
    headers={}
    if c.get('api_key_env'):headers['Authorization']='Bearer '+os.environ[c['api_key_env']]
    from agentroute.server import wait_ready
    wait_ready(a.base_url,a.model,headers=headers)
    if a.method in ['safesieve','card','agentnet','selforg']:
        verify_upstream(a.method,a.upstream)
    tokenizer,embedding=resources(c,manifest,a.model,with_embedding=a.method in ['safesieve','card','agentnet','selforg'])
    rt0=Runtime(c,a.model,manifest['models'][a.model],a.base_url,tokenizer,embedding,a.seed,'initialize')
    adapter=None;state=None;baseline_spec=None
    if a.method in ['safesieve','card','agentnet','selforg']:
        baseline_spec=verify_upstream(a.method,a.upstream)
    if a.method in ['safesieve','card']:
        from agentroute.graph_adapters import GraphAdapter
        adapter=GraphAdapter(a.method,rt0,a.upstream)
    if a.method=='agentnet':
        from agentroute.agentnet_adapter import AgentNetAdapter,load_encoder
        encoder=load_encoder(manifest['models']['BAAI/bge-large-en-v1.5'])
        adapter=AgentNetAdapter(rt0,a.upstream,encoder,a.output/'agentnet_internal')
    if adapter:
        if not a.checkpoint:raise RuntimeError('A trained checkpoint is required for '+a.method)
        state,ckmeta=checkpoint_load(a.checkpoint,a.method,a.model,c,manifest)
        # Each dataset gets its own fit. HumanEval explicitly uses the MMLU
        # checkpoint as frozen transfer because HumanEval has no train/dev split.
        expected='mmlu' if a.dataset=='humaneval' else a.dataset
        if ckmeta['dataset']!=expected:raise RuntimeError('Wrong task checkpoint; HumanEval transfers from MMLU')
        adapter.restore(state)
    identity={'dataset':a.dataset,'model':a.model,'model_revision':manifest['models'][a.model],
        'method':a.method,'split':a.split,'seed':a.seed,'config_hash':digest(c),'source_hashes':hashes,
        'data_sha256':manifest['files'][f'{a.dataset}/{a.split}.jsonl'],'ids':[r['id'] for r in rows],
        'checkpoint_sha256':file_hash(a.checkpoint) if a.checkpoint else None,
        'evaluator_image':a.evaluator_image if a.dataset=='humaneval' else None,
        'baseline_spec':baseline_spec,'claim_status':'exploratory_adapter' if baseline_spec else 'unvalidated_study',
        'initialization_embedding_events':rt0.embedding_events,
        'endpoint':a.base_url,'software':__import__('subprocess').check_output([sys.executable,'-m','pip','freeze'],text=True) if __import__('importlib').util.find_spec('pip') else 'pip_unavailable'}
    dest=a.output/digest(stable_job_identity(identity))[:20];dest.mkdir(parents=True,exist_ok=True)
    job_path=dest/'job.json'
    if job_path.exists():
        if stable_job_identity(json.loads(job_path.read_text()))!=stable_job_identity(identity):
            raise RuntimeError('Existing job identity differs; preserve it and use a new campaign')
    else:write_json(job_path,identity)
    # Preserve repeat initialization overhead separately without creating duplicate experiment cells.
    from datetime import datetime,timezone
    write_json(ROOT/'setup/job_initializations'/dest.name/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
               {'events':rt0.embedding_events,'model':a.model,'method':a.method})
    for i,row in enumerate(rows):
        path=dest/(digest(row['id'])[:20]+'.json')
        if path.exists():
            previous=json.loads(path.read_text(encoding='utf-8'))
            if previous.get('status')!='ok' or not previous.get('usage_complete'):
                raise RuntimeError('Preserved incomplete attempt requires explicit recovery; it cannot be skipped as completed')
            continue
        set_seed(a.seed);rt=Runtime(c,a.model,manifest['models'][a.model],a.base_url,tokenizer,embedding,a.seed,row['id'])
        out={'id':row['id'],'dataset':a.dataset,'model':a.model,'method':a.method,'seed':a.seed,
             'split':a.split,'status':'error','quality':0.,'format_ok':False,'claim_status':identity['claim_status']}
        start=time.perf_counter()
        import psutil
        out['started_unix_s']=time.time();out['started_monotonic_s']=start
        out['client_rss_start_bytes']=psutil.Process().memory_info().rss
        cpu_start=time.process_time()
        try:
            task=public_task(row)
            if adapter:
                adapter.restore(state) # no held-out feedback or memory carry-over
                answer=adapter.predict(rt,task)
                if isinstance(answer,tuple):answer=answer[0]
            elif a.method=='selforg':
                from agentroute.selforg_adapter import run
                answer=run(rt,task,a.upstream)
            else:answer=run_cascade(rt,task,a.method)
            out['inference_wall_s']=time.perf_counter()-start
            out.update(final_text=answer)
            out['answer_health']=answer_health(answer,rt.ledger)
            out.update(score_code(answer,row,a.evaluator_image) if row['metric']=='code' else score_text(answer,row))
            if any(call['status']!='ok' for call in rt.ledger):raise RuntimeError('Incomplete inference call')
            out['status']='ok'
        except Exception as exc:out['error_type']=type(exc).__name__
        out['wall_s']=time.perf_counter()-start;out['ended_unix_s']=time.time()
        out['client_cpu_s']=time.process_time()-cpu_start
        out['client_rss_end_bytes']=psutil.Process().memory_info().rss
        out.update(rt.summary());write_json(path,out)
        print(json.dumps({'completed':i+1,'total':len(rows),'id':row['id'],'status':out['status'],'path':str(path)}),flush=True)
        if out['status']!='ok':raise RuntimeError('Attempt saved; job stopped. Fix the cause in a separately recorded revision.')
    if a.require_format:
        saved=[json.loads((dest/(digest(row['id'])[:20]+'.json')).read_text()) for row in rows]
        if any(not r['format_ok'] or r['status']!='ok' for r in saved):
            raise RuntimeError('Task-format check failed; all attempts are preserved. Inspect final_text and answer_health before continuing.')

if __name__=='__main__':main()
