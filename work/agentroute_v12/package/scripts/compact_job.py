"""One isolated method/cell; all model calls are durably reserved and journaled."""
import argparse,copy,json,os,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agentroute.core import write_json,file_hash,digest,read_rows
from agentroute.jobs import set_seed,resources,verify_upstream,code_hashes
from agentroute.runtime import Runtime,public_task,run_cascade
from agentroute.compact_budget import Budget
from agentroute.compact_checkpoint import load as load_checkpoint
from agentroute.evaluation import score_text
from agentroute.diagnostics import answer_health

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    job=json.loads(Path(a.job).read_text());config=json.loads((ROOT/'configs/compact.json').read_text())
    manifest=json.loads((ROOT/'compact_data/manifest.json').read_text());method=job['method'];seed=job['seed']
    c=copy.deepcopy(config);choice=job.get('candidate','original')
    c['evidence_arms']['cascade']=copy.deepcopy(c['evidence_arms'][choice])
    c['evidence_arms']['route_fixed']=dict(c['evidence_arms']['cascade'],early_stop=False)
    c['evidence_arms']['toggle_filter']=dict(c['evidence_arms']['cascade'],filter_dissent=not c['evidence_arms']['cascade']['filter_dissent'])
    if job.get('agents',4)!=4:
        c['agents']=job['agents'];c['roles']+=['Algebra checker','Logic checker','Alternative solver','Evidence reviewer']
        for arm in c['evidence_arms'].values():arm['k']=c['agents']-1
    dataset=job['dataset'];split=job['split'];relative=f'{dataset}/{split}.jsonl'
    if file_hash(ROOT/'compact_data'/relative)!=manifest['files'][relative]:raise RuntimeError('Input hash mismatch')
    rows={r['id']:r for r in read_rows(ROOT/'compact_data'/relative)}
    if len(set(job['ids']))!=len(job['ids']) or not set(job['ids'])<=set(rows):raise RuntimeError('Duplicate or invalid IDs')
    if split=='test':
        lock=json.loads((ROOT/'setup/main_freeze.json').read_text())
        if lock['source_hashes']!=code_hashes() or lock['config_sha256']!=file_hash(ROOT/'configs/compact.json') or lock['data_sha256']!=file_hash(ROOT/'compact_data/manifest.json'):raise RuntimeError('Frozen inputs changed')
        if lock['candidate']!=choice:raise RuntimeError('Candidate changed after freeze')
    budget=Budget(ROOT/'results/budget.sqlite');outdir=ROOT/'results'/job['id'];outdir.mkdir(parents=True,exist_ok=True)
    identity=dict(job,config_sha256=file_hash(ROOT/'configs/compact.json'),resolved_config=c,source_hashes=code_hashes(),
        data_sha256=manifest['files'][relative],model_revision=manifest['models'][c['models'][0]],
        claim_status='common-task adaptation; selected panel only',initialized_unix_s=time.time())
    write_json(outdir/'job.json',identity)
    baseline=method in ['safesieve','card','agentnet','selforg'];upstream=ROOT/'upstream'
    if baseline:verify_upstream(method,upstream)
    tokenizer,embedding=resources(c,manifest,c['models'][0],with_embedding=baseline)
    def runtime(row_id):return Runtime(c,c['models'][0],manifest['models'][c['models'][0]],os.environ['AGENTROUTE_BASE_URL'],tokenizer,embedding,seed,row_id)
    init=runtime('initialize');adapter=None;state=None
    set_seed(seed)
    if method in ['safesieve','card']:
        from agentroute.graph_adapters import GraphAdapter
        adapter=GraphAdapter(method,init,upstream)
    elif method=='agentnet':
        from agentroute.agentnet_adapter import AgentNetAdapter,load_encoder
        adapter=AgentNetAdapter(init,upstream,load_encoder(manifest['models']['BAAI/bge-large-en-v1.5']),outdir/'native_internal')
    if adapter:
        state,meta=load_checkpoint(method,c,manifest);adapter.restore(state)
        write_json(outdir/'checkpoint.json',dict(checkpoint_sha256=meta['checkpoint_sha256'],source_dataset='mmlu',transfer=dataset!='mmlu'))
    # Fixed per-cell warmup: no benchmark content and no unmetered model calls.
    warm_id=job['id']+'/warmup'
    budget.start_attempt(warm_id,'warmup');warm=runtime('fixed-warmup')
    warm.attach_budget(budget,warm_id,'warmup',outdir/'warmup.journal.jsonl')
    for i in range(2):warm.call('Compute 12 + 13. End with FINAL: followed by the integer.',max_tokens=64,call_id=('warmup',i))
    budget.finish_attempt(warm_id,'ok');write_json(outdir/'warmup.json',warm.summary())
    import psutil
    process=psutil.Process()
    for index,id in enumerate(job['ids']):
        path=outdir/(digest(id)[:20]+'.json');attempt=job['id']+'/'+id
        if path.exists():
            prior=json.loads(path.read_text())
            if prior.get('id')==id and prior.get('status')=='ok' and prior.get('usage_complete'):continue
            raise RuntimeError('Incomplete first attempt retained; explicit audited recovery required')
        budget.start_attempt(attempt,job['phase']);row=rows[id];set_seed(seed);rt=runtime(id)
        rt.attach_budget(budget,attempt,job['phase'],outdir/(digest(id)[:20]+'.journal.jsonl'))
        out=dict(id=id,dataset=dataset,method=method,seed=seed,split=split,phase=job['phase'],
            attempt_id=attempt,status='error',quality=0.,format_ok=False,started_unix_s=time.time(),
            stratum=row.get('panel_stratum',row.get('stratum','all')),weight=1,
            client_rss_start_bytes=process.memory_info().rss,queue_note='Isolated sequential client per cell; at most two cells active; server aggregate queue sampled separately')
        peak=[out['client_rss_start_bytes']];done=threading.Event()
        def sample():
            while not done.wait(.05):peak[0]=max(peak[0],process.memory_info().rss)
        thread=threading.Thread(target=sample,daemon=True);thread.start();started=time.perf_counter();cpu=time.process_time()
        try:
            task=public_task(row)
            if adapter:
                adapter.restore(state);answer=adapter.predict(rt,task)
                if isinstance(answer,tuple):answer=answer[0]
            elif method=='selforg':
                from agentroute.selforg_adapter import run
                answer=run(rt,task,upstream)
            else:answer=run_cascade(rt,task,method)
            out['inference_wall_s']=time.perf_counter()-started;out['final_text']=answer
            out['answer_health']=answer_health(answer,rt.ledger);out.update(score_text(answer,row))
            if any(call['status']!='ok' for call in rt.ledger):raise RuntimeError('Incomplete model call')
            out['status']='ok'
        except Exception as exc:out['error_type']=type(exc).__name__
        finally:
            done.set();thread.join(timeout=1)
            out.update(wall_s=time.perf_counter()-started,ended_unix_s=time.time(),client_cpu_s=time.process_time()-cpu,
                client_rss_peak_sampled_bytes=peak[0],client_rss_end_bytes=process.memory_info().rss,
                memory_note='Whole client process sampled every 50ms; not router-exclusive allocation')
            out.update(rt.summary());write_json(path,out);budget.finish_attempt(attempt,out['status'])
            print(json.dumps({'job':job['id'],'completed':index+1,'total':len(job['ids']),'id':id,'status':out['status']}),flush=True)
        if out['status']!='ok':raise RuntimeError('First failed attempt preserved; stopping cell')
    write_json(outdir/'complete.json',{'status':'complete','count':len(job['ids']),'completed_unix_s':time.time()})
if __name__=='__main__':main()
