"""One isolated method shard; immutable per-task evidence committed before receipt."""
import argparse,hashlib,json,os,sys,time,threading,subprocess,traceback
from pathlib import Path
ROOT=Path('/work/project');sys.path.insert(0,str(ROOT))
from agentroute.core import read_rows,file_hash
from agentroute.jobs import set_seed,resources,verify_upstream
from agentroute.runtime import Runtime,public_task,run_cascade
from agentroute.compact_checkpoint import load as load_checkpoint
from agentroute.evaluation import score_text
from agentroute.diagnostics import answer_health
from telemetry import Spans,policy_overlay,resolved_config,selforg_original,self_consistency

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode();temp=p.with_suffix('.partial')
    with temp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    temp.replace(p)
def commit():subprocess.run(['sync','/evidence'],check=True,timeout=60)
def receipt(directory,task,attempt,epoch,raw,status):
    paths={'raw':'raw.json','metrics':'raw.json','score':'score.json','metering':'journal.jsonl'}
    evidence=[dict(role=k,path=(directory/v).relative_to('/evidence').as_posix(),sha256=hashlib.sha256((directory/v).read_bytes()).hexdigest()) for k,v in paths.items()]
    result=dict(task_id=task['task_id'],group=task['group'],fingerprint=task['fingerprint'],attempt_id=attempt,epoch=epoch,status=status,
      outcome='protocol_timeout' if raw['status']=='protocol_timeout' else 'correct' if raw.get('quality')==1 else 'incorrect' if raw.get('format_ok') else 'invalid_answer',evidence=evidence,
      completed_unix_s=time.time(),response_ids=[x.get('response_id') for x in raw.get('calls',[])])
    save(directory/'receipt.json',result);commit()

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();job=json.loads(Path(a.job).read_text())
    manifest=json.loads((ROOT/'compact_data/manifest.json').read_text());base=json.loads((ROOT/'configs/compact.json').read_text())
    for rel,h in job['source_lock'].items():
        if file_hash(ROOT/rel)!=h:raise RuntimeError('Frozen source mismatch '+rel)
    method=job['tasks'][0]['method'];c=resolved_config(base,job['tasks'][0]);prefix=Path('/evidence')/job['prefix'];prefix.mkdir(parents=True,exist_ok=True)
    save(prefix/'job.json',job);commit()
    baseline=method in ['safesieve','card','agentnet','selforg','selforg_original']
    if baseline:verify_upstream('selforg' if method=='selforg_original' else method,ROOT/'upstream')
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    tok,emb=resources(c,manifest,c['models'][0],with_embedding=baseline)
    def runtime(t):
        rt=Runtime(c,c['models'][0],manifest['models'][c['models'][0]],os.environ['AGENTROUTE_BASE_URL'],tok,emb,t['seed'],t['question_id'])
        if job.get('crash_after_first_model_call') and t['question_id']!='fixed-warmup':
            original_call=rt.call
            def crash_barrier(*args,**kwargs):
                text=original_call(*args,**kwargs)
                rt.log_event({'event':'intentional_development_crash_barrier','completed_model_calls':len(rt.ledger)})
                save(prefix/'CRASH_READY.json',{'question_id':rt.row_id,'completed_model_calls':len(rt.ledger),'unix_s':time.time()});commit()
                while True:time.sleep(1)
            rt.call=crash_barrier
        return rt
    init=runtime(job['tasks'][0]);adapter=None;state=None;set_seed(job['tasks'][0]['seed'])
    if method in ['safesieve','card']:
        from agentroute.graph_adapters import GraphAdapter
        adapter=GraphAdapter(method,init,ROOT/'upstream')
    elif method=='agentnet':
        from agentroute.agentnet_adapter import AgentNetAdapter,load_encoder
        adapter=AgentNetAdapter(init,ROOT/'upstream',load_encoder(manifest['models']['BAAI/bge-large-en-v1.5']),prefix/'native_internal')
    if adapter:
        state,meta=load_checkpoint(method,c,manifest);adapter.restore(state)
        save(prefix/'checkpoint.json',{'checkpoint_sha256':meta['checkpoint_sha256'],'initialization_embedding_events':init.embedding_events})
    warm=runtime(dict(job['tasks'][0],question_id='fixed-warmup'));warm.attach_budget(None,job['attempt']+'/warmup','warmup',prefix/'warmup.journal.jsonl')
    for i in range(2):warm.call('Compute 12 + 13. End with FINAL: followed by the integer.',max_tokens=64,call_id=('warmup',i))
    save(prefix/'warmup.json',warm.summary());commit()
    import psutil
    process=psutil.Process()
    for task in job['tasks']:
        if not (Path('/evidence')/job['fence_file']).exists():raise RuntimeError('Lease fenced; refusing new task')
        lease=json.loads((Path('/evidence')/job['fence_file']).read_text())
        if lease['epoch']!=job['epoch'] or lease['state']!='active':raise RuntimeError('Lease epoch invalid')
        relative=f"{task['dataset']}/{task['split']}.jsonl"
        assert file_hash(ROOT/'compact_data'/relative)==manifest['files'][relative]
        row=next(r for r in read_rows(ROOT/'compact_data'/relative) if r['id']==task['question_id'])
        directory=prefix/task['task_id'];directory.mkdir(parents=True,exist_ok=False)
        attempt=job['attempt']+'/'+task['task_id'];set_seed(task['seed']);rt=runtime(task)
        rt.attach_budget(None,attempt,task['group'],directory/'journal.jsonl')
        rt.log_event({'event':'task_started','fingerprint':task['fingerprint'],'epoch':job['epoch'],'task_id':task['task_id']})
        start=time.perf_counter();cpu=time.process_time();stop=threading.Event();samples=[]
        def sample():
            while not stop.wait(.05):samples.append({'monotonic_s':time.perf_counter(),'rss':process.memory_info().rss,'cpu':process.cpu_times()._asdict()})
        sampler=threading.Thread(target=sample,daemon=True);sampler.start()
        raw=dict(id=row['id'],dataset=task['dataset'],split=task['split'],method=method,seed=task['seed'],status='infrastructure_error',
          task_id=task['task_id'],attempt_id=attempt,fingerprint=task['fingerprint'],epoch=job['epoch'],started_unix_s=time.time(),
          assigned_gpu=job['gpu'],client_sandbox=job['client_sandbox_id'],stratum=row.get('panel_stratum',row.get('stratum','all')),
          source_provenance=job['source_lock_sha256'],runtime_lineage=job.get('runtime_lineage'),measurement_mode='live_inference_logical_peer_transport',
          scheduling=job['scheduling'],peer_transport_note='In-process peer delivery; separate physical transport/live anchors required')
        spans=Spans()
        try:
            public=public_task(row)
            with spans,policy_overlay(method):
                if adapter:
                    adapter.restore(state);answer=adapter.predict(rt,public)
                    if isinstance(answer,tuple):answer=answer[0]
                elif method=='selforg':
                    from agentroute.selforg_adapter import run
                    answer=run(rt,public,ROOT/'upstream')
                elif method=='selforg_original':answer=selforg_original(rt,public,ROOT/'upstream')
                elif method=='self_consistency':answer=self_consistency(rt,public,task['token_budget'])
                else:answer=run_cascade(rt,public,method)
            raw.update(inference_wall_s=time.perf_counter()-start,final_text=answer,answer_health=answer_health(answer,rt.ledger))
            raw.update(score_text(answer,row))
            if not rt.ledger or any(x['status']!='ok' for x in rt.ledger):raise RuntimeError('Incomplete model-call ledger')
            raw['status']='ok'
        except Exception as exc:
            raw['error_type']=type(exc).__name__;raw['error_traceback']=traceback.format_exc(limit=8)
            if isinstance(exc,RuntimeError) and str(exc)=='AgentNet per-query call ceiling exceeded':
                raw.update(status='protocol_timeout',quality=0.,format_ok=False,prediction=None,final_text='',protocol_limit='prespecified AgentNet call ceiling')
        finally:
            stop.set();sampler.join(timeout=1);raw.update(rt.summary());raw.update(wall_s=time.perf_counter()-start,client_cpu_s=time.process_time()-cpu,ended_unix_s=time.time(),
              causal_spans=spans.events,client_resource_samples=samples,client_rss_peak_sampled_bytes=max([x['rss'] for x in samples]+[process.memory_info().rss]),
              span_note='Inclusive/exclusive function spans, exclusive removes nested tracked model/delivery/native stages; Python profiler overhead included. CPU process RSS is not an allocation attribution.')
            save(directory/'raw.json',raw);save(directory/'score.json',{k:raw.get(k) for k in ['quality','format_ok','prediction','evaluation_status','status']});commit()
            receipt(directory,task,attempt,job['epoch'],raw,'measured' if raw['status'] in ['ok','protocol_timeout'] else 'infrastructure_failure')
            print(json.dumps({'task_id':task['task_id'],'status':raw['status']}),flush=True)
        if raw['status'] not in ['ok','protocol_timeout']:raise RuntimeError('Failed first attempt preserved')
    save(prefix/'complete.json',{'tasks':len(job['tasks']),'status':'finished','unix_s':time.time()});commit()
if __name__=='__main__':main()
