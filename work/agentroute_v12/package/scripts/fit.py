"""Train only on the frozen TRAIN partition; save costs and checkpoint provenance."""
import argparse,json,pickle,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentroute.core import validate_config,digest,write_json,write_rows,file_hash
from agentroute.jobs import ROOT,select_rows,set_seed,resources,verify_upstream,code_hashes
from agentroute.runtime import Runtime,public_task
from agentroute.evaluation import score_text

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['safesieve','card','agentnet'],required=True)
    p.add_argument('--model',required=True);p.add_argument('--dataset',choices=['mmlu','svamp','gsm8k','bbh'],required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8000/v1');p.add_argument('--config',type=Path,default=ROOT/'configs/study.json')
    p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--upstream',type=Path,default=ROOT/'upstream')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.upstream=a.upstream.resolve()
    c=json.loads(a.config.read_text());validate_config(c);mf=json.loads((a.data/'manifest.json').read_text())
    if a.model not in c['models']+c.get('optional_models',[]):p.error('Model is not enabled in this package')
    import torch
    if a.output.exists():raise RuntimeError('Use a new checkpoint directory; existing fit logs are immutable')
    a.output.mkdir(parents=True)
    spec=verify_upstream(a.method,a.upstream);set_seed(c['fit']['seed'])
    rows=select_rows(a.data,a.dataset,'train',0);tokenizer,embedding=resources(c,mf,a.model)
    def runtime(id):return Runtime(c,a.model,mf['models'][a.model],a.base_url,tokenizer,embedding,c['fit']['seed'],id)
    init=runtime('fit-initialization');start=time.perf_counter()
    if a.method=='agentnet':
        from agentroute.agentnet_adapter import AgentNetAdapter,load_encoder
        adapter=AgentNetAdapter(init,a.upstream,load_encoder(mf['models']['BAAI/bge-large-en-v1.5']),a.output/'native_state')
    else:
        from agentroute.graph_adapters import GraphAdapter
        adapter=GraphAdapter(a.method,init,a.upstream);adapter.start_fit()
        if a.method=='safesieve':
            try:adapter.initialize_similarity(init)
            finally:write_json(a.output/'initialization.json',init.summary())
    write_json(a.output/'initialization.json',init.summary())
    for iteration in range(c['fit']['iterations']):
        results=[]
        for j in range(c['fit']['batch_size']):
            index=iteration*c['fit']['batch_size']+j;row=rows[index%len(rows)]
            rt=runtime(f'{row["id"]}/fit/{index}');out={'id':row['id'],'split':'train','iteration':iteration,'status':'error'}
            try:
                result=adapter.predict(rt,public_task(row),training=True)
                answer,logp=result if isinstance(result,tuple) else (result,None)
                metrics=score_text(answer,row);out.update(metrics);out['final_text']=answer
                if any(call['status']!='ok' for call in rt.ledger):raise RuntimeError('Failed training inference')
                if a.method=='agentnet':adapter.learn(metrics['quality'])
                else:
                    graph=adapter.graph;adj=graph.spatial_adj_matrix
                    if hasattr(adj,'detach'):adj=adj.detach().cpu().numpy()
                    penalty=(torch.sigmoid(graph.spatial_logits)*graph.spatial_masks).sum()*c['card']['relative_edge_price'] if a.method=='card' else None
                    results.append((metrics['quality'],logp,adj,penalty))
                out['status']='ok'
            except Exception as exc:
                out['error_type']=type(exc).__name__;raise
            finally:
                out.update(rt.summary());write_json(a.output/f'train_{index:05d}.json',out)
        loss=adapter.fit_batch(results,iteration) if a.method!='agentnet' else None
        write_json(a.output/f'iteration_{iteration:03d}.json',{'iteration':iteration,'loss':loss,
            'optimizer_diagnostics':adapter.fit_log[-1] if a.method=='card' and adapter.fit_log else None})
        print('completed fitting iteration',iteration+1,flush=True)
    state=adapter.export();checkpoint=a.output/('checkpoint.pkl' if a.method=='agentnet' else 'checkpoint.pt')
    if a.method=='agentnet':
        with checkpoint.open('wb') as f:pickle.dump(state,f)
    else:torch.save(state,checkpoint)
    write_json(checkpoint.with_suffix('.json'),{'method':a.method,'dataset':a.dataset,'model':a.model,
        'model_revision':mf['models'][a.model],'checkpoint_sha256':file_hash(checkpoint),
        'config_hash':digest(c),'source_hashes':code_hashes(),'data_files':{f'{a.dataset}/train.jsonl':mf['files'][f'{a.dataset}/train.jsonl']},
        'train_ids':[r['id'] for r in rows],'upstream':spec,'fit_wall_s':time.perf_counter()-start,
        'claim_status':'adaptation_pending_native_parity'})
    print(checkpoint)

if __name__=='__main__':main()
