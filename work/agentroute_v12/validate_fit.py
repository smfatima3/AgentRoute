"""Linux validation of restored fitted adapters, with no model generation."""
import argparse,json,sys,time,os,hashlib
from pathlib import Path
P=Path('/work/project');sys.path.insert(0,str(P))
from agentroute.jobs import resources,verify_upstream,set_seed
from agentroute.runtime import Runtime
from agentroute.compact_checkpoint import load

def main():
    a=argparse.ArgumentParser();a.add_argument('--method',required=True);args=a.parse_args();m=args.method
    c=json.loads((P/'configs/compact.json').read_text());manifest=json.loads((P/'compact_data/manifest.json').read_text())
    start=time.time();set_seed(17);upstream=verify_upstream(m,P/'upstream')
    tok,emb=resources(c,manifest,c['models'][0],with_embedding=True)
    rt=Runtime(c,c['models'][0],manifest['models'][c['models'][0]],'http://127.0.0.1:1/v1',tok,emb,17,'load-validation')
    if m=='agentnet':
        from agentroute.agentnet_adapter import AgentNetAdapter,load_encoder
        adapter=AgentNetAdapter(rt,P/'upstream',load_encoder(manifest['models']['BAAI/bge-large-en-v1.5']),P/'validation/native')
    else:
        from agentroute.graph_adapters import GraphAdapter
        adapter=GraphAdapter(m,rt,P/'upstream')
    state,meta=load(m,c,manifest);adapter.restore(state)
    assert adapter.trained and not rt.ledger
    details={'keys':list(state)}
    if m=='agentnet':
        details['agent_total_task_num']={str(k):v['total_task_num'] for k,v in state['agents'].items()}
        details['agent_successful_task_num']={str(k):v['successful_task_num'] for k,v in state['agents'].items()}
    else:
        details['state_shapes']={str(k):str(getattr(v,'shape',type(v).__name__)) for k,v in state.items()}
    out=dict(method=m,status='loaded_and_restored',upstream=upstream,checkpoint_sha256=meta['checkpoint_sha256'],
      model_calls=0,elapsed_s=time.time()-start,details=details,initialization_embedding_events=rt.embedding_events)
    p=P/'validation'/f'{m}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
