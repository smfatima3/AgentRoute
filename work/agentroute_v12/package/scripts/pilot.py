"""Bounded development pilot. Never opens or evaluates held-out questions."""
import json,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agentroute.core import write_json,file_hash
from agentroute.jobs import select_rows,code_hashes

def main():
    config=ROOT/'configs/evidence.json';c=json.loads(config.read_text())
    if c['phase']!='development':raise RuntimeError('Pilot only supports development')
    methods=['single','cascade','route_vote','dense_vote','route_paid','dense_paid','no_comm_vote']
    ids={d:[r['id'] for r in select_rows(ROOT/'evidence_data',d,'dev',10)] for d in c['datasets']}
    cells=[(d,m) for d in c['datasets'] for m in methods]
    random.Random(97).shuffle(cells)
    write_json(ROOT/'setup/pilot_protocol.json',dict(purpose='development_only_not_confirmatory',
        methods=methods,ids=ids,seed=17,cell_order=cells,client_concurrency=1,
        config_sha256=file_hash(config),source_hashes=code_hashes(),
        data_manifest_sha256=file_hash(ROOT/'evidence_data/manifest.json'),
        max_inference_wall_s=2700,hardware='1x L40S',stopping='first failure or 45-minute inference budget',
        paired_reference='dense_vote',noninferiority_claim=False))
    started=time.monotonic()
    for dataset,method in cells:
        remaining=2700-(time.monotonic()-started)
        if remaining<=0:raise RuntimeError('Pilot time cap reached; partial attempts preserved')
        command=[sys.executable,str(ROOT/'scripts/run.py'),'--config',str(config),'--data',str(ROOT/'evidence_data'),
            '--dataset',dataset,'--model',c['models'][0],'--method',method,'--split','dev','--limit','10','--seed','17',
            '--base-url',os.environ['AGENTROUTE_BASE_URL']]
        print(json.dumps({'starting':dataset+'/'+method,'remaining_seconds':round(remaining)}),flush=True)
        subprocess.run(command,cwd=ROOT,check=True,timeout=remaining)
    write_json(ROOT/'setup/pilot_complete.json',{'status':'complete','cells':len(cells),'queries':len(cells)*10,
        'wall_s':time.monotonic()-started,'held_out_inference':False})
if __name__=='__main__':main()
