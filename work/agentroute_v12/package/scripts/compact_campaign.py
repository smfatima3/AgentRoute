"""Two isolated query workers, immutable cells, prospective gates and hard caps."""
import collections,json,math,os,random,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agentroute.core import read_rows,write_json,file_hash,digest
from agentroute.jobs import code_hashes
from agentroute.compact_budget import Budget
from agentroute.compact_stats import analyze,paired

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def records(job):
    rows=[]
    for path in (ROOT/'results'/job['id']).glob('*.json'):
        value=read(path)
        if 'quality' in value and 'id' in value:rows.append(value)
    if set(r['id'] for r in rows)!=set(job['ids']):raise RuntimeError('Incomplete cell; retain data and stop comparison')
    return rows

def jobs_for(phase,datasets,methods,split='test',seeds=(17,),candidate='original',ablation=False):
    manifest=read(ROOT/'compact_data/manifest.json');jobs=[]
    for dataset in datasets:
        ids=manifest['ablation_ids'][dataset] if ablation else [r['id'] for r in read_rows(ROOT/'compact_data'/dataset/(split+'.jsonl'))]
        for seed in seeds:
            for method in methods:
                jobs.append(dict(id=f'{phase}_{dataset}_{method}_{seed}',phase=phase,dataset=dataset,split=split,method=method,seed=seed,candidate=candidate,ids=ids))
    random.Random(20261029+sum(map(ord,phase))).shuffle(jobs)
    return jobs

def run_jobs(jobs,budget):
    pending=list(jobs);active=[];logs=ROOT/'results/job_logs';logs.mkdir(parents=True,exist_ok=True);last_report=0
    write_json(ROOT/'setup'/('schedule_'+jobs[0]['phase']+'_'+jobs[0]['dataset']+'.json'),jobs)
    try:
        while pending or active:
            while pending and len(active)<2:
                job=pending.pop(0);dest=ROOT/'results'/job['id']
                if (dest/'complete.json').exists():records(job);continue
                jobfile=ROOT/'setup/jobs'/(job['id']+'.json');write_json(jobfile,job)
                log=(logs/(job['id']+'.log')).open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/compact_job.py'),'--job',str(jobfile)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                active.append((proc,job,log));print(json.dumps({'started':job['id'],'pid':proc.pid}),flush=True)
            for entry in list(active):
                proc,job,log=entry
                if proc.poll() is not None:
                    log.close();active.remove(entry)
                    if proc.returncode:raise RuntimeError('Cell failure: '+job['id'])
                    records(job);print(json.dumps({'complete':job['id']}),flush=True)
            write_json(ROOT/'results/progress.json',dict(unix_s=time.time(),active=[j['id'] for _,j,_ in active],pending=[j['id'] for j in pending],budget=budget.snapshot()))
            if time.monotonic()-last_report>30:
                snapshot=budget.snapshot();print(json.dumps({'progress_unix_s':time.time(),'active':[j['id'] for _,j,_ in active],
                    'attempts':snapshot['attempts_by_phase_status'],'tokens':snapshot['authoritative_tokens'],'reserved_total':snapshot['charged_tokens_including_outstanding']}),flush=True)
                last_report=time.monotonic()
            if time.time()>=budget.snapshot()['settings']['admission_deadline_unix']+300:raise RuntimeError('Hard client shutdown deadline')
            if active:time.sleep(2)
    finally:
        for proc,job,log in active:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()

def main():
    protocol=read(ROOT/'configs/compact_protocol.json');window=read(ROOT/'configs/execution_window.json')
    budget=Budget(ROOT/'results/budget.sqlite')
    budget.initialize({k:protocol[k] for k in ['scored_attempt_cap','total_token_cap','fit_token_cap']}|{
        'admission_deadline_unix':datetime.fromisoformat(window['admission_deadline_utc'].replace('Z','+00:00')).timestamp()})
    write_json(ROOT/'setup/executed_source_hashes.json',code_hashes())
    dev=jobs_for('development',['math','bbh'],['original','corroborating','no_comm','dense'],split='dev')
    run_jobs(dev,budget);groups=collections.defaultdict(list)
    for job in dev:groups[job['method']].extend(records(job))
    orig,cor=groups['original'],groups['corroborating']
    candidate='corroborating' if sum(r['quality'] for r in cor)>=sum(r['quality'] for r in orig) and sum(r['total_online_tokens'] for r in cor)<=1.25*sum(r['total_online_tokens'] for r in orig) else 'original'
    gate=paired(groups[candidate],groups['no_comm']);negative=not(gate['difference']>0 and gate['exact_mcnemar_p']<.05) and gate['candidate_tokens']>=gate['reference_tokens']
    import numpy as np
    slo=max(5,5*math.ceil(float(np.quantile([r['inference_wall_s'] for r in groups['dense']],.95))/5))
    freeze=dict(candidate=candidate,development_negative_gate=negative,slo_s=slo,source_hashes=code_hashes(),
        config_sha256=file_hash(ROOT/'configs/compact.json'),data_sha256=file_hash(ROOT/'compact_data/manifest.json'),
        checkpoint_reuse_sha256=file_hash(ROOT/'checkpoint_reuse.json'),frozen_unix_s=time.time(),gate=gate)
    write_json(ROOT/'setup/main_freeze.json',freeze)
    write_json(ROOT/'results/development_statistics.json',analyze(groups,candidate,slo))
    print(json.dumps({'development_complete':True,'candidate':candidate,'negative_gate':negative,'slo_s':slo}),flush=True)
    methods=read(ROOT/'configs/compact.json')['methods'];mainjobs=jobs_for('primary',['mmlu'],methods,candidate=candidate)
    run_jobs(mainjobs,budget);primary=analyze({j['method']:records(j) for j in mainjobs},slo=slo)
    write_json(ROOT/'results/primary_statistics.json',primary)
    pair=primary['paired']['no_comm'];useful=(pair['difference']>0 and pair['holm_p']<.05) or (pair['candidate_tokens']<pair['reference_tokens'] and pair['paired_conservative_ci95'][0]>-.02)
    decision=dict(development_negative_gate=negative,primary_useful_tradeoff=useful,
        optional_expansion_allowed=(not negative and useful),decision_unix_s=time.time(),budget=budget.snapshot())
    write_json(ROOT/'results/expansion_decision.json',decision);print(json.dumps(decision),flush=True)
    if not decision['optional_expansion_allowed']:
        write_json(ROOT/'results/campaign_complete.json',dict(status='stopped_at_prespecified_negative_gate',
            completed_scored_attempts=920,optional_cells='deferred by prospective evidence gate; low-cost transport remains',budget=budget.snapshot()))
        return
    for dataset in ['svamp','math','bbh']:
        jobs=jobs_for('transfer',[dataset],methods,candidate=candidate);run_jobs(jobs,budget)
        write_json(ROOT/'results'/(dataset+'_statistics.json'),analyze({j['method']:records(j) for j in jobs},slo=slo))
    controls=jobs_for('ablation',['mmlu','math'],['route_fixed','dense_fixed','random_fixed','toggle_filter'],candidate=candidate,ablation=True)
    run_jobs(controls,budget)
    seeds=jobs_for('seed_sensitivity',['mmlu','math'],['cascade','dense','no_comm'],seeds=(29,43),candidate=candidate,ablation=True)
    run_jobs(seeds,budget)
    write_json(ROOT/'results/campaign_complete.json',dict(status='quality_matrix_complete',live_anchor_status='awaiting verified transport setup',budget=budget.snapshot()))
if __name__=='__main__':main()
