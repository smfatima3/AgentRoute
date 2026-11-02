"""Generate the entire finite v12 scope and raw-verified reuse receipts."""
import collections,hashlib,json,math,shutil,sys,time
from pathlib import Path
from coordinator_store import Store,atomic_json,hash_file
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1];OUT=BASE/'outputs/agentroute_v12_completion';P=ROOT/'package'
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
STAGES=['remove_answer_grouping','remove_representative_rank','remove_vote_summary']
MAIN=['cascade','dense','no_comm','safesieve','card','selforg','agentnet','single']
def evidence(role,path):return dict(role=role,path=path.relative_to(OUT).as_posix(),sha256=hash_file(path))

def main():
    if (OUT/'campaign.lock.json').exists():raise RuntimeError('Frozen campaign already exists; preserve and use an explicit lineage version for changes')
    m=read(P/'compact_data/manifest.json');reuse=read(OUT/'REUSE_AUDIT.json')
    prep=read(Path(read(OUT/'preparation_latest.json')['path'])/'state.json');assert prep['status']=='CPU_PREPARATION_PASSED' and len(prep['verified_fits'])==3
    package_sources={p.relative_to(P).as_posix():hash_file(p) for d in ['agentroute','scripts','configs','requirements'] for p in sorted((P/d).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    overlay_sources={p.relative_to(ROOT).as_posix():hash_file(p) for p in ROOT.glob('*.py')}
    source_lock=dict(package=package_sources,overlay=overlay_sources,original_policy='v11 original unchanged',upstream=read(P/'configs/upstream.json'))
    stage_note={
      'early_unanimity_stop':'mapped to route_fixed; no duplicate stage task',
      'same_answer_suppression':'mapped to toggle_filter; no duplicate stage task',
      'answer_grouping':'remove_answer_grouping: retain every distinct sender after dissent filtering',
      'representative_ranking':'remove_representative_rank: first candidate in original sender order per answer group, instead of evidence/length ranking',
      'global_vote_summary':'remove_vote_summary: remove prior answer-summary delivery and prompt addition; peer reports retained',
      'recipient_cap':'k=N-1 in executed source; cap removal is identical and adds no distinct setting',
      'payload_truncation':'unset in executed source; no active stage to remove',
      'candidate_scoring':'answer_key parsing/ranking only; no embeddings or paid router in AgentRoute; profiler records actual stage functions',
      'local_selector':'aggregation_controls plus fixed routing/dense controls form required 2x2',
      'random_fixed_note':'Frozen old random_fixed k=3 equals N-1 at N=4; all candidates survive; report this degeneracy rather than claim sparse random advantage',
      'source':'package/agentroute/evidence_policy.py'}
    budgets={'math':8987,'mmlu':3108,'svamp':8987,'bbh':3108}
    protocol=dict(model=m['models'],source_lock_sha256=digest(source_lock),split_manifest_sha256=hash_file(P/'compact_data/manifest.json'),
      stage_inventory=stage_note,policy_selection='No new policy selection; preserve original development candidate and historical MMLU evidence',
      inference={'gpu':'L40S','replicas':2,'queries_per_replica':2,'agents':4,'rounds_max_controlled':2,'max_tokens_per_call':2048,'temperature':.2,
        'thinking':False,'prefix_cache':False,'dtype':'bfloat16','max_model_len':32768,'vllm':'0.8.5','engine':'V1','torch_cpu_threads':4,'torch_interop_threads':1},
      scheduling={'shard_size':10,'method_block_rotation':'MMLU completion first, other main panels second, then controls; hash(seed=20261031,bucket) within tier; two independent query slots per device','timed_fit_overlap':False},
      fitting={'recipe':read(P/'configs/compact.json')['fit'],'source_dataset':'mmlu','source_questions':120,'reuse':'exact compatible checkpoint and all 120 training records plus 30 iterations validated'},
      selforg={'main':'published efficient consensus variant as implemented in pinned source: min >= .95 and range <= .1',
        'alternate':'same source with consensus early termination disabled; original fixed-round variant; common-task adaptation, not native-paper reproduction',
        'reference':'https://arxiv.org/html/2510.00685v2#A5'},
      self_consistency={'token_budgets':budgets,'derivation':'ceil mean original AgentRoute development total tokens: MATH8986.05, BBH3107.3. MMLU uses BBH discrete-answer budget; SVAMP uses math budget. No test outcomes used.',
        'selection':'local plurality of admissible answer keys, earliest sample tie break','reservation':'exact chat-template prompt + 2048 output before each independent sample'},
      transport={'host_runtime':'vm','regions':['us-east','us-west'],'shape_target_mbit_s':10,'extra_one_way_delay_ms':25,'bandwidth_tolerance_fraction':.15,
        'rtt_delta_tolerance_ms':15,'rate_calibration':'dense ramps at 1,4,12,32,64,128 qps, 8s steady+2s warm; first >5% failure/drop; low=max(.25,high/4). If bracket not found report saturation unlocated.',
        'rates_freeze':'separate immutable network calibration before any transport comparison; no quality-dependent calibration',
        'steady_duration':'max(60,ceil(1000/rate))','warmup_s':10,'deadline_s':30,'slo_s':5,'query_concurrency_cap':32,'delivery_threads':128,'repetitions':3,'seed':20261031,
        'live_anchor_slo_s':60,'live_anchor_selection':'first 10 fixed math development IDs, independent of outcomes'},
      statistics={'primary':'MMLU seed17, equal fixed-panel stratum weights as existing manifest','other_panels':'exploratory selected panels',
        'bootstrap':10000,'unit':'paired question; seeds clustered by question','tests':'Wilson accuracy; conservative paired gain-harm CI; exact McNemar and Holm7 per dataset; stratified paired token-ratio bootstrap',
        'noninferiority_margin':.02,'margin_status':'inherited screening threshold, not validated application acceptance criterion; never widen',
        'optional_stopping':False},
      persistence={'controller_location':'Modal CPU sandbox; single-writer SQLite on its mounted Volume, not shared across writers','no_codex_automation':True,
        'status_interval_s':300,'verified_result_export_interval_s':1200,'commit':'v2 sync after closed task records; provider readback SHA before acceptance; coordinator syncs /workspace after checkpoint','failed_attempts':'retained with usage and immutable journals'},
      budget={'user_reported_remaining_credit_usd':30,'execution_ceiling_usd':28,'safety_reserve_usd':2,'no_paid_spillover':True,'concurrency_limit_gpu':2,'automatic_infra_attempts':3},
      measurement_contract={'families':list(range(1,15)),'old_telemetry':'reuse only measured fields; per-request server waits and per-stage details missing in v11 remain missing until separately measured',
        'physical_bytes':'packet capture on isolated transport NIC; HTTP body not a wire-byte substitute','timing':'profiler overhead included, old and new timing cohorts separate',
        'required_measurement_supplements':'new MMLU telemetry on fixed ablation subset for methods with historical-only records; cannot substitute old quality outcome/latency'} )
    atomic_json(OUT/'source_lock.json',source_lock);atomic_json(OUT/'protocol.json',protocol);atomic_json(OUT/'STAGE_INVENTORY.json',stage_note)
    tasks=[]
    def add(group,method,dataset=None,q=None,seed=17,split='test',**extra):
        task=dict(group=group,method=method,dataset=dataset,question_id=q,seed=seed,split=split,**extra)
        task['task_id']=group+'-'+method+'-'+digest(task)[:20]
        task['fingerprint']=digest({'protocol':digest(protocol),'task':task});tasks.append(task)
    dev={d:[json.loads(x)['id'] for x in (P/f'compact_data/{d}/dev.jsonl').read_text(encoding='utf8').splitlines()] for d in ['math','bbh']}
    for d,ids in dev.items():
        for method in ['original','corroborating','dense','no_comm']:
            for q in ids:add('development',method,d,q,split='dev')
    for d,group in [('mmlu','main_mmlu'),('svamp','transfer_svamp'),('math','short_math'),('bbh','short_bbh')]:
        ids=[json.loads(x)['id'] for x in (P/f'compact_data/{d}/test.jsonl').read_text(encoding='utf8').splitlines()]
        for method in MAIN:
            for q in ids:add(group,method,d,q)
    for d,ids in m['ablation_ids'].items():
        for q in ids:
            for method in ['route_fixed','dense_fixed','random_fixed','toggle_filter']:add('component_ablations',method,d,q)
            for method in ['cascade','dense','no_comm']:
                for seed in [29,43]:add('seed_sensitivity',method,d,q,seed)
            for method in ['route_paid','dense_paid']:add('aggregation_controls',method,d,q)
            add('self_consistency','self_consistency',d,q,token_budget=budgets[d])
            add('selforg_variant','selforg_original',d,q)
            for method in STAGES:add('stage_ablations',method,d,q)
    for n in [4,8]:
        for method in ['cascade','dense']:
            for q in dev['math'][:10]:add('live_anchors',method,'math',q,split='dev',agents=n,physical_network=True)
    for method in ['cascade','dense','safesieve','card','selforg','agentnet']:
        for rep in range(3):add('transport_anchors',method,agents=4,mode='ordinary',load='low',repetition=rep)
    for method in ['cascade','dense']:
        for n in [4,8,16]:
            for mode in ['ordinary','constrained']:
                for load in ['low','high']:
                    for rep in range(3):add('transport_scaling',method,agents=n,mode=mode,load=load,repetition=rep)
    for method in ['safesieve','card','agentnet']:add('baseline_fitting',method,'mmlu',seed=11,split='train')
    counts=collections.Counter(t['group'] for t in tasks);spec=read(ROOT/'campaign_spec.json')
    required={**spec['generation_counts'],**spec['transport_counts'],**spec['fitting_counts'],'stage_ablations':120};assert dict(counts)==required,(counts,required)
    lock=dict(campaign_id='agentroute-v12-completion-20261031',frozen=True,frozen_unix_s=time.time(),model='Qwen/Qwen3-1.7B',protocol_sha256=digest(protocol),
      stage_inventory_reviewed=True,additional_stage_ids=STAGES,stage_inventory_note='Source-linked inventory in STAGE_INVENTORY.json',tasks=tasks,
      exact_counts=dict(counts),split_manifest=m,source_lock=source_lock)
    atomic_json(OUT/'campaign.lock.json',lock)
    store=Store(OUT/'coordinator.sqlite');store.initialize(tasks)
    keyed={(t['dataset'],t['split'],t['method'],t['seed'],t['question_id']):t for t in tasks if t['group'] in ['development','main_mmlu']}
    for item in reuse['generation']:
        task=keyed[tuple(item['identity'])];path=OUT/item['raw']['path'];row=read(path)
        ev=[dict(item['raw'],role=role) for role in ['raw','metrics','score','metering']]+[evidence('reuse_audit',OUT/'REUSE_AUDIT.json')]
        receipt={k:task[k] for k in ['task_id','group','fingerprint']};receipt.update(status='verified_reuse',attempt_id=item['attempt_id'],
          outcome='correct' if row['quality'] else 'incorrect' if row['format_ok'] else 'invalid_answer',evidence=ev,telemetry_reuse=item['telemetry_reuse'])
        store.accept(receipt,OUT,reuse=True)
    for item in reuse['fitting']:
        task=next(t for t in tasks if t['group']=='baseline_fitting' and t['method']==item['method'])
        validation=next(x for x in prep['verified_fits'] if x['method']==item['method']);validation_path=Path(validation['local_path'])
        assert hash_file(validation_path)==validation['sha256']
        ev=[evidence(role,OUT/'REUSE_AUDIT.json') for role in ['raw','metrics','metering','reuse_audit']]+[dict(item['checkpoint'],role='checkpoint'),evidence('restore_validation',validation_path)]
        store.accept(dict(task_id=task['task_id'],group=task['group'],fingerprint=task['fingerprint'],status='verified_reuse',outcome='trained',evidence=ev),OUT,reuse=True)
    with (OUT/'results_index.jsonl').open('w',encoding='utf8') as f:
        for receipt in store.accepted():f.write(json.dumps(receipt)+'\n')
    atomic_json(OUT/'STATUS.json',dict(status='READY_FOR_GPU_RECOVERY_TEST',**store.status()))
    print(json.dumps({'expected':dict(counts),'accepted':len(store.accepted()),'manifest_sha256':hash_file(OUT/'campaign.lock.json')}))
if __name__=='__main__':main()
