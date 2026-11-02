"""Question-level paired summaries from real complete logs; no placeholder rows."""
import argparse,collections,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentroute.core import write_json
from agentroute.jobs import ROOT
from agentroute.statistics import quality_interval,paired_token_saving

def interval(values,seed=97):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    means=np.array([x[rng.integers(0,len(x),len(x))].mean() for _ in range(10000)])
    return [float(v) for v in np.quantile(means,[.025,.975])]

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results')
    p.add_argument('--output',type=Path,default=ROOT/'analysis.json');a=p.parse_args()
    groups=collections.defaultdict(dict);expected={};progress=[];configs=set();sources=set()
    for path in a.results.glob('*/job.json'):
        job=json.loads(path.read_text(encoding='utf-8'));configs.add(job['config_hash']);sources.add(json.dumps(job['source_hashes'],sort_keys=True))
        key=(job['model'],job['dataset'],job['split'],job['method'])
        expected.setdefault(key,set()).update((id,job['seed']) for id in job['ids'])
        for record in path.parent.glob('*.json'):
            if record.name=='job.json':continue
            row=json.loads(record.read_text(encoding='utf-8'));pair=(row['id'],row['seed'])
            if pair in groups[key]:raise ValueError('Duplicate experiment cell; select one immutable run set')
            groups[key][pair]=row
    if len(configs)>1 or len(sources)>1:raise ValueError('Mixed protocols/code versions; analyze separately')
    summaries=[];comparisons=[]
    for key,rows in groups.items():
        complete=set(rows)==expected[key];values=list(rows.values())
        record={'model':key[0],'dataset':key[1],'split':key[2],'method':key[3],
            'attempted':len(values),'expected':len(expected[key]),'complete':complete,
            'failures':sum(v['status']!='ok' for v in values),'claim_status':'exploratory_pending_baseline_parity'}
        if complete:
            record.update(quality_interval(values))
            decisions=[v['local_decision'] for v in values if v.get('local_decision')]
            if decisions:
                record['local_decision_diagnostics']={'questions':len(decisions),
                    'initial_unanimity_stops':sum(d['stop_reason']=='initial_unanimity' for d in decisions),
                    'final_ties':sum(len(d['tied_answers'])>1 for d in decisions),
                    'no_valid_votes':sum(d['status']=='no_valid_vote' for d in decisions),
                    'initial_invalid_answers':sum(a is None for d in decisions for a in d['initial_answers']),
                    'latest_invalid_answers':sum(a is None for d in decisions for a in d['latest_answers']),
                    'invalid_revision_fallbacks':sum(d['invalid_revision_fallbacks'] for d in decisions)}
            record['accuracy']=float(np.mean([v['quality'] for v in values]))
            record['format_error_rate']=float(np.mean([not v['format_ok'] for v in values]))
            record['literal_placeholder_rate']=float(np.mean([v.get('answer_health',{}).get('literal_placeholder',False) for v in values]))
            record['truncated_calls']=sum(v.get('answer_health',{}).get('truncated_calls',0) for v in values)
            record['router_fallbacks']=sum(sum(bool(r.get('fallback')) for r in v.get('routes',[])) for v in values)
            for metric in ['peer_messages','peer_payload_tokens','all_delivered_payload_tokens','llm_calls']:
                record[metric]=float(np.mean([v[metric] for v in values]))
            record['total_online_tokens']=float(np.mean([v['total_online_tokens'] for v in values])) if all(v['usage_complete'] for v in values) else None
            latency=[v.get('inference_wall_s',v['wall_s']) for v in values]
            record['latency_p50_p95_p99_s']=[float(x) for x in np.quantile(latency,[.5,.95,.99])]
        summaries.append(record)
        base_key=(*key[:3],'cascade')
        if key[3]=='cascade' or base_key not in groups or not complete:continue
        route=groups[base_key]
        if set(route)!=set(rows) or set(route)!=expected.get(base_key,set()):continue
        ids=sorted({id for id,seed in rows});delta=[];reductions=collections.defaultdict(list)
        for id in ids:
            pairs=[pair for pair in rows if pair[0]==id]
            delta.append(np.mean([route[p]['quality']-rows[p]['quality'] for p in pairs]))
            for metric in ['peer_messages','peer_payload_tokens','total_online_tokens']:
                if all(rows[p].get(metric) is not None and route[p].get(metric) is not None for p in pairs):
                    reductions[metric].append(float(np.mean([rows[p][metric]-route[p][metric] for p in pairs])))
        lo,hi=interval(delta)
        cmp={'model':key[0],'dataset':key[1],'split':key[2],'baseline':key[3],'questions':len(ids),
            'quality_delta_agentroute_minus_baseline':float(np.mean(delta)),'paired_95ci':[lo,hi],
            'noninferiority_margin':.02,'noninferiority_claim':'withheld_pending_dev_selection_native_parity_and_adequate_precision'}
        cmp.update(paired_token_saving(route,rows))
        for metric,values in reductions.items():
            if len(values)==len(ids):cmp['absolute_reduction_'+metric]={'mean':float(np.mean(values)),'paired_95ci':interval(values)}
        # Prespecified seed 17 only. Repetitions are never independent questions.
        win=sum(route[(id,17)]['quality']>rows[(id,17)]['quality'] for id in ids if (id,17) in rows)
        lose=sum(route[(id,17)]['quality']<rows[(id,17)]['quality'] for id in ids if (id,17) in rows)
        n=win+lose
        cmp.update(agentroute_only_correct=win,baseline_only_correct=lose,discordant_questions=n)
        cmp['mcnemar_exact_p_seed17']=min(1.,2*sum(math.comb(n,k) for k in range(min(win,lose)+1))/2**n) if n else 1.
        comparisons.append(cmp)
    order=sorted(range(len(comparisons)),key=lambda i:comparisons[i]['mcnemar_exact_p_seed17']);previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,comparisons[i]['mcnemar_exact_p_seed17']*(len(order)-rank)))
        comparisons[i]['holm_p']=previous
    write_json(a.output,{'status':'measured' if groups else 'no_model_runs','summaries':summaries,'paired_comparisons':comparisons,
        'evaluated_protocol':{'config_hashes':sorted(configs),'source_hash_sets':[json.loads(s) for s in sorted(sources)]},
        'evidence_scope':{'component_ablations_performed':False,'causal_component_claims_supported':False,
                          'statistics_authenticate_raw_execution':False,
                          'provenance_requirements':['immutable raw attempts','matching IDs and seeds','model and source hashes','authoritative usage','execution/export records']},
        'notes':['Failures remain in denominators. Incomplete jobs have no accuracy estimate.',
                 'Statistical significance is not authenticity; provenance must be audited separately.',
                 'Non-significance does not establish equivalence. Development tuning prevents confirmatory claims on these same questions.',
                 'Per-comparison confidence intervals are not simultaneous family-wise intervals; Holm correction applies to accuracy tests.',
                 'Paired bootstrap averages seeds within questions. Holm correction spans emitted comparisons.',
                 'Concurrency-2 latency is shared-server latency; profile concurrency-1 separately.',
                 'HumanEval accuracy is per-seed pass@1, not pass@3.']})
    print(a.output)

if __name__=='__main__':main()
