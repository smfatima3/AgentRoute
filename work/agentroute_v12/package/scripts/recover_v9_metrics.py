"""Read-only expansion of v9 evidence. No new model observations are created."""
import argparse,collections,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentroute.core import file_hash,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    groups=collections.defaultdict(list);provenance={};expected={}
    for job_path in sorted((a.source/'live_raw/project/results').glob('*/job.json')):
        job=json.loads(job_path.read_text(encoding='utf-8'));method=job['method'];expected[method]=len(job['ids'])
        provenance[str(job_path.relative_to(a.source))]=file_hash(job_path)
        for path in sorted(job_path.parent.glob('*.json')):
            if path.name=='job.json':continue
            row=json.loads(path.read_text(encoding='utf-8'));groups[method].append(row)
            provenance[str(path.relative_to(a.source))]=file_hash(path)
    summaries={}
    for method,rows in sorted(groups.items()):
        assert len(rows)==expected[method] and len({(r['id'],r['seed']) for r in rows})==len(rows)
        assert all(r['usage_complete'] for r in rows)
        categories=collections.defaultdict(lambda:dict(calls=0,prompt_tokens=0,completion_tokens=0,client_wall_s=0.))
        deliveries=collections.defaultdict(lambda:dict(messages=0,utf8_payload_bytes=0,payload_tokens=0))
        for r in rows:
            for c in r['calls']:
                t=categories[c['category']];t['calls']+=1
                t['prompt_tokens']+=c['prompt_tokens'];t['completion_tokens']+=c['completion_tokens'];t['client_wall_s']+=c['wall_s']
            for d in r['deliveries']:
                assert len(d['text'].encode('utf-8'))==d['payload_bytes']
                t=deliveries[d['kind']];t['messages']+=1;t['utf8_payload_bytes']+=d['payload_bytes'];t['payload_tokens']+=d['payload_tokens']
        walls=[r.get('inference_wall_s',r['wall_s']) for r in rows]
        summaries[method]={'questions':len(rows),'correct':sum(r['quality'] for r in rows),
            'invalid_outputs':sum(not r['format_ok'] for r in rows),'failed_queries':sum(r['status']!='ok' for r in rows),
            'failed_calls':sum(c['status']!='ok' for r in rows for c in r['calls']),
            'truncated_calls':sum(c.get('finish_reason')=='length' for r in rows for c in r['calls']),
            'by_call_purpose':dict(categories),'logical_deliveries_by_purpose':dict(deliveries),
            'inference_latency_s':dict(zip(['p50','p95','p99'],map(float,np.quantile(walls,[.5,.95,.99])))),
            'router_wall_s_total':sum(route['wall_s'] for r in rows for route in r.get('routes',[])),
            'latency_scope':'Descriptive v9 A100 concurrency-2 shared-server observations; p99 from 50 is not a tail guarantee',
            'wire_bytes':None,'router_cpu_s':None,'queue_latency_s':None,'load_throughput':None,
            'slo_goodput':None,'gpu_utilization':None}
    stats=json.loads((a.source/'statistical_analysis.json').read_text())
    diagnostics=json.loads((a.source/'v9_answer_diagnostics.json').read_text())
    write_json(a.output/'v9_recovered_metrics.json',{'evidence_class':'reanalysis_of_immutable_v9',
        'source_files':provenance,'summaries':summaries,'paired_comparisons':stats['paired_comparisons'],
        'answer_survival':diagnostics['counts'],
        'limitations':['All 50 questions are development evidence. No new inference.',
        'Logical payload size is not measured wire traffic. No network-capacity claim.',
        'No matched-selector controls in v9. Routing-only attribution remains unmeasured.',
        'Raw execution provenance supports authenticity; p-values do not authenticate execution.']})
    lines=['# Recovered v9 metrics','',
        'Read-only reanalysis of 400 saved observations: 50 common MMLU development questions, seed 17. No new model run.',
        '', '| Method | Correct | Prompt tokens | Completion tokens | Worker / aggregate calls | Peer bytes | Control bytes | Median latency s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        cats=s['by_call_purpose'];d=s['logical_deliveries_by_purpose']
        lines.append(f"| {name} | {s['correct']:g}/50 | {sum(x['prompt_tokens'] for x in cats.values()):,} | {sum(x['completion_tokens'] for x in cats.values()):,} | {cats.get('worker',{}).get('calls',0)} / {cats.get('aggregate',{}).get('calls',0)} | {d.get('peer',{}).get('utf8_payload_bytes',0):,} | {d.get('control',{}).get('utf8_payload_bytes',0):,} | {s['inference_latency_s']['p50']:.2f} |")
    lines+=['','Tokens, calls and bytes are totals over 50 questions. Peer/control bytes are logical UTF-8 payload only; paid aggregation deliveries are separately retained in the JSON.',
        '', 'AgentRoute scored 33/50 (66%; Wilson 95% CI 52.2–77.6%). Its +4 percentage-point difference versus broadcast had a paired 95% bootstrap interval of −8 to +16 points. All seven Holm-adjusted accuracy p-values were 1.0. Neither superiority nor noninferiority was established.',
        '', 'The saved traces contain four questions where a correct initial worker answer disappeared during revision; one gained a correct answer. Eight wrong outputs stopped on unanimous wrong answers. Two final decisions failed to select a correct answer still available to the local selector. These counts motivate the new controls; they do not prove which change will improve accuracy.',
        '', 'Still unavailable from v9: physical wire traffic, open-loop throughput/SLO goodput, queue spans, GPU utilization, router-attributable memory, matched-selector routing effects, and confirmatory held-out evidence.']
    (a.output/'V9_RECOVERED_METRICS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'methods':len(summaries),'queries':sum(len(v) for v in groups.values()),'output':str(a.output)}))
if __name__=='__main__':main()
