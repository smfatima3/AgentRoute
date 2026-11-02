"""Fresh-inference matched controls. All routing and local voting are gold-free.

No reported confidence or correctness is inferred from correlated agent votes.
Sparse controls match configured caps, not necessarily realized payload volume.
"""
import collections
import json
import random
import time
from .answer_keys import answer_key
from .evaluation import final_answer
from .prompts import task_prompt, output_instruction


def validate_evidence_config(c):
    if c['models']!=['Qwen/Qwen3-1.7B'] or c['datasets']!=['math','bbh']:
        raise ValueError('Evidence scope is pinned to 1.7B, MATH and BBH')
    n=c['agents']
    if type(n) is not int or not 2<=n<=32 or type(c['rounds']) is not int or not 1<=c['rounds']<=4:
        raise ValueError('Unsupported agent/round count')
    if len(c['roles'])<n or len(set(c['roles'][:n]))!=n:raise ValueError('Distinct role names required')
    if c['routing']['paid_llm_routing'] is not False:raise ValueError('Paid routing is disabled')
    if c['local_decision']['paid_aggregator'] is not False:raise ValueError('Default aggregator must remain local')
    if not c['seeds'] or len(set(c['seeds']))!=len(c['seeds']):raise ValueError('Distinct seeds required')
    if type(c['max_tokens']) is not int or c['max_tokens']<=0:raise ValueError('Positive token cap required')
    if c['noninferiority_margin'] is not None and not 0<c['noninferiority_margin']<1:
        raise ValueError('Noninferiority margin must be explicit or unset')
    for name,a in c['evidence_arms'].items():
        if name not in c['methods']+c['controls']:raise ValueError('Undeclared arm')
        if a['communication'] not in ['route','dense','none','random','fixed']:raise ValueError('Unknown communication')
        if a['aggregation'] not in ['vote','paid']:raise ValueError('Unknown aggregation')
        if a['aggregation']=='paid' and name not in c['controls']:raise ValueError('Paid aggregator is control-only')
        if type(a['early_stop']) is not bool:raise ValueError('Boolean stopping rule required')
        if a.get('k') is not None and (type(a['k']) is not int or not 0<=a['k']<n):raise ValueError('Invalid k')
        if a.get('payload_cap_bytes') is not None and a['payload_cap_bytes']<=0:raise ValueError('Invalid payload cap')


def task_metric(task):
    # Match public, fixed instructions; never consult labels or task metadata.
    for metric in ['math','mcq','numeric']:
        if task['output_instruction']==output_instruction(metric):return metric
    return 'exact'


def select_reports(own,candidates,metric,filter_dissent=True,group_answers=True):
    groups={}
    for r in candidates:
        key=answer_key(r['text'],metric)
        if filter_dissent and key is not None and key==answer_key(own,metric):continue
        group=('answer',key) if group_answers and key is not None else ('sender',r['sender'])
        groups.setdefault(group,[]).append(r)
    def rank(r):
        # Prefer evidence over a bare final, then deterministic shortest evidence.
        text=r['text'].strip();bare='\n' not in text
        return bare,len(text.encode('utf-8')),r['sender']
    return sorted([min(g,key=rank) for g in groups.values()],key=lambda r:r['sender'])


def choose_answer(initial,latest,metric):
    if [r['sender'] for r in initial]!=[r['sender'] for r in latest]:raise ValueError('Agent order changed')
    original=[answer_key(r['text'],metric) for r in initial]
    revised=[answer_key(r['text'],metric) for r in latest]
    effective=[new if new is not None else old for old,new in zip(original,revised)]
    counts=collections.Counter(a for a in effective if a is not None)
    trace={'initial_answers':original,'latest_answers':revised,'effective_votes':effective,
        'initial_support':dict(collections.Counter(a for a in original if a is not None)),
        'vote_counts':dict(counts),'invalid_revision_fallbacks':sum(n is None and o is not None for n,o in zip(revised,original)),
        'tie_break':'latest_plurality_then_initial_support_then_agent_order'}
    if not counts:return '',dict(trace,status='no_valid_vote',selected_answer=None,tied_answers=[])
    tied=[a for a,count in counts.items() if count==max(counts.values())]
    initial_counts=collections.Counter(original)
    chosen=min(tied,key=lambda a:(-initial_counts[a],effective.index(a)))
    i=effective.index(chosen);report=latest[i] if revised[i] is not None else initial[i]
    # Preserve the original expression/case in output, not a lossy normalized key.
    return 'FINAL: '+final_answer(report['text']),dict(trace,status='selected',selected_answer=chosen,
        selected_sender=report['sender'],tied_answers=sorted(tied))


def run_evidence(rt,task,method):
    import psutil
    from .runtime import event_seed
    a=rt.config['evidence_arms'][method];metric=task_metric(task)
    n=rt.config['agents'];roles=rt.config['roles'][:n];initial=[]
    compact=rt.config.get('study')=='AgentRoute-compact-v11'
    if compact:
        for i,role in enumerate(roles):rt.deliver('coordinator',i,json.dumps({'agent':i,'role':role}),'registry')
    for i,role in enumerate(roles):
        initial.append({'sender':i,'text':rt.call(task_prompt(task),
            'You are the '+role+'. '+task['output_instruction'],call_id=(0,i))})
    latest=initial;all_rounds=[initial];stopped=False
    for rnd in range(1,rt.config['rounds']):
        keys=[answer_key(r['text'],metric) for r in latest]
        if compact:
            for r,k in zip(latest,keys):rt.deliver(r['sender'],'coordinator',json.dumps({'round':rnd-1,'answer':k}),'stop_control')
        if a['early_stop'] and None not in keys and len(set(keys))==1:
            if compact:
                for i in range(n):rt.deliver('coordinator',i,json.dumps({'stop':True}),'stop_control')
            stopped=True;break
        if compact:
            for i in range(n):rt.deliver('coordinator',i,json.dumps({'stop':False}),'stop_control')
        summary=json.dumps({'previous_answers':[{'sender':r['sender'],'answer':k} for r,k in zip(latest,keys)],
            'note':'Agreement is not proof. Check the evidence; preserve justified dissent.'},ensure_ascii=False)
        current=[]
        for i,role in enumerate(roles):
            started=time.perf_counter();cpu=time.process_time();rss=psutil.Process().memory_info().rss
            candidates=[r for r in latest if r['sender']!=i];mode=a['communication']
            if mode=='route':
                selected=select_reports(latest[i]['text'],candidates,metric,
                    a.get('filter_dissent',True),a.get('group_answers',True))
            elif mode=='dense':selected=candidates
            elif mode=='none':selected=[]
            elif mode=='random':
                selected=random.Random(event_seed(rt.seed,rt.row_id,'edges',rnd,i)).sample(candidates,len(candidates))
            else:selected=sorted(candidates,key=lambda r:(r['sender']-i)%n)
            k=a.get('k')
            if k is not None:selected=selected[:k]
            bytecap=a.get('payload_cap_bytes');truncated=0
            if bytecap is not None:
                capped=[]
                for report in selected:
                    raw=report['text'].encode('utf-8');truncated+=len(raw)>bytecap
                    capped.append(dict(report,text=raw[:bytecap].decode('utf-8',errors='ignore')))
                selected=capped
            selected=sorted(selected,key=lambda r:r['sender'])
            rt.routes.append({'recipient':i,'round':rnd,'candidates':len(candidates),'kept':len(selected),
                'budget':k,'payload_cap_bytes':bytecap,'truncated_reports':truncated,'policy':mode,
                'selected_senders':[r['sender'] for r in selected],
                'wall_s':time.perf_counter()-started,'cpu_s':time.process_time()-cpu,
                'process_rss_before_bytes':rss,'process_rss_after_bytes':psutil.Process().memory_info().rss,
                'memory_scope':'whole client process, not router-attributable peak',
                'fallback':False,'llm_candidates':0})
            for r in selected:rt.deliver(r['sender'],i,r['text'])
            prompt=task_prompt(task,[latest[i]['text']],selected)
            # A genuine no-communication arm receives neither reports nor vote summaries.
            if mode!='none' and a.get('vote_summary',True):
                rt.deliver('vote-summary',i,summary,'control');prompt+='\n\nPrevious answer record:\n'+summary
            current.append({'sender':i,'text':rt.call(prompt,'You are the '+role+
                '. Recheck the original problem independently. Change your answer only when the evidence warrants it; '
                'popularity alone is not evidence. '+task['output_instruction'],call_id=(rnd,i))})
        latest=current;all_rounds.append(current)
    output,decision=choose_answer(initial,latest,metric)
    if compact and a['aggregation']=='vote':
        for r in latest:rt.deliver(r['sender'],'selector',r['text'],'aggregate')
    if a['aggregation']=='paid':
        for r in latest:rt.deliver(r['sender'],'aggregator',r['text'],'aggregate')
        output=rt.call(task_prompt(task,peer_reports=latest),
            'Solve the original question. Check reports for errors and resolve disagreements from evidence. '
            'Do not copy a report merely because several agents repeat it. '+task['output_instruction'],
            'aggregate',call_id='aggregate')
    rt.local_decision=dict(decision,version='v10-generalized-local-vote',arm=method,metric=metric,
        stop_reason='unanimity' if stopped else 'round_limit',worker_rounds=len(all_rounds),
        initial_reports=initial,latest_reports=latest,all_round_reports=all_rounds,
        paid_router_calls=0,paid_aggregator_calls=int(a['aggregation']=='paid'),
        selector_output=output if a['aggregation']=='vote' else None,
        sparse_budget_note='Configured edge and per-message payload caps; realized usage is not forced equal')
    return output
