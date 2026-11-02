"""Local MCQ communication and answer selection. No labels, embeddings or LLM router.

Agreement is an operational stopping rule, not a calibrated confidence estimate.
The historical CLI name `cascade` identifies this explicit v9 variant via config
and source hashes. Its output must never be mixed with v8 cascade observations.
"""
import collections
import json
import re
import time
from .evaluation import final_answer
from .prompts import task_prompt


def answer(text):
    parsed=final_answer(text)
    match=re.fullmatch(r'\(?([A-D])\)?[.]?',parsed,re.I) if parsed is not None else None
    return match.group(1).upper() if match else None


def select_reports(own_text,candidates):
    """Preserve one representative of every dissenting answer; never cap at k=2.

    Prefer the shortest report with evidence within a same-answer group. Full
    original reports and every initial vote remain in the saved trace. This
    compression can lose distinct same-answer arguments; it is not lossless.
    Unparsed reports are not discarded as irrelevant merely for being unparsed.
    """
    own=answer(own_text);groups={};seen=set()
    for report in candidates:
        text=report['text'];key=answer(text)
        if key is not None and key==own:continue
        if text in seen:continue
        seen.add(text)
        group=('answer',key) if key is not None else ('unparsed',report['sender'])
        groups.setdefault(group,[]).append(report)
    selected=[]
    for group in groups.values():
        def rank(report):
            text=report['text'].strip()
            bare=bool(re.fullmatch(r'(?:(?:FINAL(?: ANSWER)?|ANSWER)\s*:\s*)?\(?[A-D]\)?[.]?',text,re.I))
            return (bare,len(text),report['sender'])
        selected.append(min(group,key=rank))
    return sorted(selected,key=lambda r:r['sender'])


def choose_answer(initial,latest):
    """One vote per agent. An unparseable revision falls back to that agent's
    original valid answer. Ties use initial support, then fixed agent order.
    No confidence is inferred from correlated votes.
    """
    original=[answer(r['text']) for r in initial]
    revised=[answer(r['text']) for r in latest]
    effective=[new if new is not None else old for old,new in zip(original,revised)]
    counts=collections.Counter(a for a in effective if a is not None)
    trace={'initial_answers':original,'latest_answers':revised,'effective_votes':effective,
           'initial_support':dict(collections.Counter(a for a in original if a is not None)),
           'vote_counts':dict(counts),'invalid_revision_fallbacks':sum(new is None and old is not None for old,new in zip(original,revised)),
           'tie_break':'latest_plurality_then_initial_support_then_agent_order'}
    if not counts:return '',dict(trace,status='no_valid_vote',selected_answer=None,tied_answers=[])
    tied=[a for a,count in counts.items() if count==max(counts.values())]
    initial_counts=collections.Counter(original)
    chosen=min(tied,key=lambda a:(-initial_counts[a],effective.index(a)))
    return 'FINAL: '+chosen,dict(trace,status='selected',selected_answer=chosen,tied_answers=sorted(tied))


def run_local(rt,task):
    if 'multiple-choice question' not in task['output_instruction']:
        raise ValueError('v9 local vote is validated structurally for MCQ only; other task policies are not implemented')
    config=rt.config
    if config['rounds']!=2 or config['agents']!=4:
        raise ValueError('v9 local-vote protocol requires four agents and at most two worker rounds')
    roles=config['roles'][:4];initial=[]
    for i,role in enumerate(roles):
        text=rt.call(task_prompt(task),'You are the '+role+'. '+task['output_instruction'],call_id=(0,i))
        initial.append({'sender':i,'text':text})
    votes=[answer(r['text']) for r in initial]
    if config['local_decision']['unanimous_early_stop'] and None not in votes and len(set(votes))==1:
        output,decision=choose_answer(initial,initial)
        rt.local_decision=dict(decision,version='v9-local-vote',stop_reason='initial_unanimity',worker_rounds=1,
                               initial_reports=initial,latest_reports=initial,paid_router_calls=0,paid_aggregator_calls=0)
        return output
    latest=[]
    vote_summary=json.dumps({'initial_answers':[{'sender':r['sender'],'answer':a} for r,a in zip(initial,votes)],
                             'note':'Agreement is not proof. Check the evidence; preserve justified dissent.'},ensure_ascii=False)
    for i,role in enumerate(roles):
        start=time.perf_counter();candidates=[r for r in initial if r['sender']!=i]
        selected=select_reports(initial[i]['text'],candidates)
        rt.routes.append({'recipient':i,'round':1,'candidates':len(candidates),'kept':len(selected),'budget':None,
                          'policy':'answer_group_disagreement','fallback':False,'llm_candidates':0,'wall_s':time.perf_counter()-start,
                          'lexical_s':0.,'embedding_s':0.,'llm_s':0.,'lexical_resolved':0,'embedding_resolved':0,
                          'selected_senders':[r['sender'] for r in selected]})
        for report in selected:rt.deliver(report['sender'],i,report['text'])
        rt.deliver('initial-vote-summary',i,vote_summary,'control')
        prompt=task_prompt(task,[initial[i]['text']],selected)+'\n\nInitial answer record:\n'+vote_summary
        text=rt.call(prompt,'You are the '+role+'. Recheck the original problem independently. '
                     'Change your answer only when the evidence warrants it; popularity alone is not evidence. '+task['output_instruction'],call_id=(1,i))
        latest.append({'sender':i,'text':text})
    output,decision=choose_answer(initial,latest)
    rt.local_decision=dict(decision,version='v9-local-vote',stop_reason='deterministic_final_vote',worker_rounds=2,
                           initial_reports=initial,latest_reports=latest,paid_router_calls=0,paid_aggregator_calls=0)
    return output
