"""Observation and prespecified control overlays; original upstream files unchanged."""
import contextlib,copy,hashlib,json,os,sys,time,types
from pathlib import Path

class Spans:
    def __init__(self):self.events=[];self.frames={};self.stack=[]
    def profile(self,frame,event,arg):
        if event not in ('call','return'):return
        path=frame.f_code.co_filename.replace('\\','/');name=frame.f_code.co_name
        tracked=('/upstream/' in path or path.endswith('/agentroute/evidence_policy.py') or
          path.endswith('/agentroute/runtime.py') and name in ['call','deliver'] or
          path.endswith('/agentroute/agentnet_adapter.py') and name in ['encode','llm'])
        if not tracked:return
        if event=='call':
            rec={'function':name,'source':path.split('/project/')[-1],'start_monotonic_s':time.perf_counter(),
                 'cpu_start_s':time.process_time(),'children_wall_s':0.,'children_cpu_s':0.}
            self.frames[id(frame)]=rec;self.stack.append(rec)
        else:
            rec=self.frames.pop(id(frame),None)
            if rec is None:return
            wall=time.perf_counter()-rec['start_monotonic_s'];cpu=time.process_time()-rec.pop('cpu_start_s')
            rec.update(wall_s=wall,cpu_s=cpu,exclusive_wall_s=max(0.,wall-rec['children_wall_s']),exclusive_cpu_s=max(0.,cpu-rec['children_cpu_s']))
            self.events.append(rec)
            if self.stack and self.stack[-1] is rec:self.stack.pop()
            if self.stack:self.stack[-1]['children_wall_s']+=wall;self.stack[-1]['children_cpu_s']+=cpu
    def __enter__(self):sys.setprofile(self.profile);return self
    def __exit__(self,*exc):sys.setprofile(None)

@contextlib.contextmanager
def policy_overlay(method):
    import agentroute.evidence_policy as policy
    old=policy.select_reports
    if method=='remove_representative_rank':
        def first(own,candidates,metric,filter_dissent=True,group_answers=True):
            groups={}
            for report in candidates:
                key=policy.answer_key(report['text'],metric)
                if filter_dissent and key is not None and key==policy.answer_key(own,metric):continue
                group=('answer',key) if group_answers and key is not None else ('sender',report['sender'])
                groups.setdefault(group,report)
            return sorted(groups.values(),key=lambda r:r['sender'])
        policy.select_reports=first
    try:yield
    finally:policy.select_reports=old

def selforg_original(rt,task,upstream):
    import importlib
    sys.path.insert(0,str(Path(upstream)/'selforg'))
    mod=importlib.import_module('methods.selforg.selforg_main');old=mod.SelfOrg_Main._check_for_consensus
    mod.SelfOrg_Main._check_for_consensus=lambda self,sims:False
    try:
        from agentroute.selforg_adapter import run
        return run(rt,task,upstream)
    finally:mod.SelfOrg_Main._check_for_consensus=old

def resolved_config(base,task):
    c=copy.deepcopy(base);a=c['evidence_arms'];a['cascade']=copy.deepcopy(a['original'])
    a['route_fixed']=dict(a['cascade'],early_stop=False)
    a['toggle_filter']=dict(a['cascade'],filter_dissent=False)
    a['route_paid']=dict(a['route_fixed'],aggregation='paid')
    a['dense_paid']=dict(a['dense_fixed'],aggregation='paid')
    a['remove_answer_grouping']=dict(a['cascade'],group_answers=False)
    a['remove_representative_rank']=dict(a['cascade'])
    a['remove_vote_summary']=dict(a['cascade'],vote_summary=False)
    c['controls']=list(dict.fromkeys(c['controls']+['route_paid','dense_paid','remove_answer_grouping','remove_representative_rank','remove_vote_summary']))
    if task.get('agents',4)!=4:
        c['agents']=task['agents'];c['roles']+=['Algebra checker','Logic checker','Alternative solver','Evidence reviewer']
        for arm in a.values():arm['k']=c['agents']-1
    return c

def self_consistency(rt,task,budget):
    from agentroute.prompts import task_prompt
    from agentroute.evidence_policy import choose_answer,task_metric
    messages=[{'role':'system','content':'Solve the task carefully.'},{'role':'user','content':task_prompt(task)}]
    tokens=rt.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    reserve=len(tokens)+rt.config['max_tokens'];used=0;samples=[]
    while used+reserve<=budget:
        text=rt.call(messages[1]['content'],messages[0]['content'],call_id=('self_consistency',len(samples)))
        call=rt.ledger[-1];spent=call['prompt_tokens']+call['completion_tokens']
        if spent>reserve:raise RuntimeError('Self-consistency reservation mismatch')
        used+=spent;samples.append({'sender':len(samples),'text':text})
    output,decision=choose_answer(samples,samples,task_metric(task)) if samples else ('',{'status':'budget_precluded_sample'})
    rt.local_decision=dict(decision,token_budget=budget,sample_reservation=reserve,actual_tokens=used,unused_budget=budget-used,
       independent_samples=len(samples),selector='gold-free plurality then earliest sample',budget_source='frozen development mean, dataset mapping in protocol')
    return output
