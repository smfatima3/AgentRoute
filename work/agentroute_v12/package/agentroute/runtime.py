"""Common task boundary, authoritative usage ledger, and unchanged cascade selection."""
import json, random, time, os, hashlib
from pathlib import Path
from .client import ChatClient
from .core import digest
from .routing import Router
from .prompts import output_instruction,task_prompt

def event_seed(seed,*parts): return int(digest([seed,*parts])[:8],16)%(2**31-1)

def public_task(row):
    # Allowlist; no metadata, gold answers, canonical solutions, or executable tests.
    return {'question':row['question'],'output_instruction':output_instruction(row['metric'])}

class MeteredEmbedding:
    def __init__(self,encoder,events):self.encoder=encoder;self.events=events
    def encode(self,texts,*args,**kwargs):
        start=time.perf_counter()
        try:return self.encoder.encode(texts,*args,**kwargs)
        finally:self.events.append({'texts':1 if isinstance(texts,str) else len(texts),'wall_s':time.perf_counter()-start})
    def __getattr__(self,key):return getattr(self.encoder,key)

class Runtime:
    def __init__(self,config,model,revision,base_url,tokenizer,embedding,seed,row_id):
        self.config=config; self.model=model;self.seed=seed;self.row_id=row_id
        self.tokenizer=tokenizer;self.embedding_events=[];self.embedding=MeteredEmbedding(embedding,self.embedding_events)
        self.ledger=[];self.deliveries=[];self.routes=[];self.raw_calls=[];self.errors=[];self.local_decision=None
        self.client=ChatClient({'model':model,'revision':revision,'base_url':base_url,
            'temperature':config['temperature'],'timeout_s':config['timeout_s'],
            'stream':config.get('stream',False),'prices':None,'api_key_env':config.get('api_key_env'),
            'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}})
        self.router=Router(config['routing'],self.client,self.embedding)
        self.router.tokenizer=tokenizer
        self.budget=None;self.attempt_id=None;self.phase=None;self.journal=None

    def attach_budget(self,budget,attempt_id,phase,journal):
        self.budget=budget;self.attempt_id=attempt_id;self.phase=phase;self.journal=Path(journal)
        self.journal.parent.mkdir(parents=True,exist_ok=True)

    def log_event(self,event):
        if self.journal:
            with self.journal.open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(unix_s=time.time(),**event),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

    def call(self,prompt,system='Solve the task carefully.',category='worker',max_tokens=None,call_id=None):
        index=len(self.ledger);key=f'{self.attempt_id}/{index}'
        if self.budget:self.budget.reserve_call(key,self.attempt_id,self.phase,category)
        self.log_event({'event':'call_started','call_key':key,'category':category,'system':system,'prompt':prompt,'call_id':call_id})
        try:
            text=self.client.call([{'role':'system','content':system},{'role':'user','content':prompt}],
                event_seed(self.seed,self.row_id,call_id if call_id is not None else index),
                max_tokens or self.config['max_tokens'],category,self.ledger)
        finally:
            if len(self.ledger)>index:
                self.log_event({'event':'call_finished','call_key':key,'record':self.ledger[-1]})
                if self.budget:self.budget.finish_call(key,self.ledger[-1])
        self.raw_calls.append({'category':category,'system':system,'prompt':prompt,'response':text})
        return text

    def deliver(self,sender,recipient,text,kind='peer'):
        envelope={'sender':str(sender),'recipient':str(recipient),'kind':kind,'text':text}
        serialized=json.dumps(envelope,ensure_ascii=False,separators=(',',':')).encode('utf-8')
        record={'sender':str(sender),'recipient':str(recipient),'kind':kind,
            'text':text,'payload_tokens':len(self.tokenizer.encode(text,add_special_tokens=False)),
            'payload_bytes':len(text.encode()),'serialized_bytes':len(serialized),
            'sha256':hashlib.sha256(serialized).hexdigest(),'logical_event_unix_s':time.time(),
            'after_model_call_count':len(self.ledger),'delivery_scope':'logical in-process'}
        self.deliveries.append(record);self.log_event({'event':'logical_delivery','record':record})

    def summary(self):
        complete=bool(self.ledger) and all(c.get('prompt_tokens') is not None and c.get('completion_tokens') is not None for c in self.ledger)
        peer=[d for d in self.deliveries if d['kind']=='peer']
        return {'calls':self.ledger,'raw_calls':self.raw_calls,'deliveries':self.deliveries,'routes':self.routes,
            'local_decision':self.local_decision,
            'embedding_events':self.embedding_events,'embedding_wall_s':sum(e['wall_s'] for e in self.embedding_events),
            'routing_wall_s':sum(r['wall_s'] for r in self.routes) if self.routes else None,
            'usage_complete':complete,'llm_calls':len(self.ledger),
            'total_online_tokens':sum(c['prompt_tokens']+c['completion_tokens'] for c in self.ledger) if complete else None,
            'peer_messages':len(peer),'peer_payload_tokens':sum(d['payload_tokens'] for d in peer),
            'peer_payload_bytes':sum(d['payload_bytes'] for d in peer),
            'all_logical_deliveries':len(self.deliveries),'all_delivered_payload_tokens':sum(d['payload_tokens'] for d in self.deliveries),
            'candidate_messages':sum(r['candidates'] for r in self.routes) if self.routes else None,
            'hard_budget_violations':sum(r['kept']>r['budget'] for r in self.routes if r.get('budget') is not None),
            'actual_network_bytes':None,'network_note':'In-process semantic-policy study; physical transport measured separately.'}

def run_cascade(rt,task,method):
    c=rt.config
    if c.get('study') in ['AgentRoute-v10-evidence','AgentRoute-compact-v11'] and method in c['evidence_arms']:
        from .evidence_policy import run_evidence
        return run_evidence(rt,task,method)
    if method=='cascade':
        from .local_decision import run_local
        return run_local(rt,task)
    if method=='single':return rt.call(task_prompt(task),call_id='single')
    prior=[]
    for rnd in range(c['rounds']):
        current=[]
        for i,role in enumerate(c['roles'][:c['agents']]):
            candidates=[m for m in prior if m['sender']!=i]
            seed=event_seed(rt.seed,rt.row_id,rnd,i)
            selected,route=rt.router.select(method,task['question'],role,candidates,seed,random.Random(seed),rt.ledger)
            route.update(recipient=i,round=rnd);rt.routes.append(route)
            for m in selected:rt.deliver(m['sender'],i,m['text'])
            prompt=task_prompt(task,[m['text'] for m in prior if m['sender']==i],selected)
            text=rt.call(prompt,'You are the '+role+'. Treat peer reports as fallible evidence. '+task['output_instruction'],call_id=(rnd,i))
            current.append({'sender':i,'text':text})
        prior=current
    for m in prior:rt.deliver(m['sender'],'aggregator',m['text'],'aggregate')
    return rt.call(task_prompt(task,peer_reports=prior),
        'Solve the original question. Check the reports for errors and resolve disagreements from evidence. '
        'Do not copy a report merely because several agents repeat it. '+task['output_instruction'],
        'aggregate',call_id='aggregate')
