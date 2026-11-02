"""Pinned upstream SelfOrg, with no reference input and metered task-backbone calls."""
import importlib, inspect, json, os, sys, types
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def working_directory(path):
    old=os.getcwd();os.chdir(path)
    try:yield
    finally:os.chdir(old)

def run(rt,task,upstream):
    # One adapter per subprocess avoids upstream top-level package-name collisions.
    root=Path(upstream)/'selforg';sys.path.insert(0,str(root.resolve()))
    with working_directory(root):
        module=importlib.import_module('methods.selforg.selforg_main')
        general={'model_api_config':{},'model_name':rt.model,'model_temperature':rt.config['temperature'],
                 'model_max_tokens':rt.config['max_tokens'],'model_timeout':rt.config['timeout_s']}
        agent=module.SelfOrg_Main(general)
        agent.num_agents=rt.config['agents'];agent.roles=rt.config['roles'][:agent.num_agents]
        for role in agent.roles:agent.role_map[role]='You are the '+role+'. Solve and check the task.'
        for key,value in rt.config['selforg'].items():setattr(agent,key,value)
        agent.temperature=rt.config['temperature']
        module.set_seed(rt.seed)
        agent.enforce_dag=True;agent.aggregate_mode='weighted'
        # Exact pinned MiniLM object; avoids unpinned download in upstream constructor.
        agent._ensure_emb_model=types.MethodType(lambda self:setattr(self,'_emb_model',rt.embedding),agent)
        agent._emb_model=rt.embedding
        def embed_many(self,texts):
            return rt.embedding.encode(texts,normalize_embeddings=True,show_progress_bar=False).tolist()
        agent._embed_many=types.MethodType(embed_many,agent)
        def call(self,prompt=None,system_prompt=None,messages=None,model_name=None,temperature=None):
            if messages is not None: raise ValueError('Unexpected SelfOrg message API')
            return rt.call(prompt,(system_prompt or '')+'\n'+task['output_instruction'])
        agent.call_llm=types.MethodType(call,agent)
        old_update=agent._update_prompt
        def update(self,query,own_response,incoming):
            frame=inspect.currentframe().f_back
            recipient=frame.f_locals.get('i')
            if frame.f_code.co_name!='_propagate_on_dag' or not isinstance(recipient,int):
                raise RuntimeError('Pinned upstream recipient instrumentation no longer matches')
            for sender,text in incoming:rt.deliver(sender,recipient,text)
            return old_update(query,own_response,incoming)
        agent._update_prompt=types.MethodType(update,agent)
        result=agent.inference({'query':task['question']+'\n'+task['output_instruction']})
        return result['response']
