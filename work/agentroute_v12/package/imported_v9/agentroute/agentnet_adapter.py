"""AgentNet upstream control flow and memories with a common task/LLM interface.

The referenced implementation is Yang et al. (NeurIPS 2025). Native parity is
still required. This module never calls the upstream .evaluate() feedback loop.
"""
import copy, importlib, json, sys, types
from pathlib import Path

class AgentNetAdapter:
    def __init__(self,rt,upstream,encoder,fit_dir):
        root=(Path(upstream)/'agentnet/AgentNet_Code').resolve();sys.path.insert(0,str(root))
        self.rt=rt;self.last_chain=None;self.last_agent=None;self.task=None
        # The caller provides the pinned, original BGE FlagModel; this boundary
        # avoids an unpinned import-time download in upstream src/pool.py.
        owner=self
        class MeteredBGE:
            def encode(self,texts,*args,**kwargs):
                import time
                start=time.perf_counter()
                try:return encoder.encode(texts,*args,**kwargs)
                finally:
                    if hasattr(owner.rt,'embedding_events'):
                        owner.rt.embedding_events.append({'encoder':'BGE-large','texts':1 if isinstance(texts,str) else len(texts),'wall_s':time.perf_counter()-start})
        flag=types.ModuleType('FlagEmbedding');flag.FlagModel=lambda *a,**k:MeteredBGE()
        existing=sys.modules.get('FlagEmbedding');sys.modules['FlagEmbedding']=flag
        try:
            settings=importlib.import_module('config.setting')
            abilities=['reasoning','mathematical','language','knowledge','sequence','spatial','inference']
            settings.BASIC_PROBLEM_COMPLEXITY['common']=1.0
            settings.task_to_ability_map['common']=abilities
            agent_module=importlib.import_module('src.agent')
            experiment_module=importlib.import_module('src.experiment')
            self.Task=importlib.import_module('src.task').Task
        finally:
            if existing is None:sys.modules.pop('FlagEmbedding',None)
            else:sys.modules['FlagEmbedding']=existing
        agent_module.llm_model_response_map['agentroute_endpoint']=self.llm
        c=rt.config;ac=c['agentnet']
        cfg={'LLM':'agentroute_endpoint','abilities':{k:0.6 for k in abilities},
             'executor_memory_limit':40,'embedding_cache_limit':1000,'router_memory_limit':-1,
             'decay_rate':0.1,'decay_interval':10,'router_retrieval_num':ac['retrieval_num'],
             'executor_retrieval_num':ac['retrieval_num']}
        fit_dir=Path(fit_dir);fit_dir.mkdir(parents=True,exist_ok=True)
        file_args={k:str(fit_dir/(k+'.json')) for k in ['json_file_path_task_result','json_file_path_agent_info',
            'json_file_path_edge_weight','json_file_path_task_history','json_file_path_experience']}
        self.engine=experiment_module.Experiment(
            {'forward_path_max_length':ac['forward_path_max_length'],'max_execution_times':ac['max_execution_times'],'user_react':False},
            [{'id':i,'config':copy.deepcopy(cfg)} for i in range(c['agents'])],
            {'graph_type':'complete','global_router_experience':False},[],[],**file_args)
        for id,agent in self.engine.agent_graph.agents.items():
            original_execute=agent.execute_task
            def execute_task(task_chain,constraints,*args,decision='execute',_original=original_execute,**kwargs):
                # Repair upstream's execute-branch positional argument mismatch.
                return _original(task_chain,constraints,self.engine.split_constraints,
                                 self.engine.thought_constraints,user_react=False,decision=decision)
            agent.execute_task=execute_task
            original_decide=agent.decide_action
            def decide(task_chain,forward_times,_original=original_decide,_id=id):
                if self.last_agent is not None and self.last_agent!=_id:
                    t=task_chain.get_current_task()
                    self.rt.deliver(self.last_agent,_id,json.dumps({'question':t.major_problem,
                        'description':t.description,'progress':t.progress_text,'context':t.context}),'peer')
                self.last_agent=_id
                return _original(task_chain,forward_times)
            agent.decide_action=decide
        original_neighbors=self.engine.agent_graph.collect_neighbors_info
        def neighbors(agent_id,task):
            data=original_neighbors(agent_id,task)
            for sender,value in data.items():
                self.rt.deliver(sender,agent_id,json.dumps(value,default=str),'control')
            return data
        for agent in self.engine.agent_graph.agents.values():agent.set_collect_neighbors_info(neighbors)
        self.trained=False

    def llm(self,system_prompt,query_prompt):
        if len(self.rt.ledger)>=self.rt.config['agentnet']['max_calls_per_query']:
            raise RuntimeError('AgentNet per-query call ceiling exceeded')
        category='router' if 'DECISION' in system_prompt.upper() or 'ROUTER' in system_prompt.upper() else 'worker'
        return self.rt.call(query_prompt,system_prompt,category)

    def predict(self,rt,task,training=False):
        if not training and not self.trained:raise RuntimeError('AgentNet fitted memory checkpoint required')
        self.rt=rt;self.task=task;self.last_agent=None
        self.engine.constraints=task['output_instruction']
        self.engine.split_constraints='Solve the assigned subproblem. '+task['output_instruction']
        self.engine.thought_constraints='Reason about the task without external tools.'
        text=task['question']+'\n'+task['output_instruction']
        # Metadata-free common category; official BBH categories are retained
        # only in evaluator metadata and never used to route the held-out query.
        t=self.Task(task_id=rt.row_id,task_type='common',complexity=1.0,
                    major_problem=text,description=text,correct_answer='',test_cases=[])
        self.last_chain=self.engine.solve_a_single_task(t)
        return self.last_chain.get_final_result()

    def learn(self,quality):
        if self.last_chain is None:raise RuntimeError('No training task')
        # Called by training runner only, with the quality of a TRAIN item.
        self.engine.update_agent_graph(self.last_chain,{'success':bool(quality),
            'task_chain_id':self.last_chain.task_chain_id})

    def export(self):
        graph=self.engine.agent_graph
        fields=['abilities','success_rate','total_task_num','successful_task_num',
                'task_correlations','decay_count_down','current_load']
        agents={}
        for id,a in graph.agents.items():
            agents[id]={k:copy.deepcopy(getattr(a,k)) for k in fields}
            agents[id]['router_pool']=copy.deepcopy(a.router_module.router_pool)
            agents[id]['executor_pool']=copy.deepcopy(a.executor_module.executor_pool)
        return {'agents':agents,'edge_weight':copy.deepcopy(graph.edge_weight),
            'edge_success_rate':copy.deepcopy(graph.edge_success_rate),
            'agent_neighbor_dict':copy.deepcopy(graph.agent_neighbor_dict)}

    def restore(self,state):
        graph=self.engine.agent_graph
        for id,data in state['agents'].items():
            a=graph.agents[id]
            for k,v in data.items():
                if k=='router_pool':a.router_module.router_pool=copy.deepcopy(v)
                elif k=='executor_pool':a.executor_module.executor_pool=copy.deepcopy(v)
                else:setattr(a,k,copy.deepcopy(v))
        for k in ['edge_weight','edge_success_rate','agent_neighbor_dict']:setattr(graph,k,copy.deepcopy(state[k]))
        self.trained=True

def load_encoder(revision):
    from huggingface_hub import snapshot_download
    from FlagEmbedding import FlagModel
    path=snapshot_download('BAAI/bge-large-en-v1.5',revision=revision,
        allow_patterns=['*.json','*.txt','*.safetensors','*.bin','1_Pooling/*'])
    return FlagModel(path,query_instruction_for_retrieval='Represent this sentence for searching relevant passages:',
                     use_fp16=False,devices=['cpu'])
