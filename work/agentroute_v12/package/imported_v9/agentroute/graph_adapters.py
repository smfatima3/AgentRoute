"""Controlled common-task adapters around the PINNED SafeSieve and CARD graph code.

Task nodes and endpoint clients are replaced; graph selection/pruning/GNN code is
loaded from upstream unchanged. These are adaptations, not native reproductions.
See docs/BASELINES.md before interpreting results.
"""
import asyncio, copy, importlib, json, sys, types
from pathlib import Path
import numpy as np
from .prompts import task_prompt

def package(name,path):
    if name not in sys.modules:
        m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m

def interface(name,**symbols):
    m=types.ModuleType(name);m.__dict__.update(symbols);sys.modules[name]=m

class GraphAdapter:
    def __init__(self,method,rt,upstream):
        import torch
        self.method=method;self.rt=rt;self.task=None;self.fit_log=[]
        self.name='SafeSieve' if method=='safesieve' else 'CARD'
        root=(Path(upstream)/method/self.name).resolve();name=self.name
        # Avoid importing upstream API clients/tools as an import-time side effect.
        # Modules below implement the explicit common-task interface.
        package(name,root)
        for sub in ['graph','agents','prompt','llm','dynamic','utils','gnn']:
            package(name+'.'+sub,root/sub)
        Node=importlib.import_module(name+'.graph.node').Node
        owner=self
        class CommonNode(Node):
            def __init__(node,role='Aggregator',domain='',llm_name='',**kwargs):
                super().__init__(id=None,agent_name='Common',domain=domain,llm_name=llm_name)
                node.role=role;node.external_tool_type=None;node.external_tool=None;node.external_source=None
            def _process_inputs(node,raw_inputs,spatial_info,temporal_info,**kwargs):
                return raw_inputs
            def _execute(node,input,spatial_info,temporal_info,**kwargs):
                peers=[];own=[]
                for kind,info in [('spatial',spatial_info),('temporal',temporal_info)]:
                    for sender,data in info.items():
                        text=data['output']
                        if sender==node.id:own.append(text);continue
                        peer_kind='aggregate' if node.role=='Aggregator' else 'peer'
                        owner.rt.deliver(sender,node.id,text,peer_kind)
                        peers.append({'sender':sender,'role':data['role'],'text':text,'edge_type':kind})
                return owner.rt.call(task_prompt(owner.task,own,peers),
                    'You are the '+node.role+'. Treat peer reports as fallible evidence. '+owner.task['output_instruction'],
                    'aggregate' if node.role=='Aggregator' else 'worker')
            async def _async_execute(node,input,spatial_info,temporal_info,**kwargs):
                return node._execute(input,spatial_info,temporal_info,**kwargs)
        class Registry:
            registry={'Common':CommonNode,'CommonFinal':CommonNode}
            @classmethod
            def get(cls,key,**kwargs):return cls.registry[key](**kwargs)
        interface(name+'.agents.agent_registry',AgentRegistry=Registry)
        if method=='card':
            roles=rt.config['roles'][:rt.config['agents']]
            class Prompt:
                def get_role_connection(self):
                    # A complete role-feature graph collapses homogeneous GCN
                    # representations before CARD learns its communication graph.
                    # This explicit common-role prior is separate from the learned
                    # communication DAG; it does not restrict candidate message edges.
                    if rt.config['card']['role_feature_prior']!='ordered_chain':
                        raise ValueError('Unknown CARD role-feature prior')
                    return list(zip(roles[:-1],roles[1:]))
                def get_description(self,role):return 'You are the '+role+'. Solve and check the supplied task.'
            class Prompts:
                @staticmethod
                def get(domain):return Prompt()
            interface(name+'.prompt.prompt_set_registry',PromptSetRegistry=Prompts)
            interface(name+'.llm.profile_embedding',get_sentence_embedding=lambda text:owner.rt.embedding.encode(text,show_progress_bar=False))
            class ModelInfo:
                def get_llm_feature_information(self,model):
                    return f'{rt.model}: a Qwen3 dense language model used in non-thinking mode without external tools.'
            interface(name+'.dynamic.llm_information',Dyllm=ModelInfo)
            class NoTools:
                @staticmethod
                def get(name):raise RuntimeError('External tools are excluded from this controlled comparison')
            interface(name+'.dynamic.dytool_registry',ToolRegistry=NoTools)
        Graph=importlib.import_module(name+'.graph.graph').Graph
        n=rt.config['agents'];kwargs=[{'role':r,'llm_name':rt.model} for r in rt.config['roles'][:n]]
        params=dict(domain='common',llm_name=rt.model,agent_names=['Common']*n,decision_method='CommonFinal',
            optimized_spatial=True,optimized_temporal=False,initial_spatial_probability=0.5,
            fixed_temporal_masks=[[0]*n for _ in range(n)])
        if method=='card':params.update(node_kwargs={'common':kwargs},allow_random_combination=False)
        else:params.update(node_kwargs=kwargs)
        self.graph=Graph(**params)
        if method=='safesieve':
            # edge_matrix imports prompt_similarity although it does not use it.
            interface(name+'.graph.prompt_similarity',compute_prompt_similarity_matrix=lambda p:None)
            self.EdgeManager=importlib.import_module(name+'.graph.edge_matrix').EdgeMatrixManager
            self.threshold=importlib.import_module(name+'.graph.pruning_algorithm').dynamic_threshold
            self.graph.spatial_logits=torch.nn.Parameter(torch.full((n*n,),0.5),requires_grad=False)
        self.trained=False

    def predict(self,rt,task,training=False):
        import torch
        self.rt=rt;self.task=task
        if not training and not self.trained:raise RuntimeError('A fit checkpoint is required')
        kwargs={'fixed_group':'common'} if self.method=='card' else {}
        self.graph.clear_spatial_connection();self.graph.clear_temporal_connection()
        for node in self.graph.nodes.values():
            node.outputs=[];node.inputs=[];node.last_memory={'inputs':[],'outputs':[],'raw_inputs':[]}
        start_errors=len([r for r in rt.ledger if r.get('status')=='error'])
        answers,logp=asyncio.run(self.graph.arun({'task':task['question']},num_rounds=rt.config['rounds'],max_tries=1,**kwargs))
        if len([r for r in rt.ledger if r.get('status')=='error'])>start_errors:
            raise RuntimeError('Upstream node swallowed an inference failure')
        if not answers or not isinstance(answers[-1],str):raise RuntimeError('No baseline response')
        return answers[-1],logp

    def initialize_similarity(self,rt):
        """Offline role-pair scores from the study backbone, explicitly metered."""
        n=rt.config['agents'];sim=np.zeros((n,n),dtype=np.float32)
        roles=rt.config['roles'][:n]
        for i in range(n):
            for j in range(i+1,n):
                text=rt.call(json.dumps({'role_a':roles[i],'role_b':roles[j]}),
                    'Assess role complementarity for solving tasks. Return only a JSON object with key score and a numeric value between 0 and 10. Do not use angle brackets or placeholders.',
                    'offline_role_similarity',max_tokens=128)
                val=json.loads(text)['score']
                if not isinstance(val,(int,float)) or isinstance(val,bool) or not 0<=val<=10:raise ValueError('Invalid role similarity score')
                sim[i,j]=sim[j,i]=val/10
        self.edge_manager=self.EdgeManager(n,sim,alpha=rt.config['safesieve']['alpha'],beta=1.0)
        self.pruning_count=0

    def fit_batch(self,batch_results,iteration):
        """batch_results: (quality, log_probability, realized adjacency, edge penalty).
        CARD keeps per-query gradients; SafeSieve accumulates train-only evidence.
        """
        import torch
        c=self.rt.config
        if self.method=='card':
            loss=torch.stack([-logp*quality+c['card']['penalty']*penalty
                              for quality,logp,adj,penalty in batch_results]).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite CARD training loss')
            self.optimizer.zero_grad();loss.backward()
            params=[p for group in self.optimizer.param_groups for p in group['params']]
            gradients=[p.grad for p in params if p.grad is not None]
            if any(not torch.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite CARD gradients')
            norm=sum(float(g.detach().abs().sum()) for g in gradients)
            before=[p.detach().clone() for p in params]
            self.optimizer.step()
            changed=any(not torch.equal(old,p.detach()) for old,p in zip(before,params))
            self.training_updates+=int(changed)
            self.fit_log.append({'iteration':iteration,'gradient_l1':norm,'weights_changed':changed})
            return float(loss.detach())
        em=self.edge_manager
        for quality,logp,adj,penalty in batch_results:
            em.update_complementarity(adj,solved=quality==1)
        # This adapter deliberately omits the upstream driver's second count of
        # the unexecuted template graph and its random-answer replacement.
        sc=c['safesieve'];em.update_beta_weight(1+(iteration*c['fit']['batch_size']*sc['beta_growth_rate'])/100)
        score=em.get_edge_score();threshold=self.threshold(em.beta,sc['min_threshold'],sc['max_threshold'],sc['beta_growth_rate'])
        n=self.graph.num_nodes;active=(self.graph.spatial_masks.detach().numpy().reshape(n,n)>0)&~np.eye(n,dtype=bool)
        if (iteration+1)%sc['prune_every']==0 and iteration+1>=sc['start_pruning_batch'] and self.pruning_count<sc['max_pruning_times'] and np.any(score[active]<threshold):
            self.graph.update_masks_with_zero_extension(sc['pruning_rate'],similarity_matrix=score)
            self.pruning_count+=1
            ids=self.graph.detect_isolated_nodes(sc['min_agents'])
            if ids:
                old=list(self.graph.node_order);self.graph.remove_isolated_nodes(ids)
                remaining=[i for i,id in enumerate(old) if id not in ids]
                em.remove_nodes_and_update_matrix([old.index(id) for id in ids],remaining)
        return None

    def start_fit(self):
        import torch
        if self.method=='card':
            self.training_updates=0
            modules=[self.graph.gcn,self.graph.gcn_dynamic,self.graph.feature_fusion,self.graph.mlp]
            for m in modules:m.train()
            # Documented repair: train the MLP as well as the GNN/fusion modules.
            self.optimizer=torch.optim.Adam([p for m in modules for p in m.parameters()],lr=self.rt.config['fit']['lr'])

    def export(self):
        if self.method=='card':
            if self.training_updates==0:raise RuntimeError('CARD made no trainable weight updates. Inspect its role-feature prior and gradients; no checkpoint will be saved.')
            return {'method':self.method,'training_updates':self.training_updates,
                    'weights':{k:getattr(self.graph,k).state_dict() for k in ['gcn','gcn_dynamic','feature_fusion','mlp']}}
        g=self.graph
        return {'method':self.method,'roles':[g.nodes[x].role for x in g.node_order],
            'spatial_masks':g.spatial_masks.detach(),'spatial_logits':g.spatial_logits.detach(),
            'pruning_count':self.pruning_count}

    def restore(self,state):
        import torch
        if self.method=='card':
            self.training_updates=state['training_updates']
            if self.training_updates<=0:raise ValueError('Untrained CARD checkpoint')
            for k,v in state['weights'].items():getattr(self.graph,k).load_state_dict(v)
            for k in state['weights']:getattr(self.graph,k).eval()
        else:
            g=self.graph;keep=set(state['roles']);remove=[id for id,node in g.nodes.items() if node.role not in keep]
            if remove:g.remove_isolated_nodes(remove)
            if [g.nodes[x].role for x in g.node_order]!=state['roles']:raise ValueError('Role ordering mismatch')
            g.spatial_masks=torch.nn.Parameter(state['spatial_masks'],requires_grad=False)
            g.spatial_logits=torch.nn.Parameter(state['spatial_logits'],requires_grad=False)
        self.trained=True
