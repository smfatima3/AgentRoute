import os
import math 
import random 
from collections import defaultdict 

from ..mas_base import MAS 

from ..utils import handle_retry_error

# from sentence_transformers import SentenceTransformer, models 
import torch 
import numpy as np
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_exponential, stop_after_attempt 

import threading

import heapq 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SelfOrg_Main(MAS):
    """
    Self-Organizing Multi-Agent System paper implementation. 
    
    """
    _GLOBAL_EMB_MODEL = None 
    _GLOBAL_EMB_LOCK = threading.Lock()
    
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name 
        super().__init__(general_config, method_config_name)
        
        # core knobs (sensible defaults)
        mc = self.method_config
        self.num_agents = mc.get("num_agents", 5) # 6 
        self.roles = mc.get("roles", ["Assistant"] * self.num_agents) 
        self.temperature = mc.get("temperature", 0.2) 
        self.random_seed = mc.get("random_seed", 101) 
        
        # similarity pruning 
        self.top_k = mc.get("top_k", 2) # to keep top-k most similar peers for each agent 
        self.sim_threshold = mc.get("sim_threshold", -1) # if >=0, also keep any sim >= threshold 
        
        # DAG propagation 
        self.max_rounds = mc.get("max_rounds", 2) # rounds of DAG propagation after graph forms 
        self.enforce_dag = mc.get("enforce_dag", True) 
        
        # aggregation 
        self.aggregate_mode = mc.get("aggregate_mode", "weighted") # weighted or single 
        self.embed_dim = mc.get("embedding_dim", 512) 
        
        # internals 
        set_seed(self.random_seed)
        
        # prompts 
        self.role_map = self._get_role_map() 
        
        # embedding model configs 
        self.emb_model_name = mc.get("emb_model", "sentence-transformers/all-MiniLM-L6-v2")
        self._emb_model = None 
        self._forward_lock = threading.Lock()
        
        # consensus based params
        self.consensus_min_sim = mc.get("consensus_min_sim", 0.95)
        self.consensus_range_eps = mc.get("consensus_range_eps", 0.05)
        self.reform = mc.get("reform", False)
    
    def inference(self, sample):
        query = sample["query"]
        reference = sample.get("reference", None)
        
        # 1) decentralized initialization 
        init_answers = [] 
        for i in range(self.num_agents):
            system_prompt = self.role_map.get(self.roles[i], self.role_map["Assistant"])
            prompt = self._init_prompt(query) 
            ans = self._call_llm(prompt=prompt, system_prompt=system_prompt, temperature=self.temperature) 
            init_answers.append(ans)
        
        # 2) embedding similarity evaluation 
        sims = self._pairwise_sims(init_answers)
        
        contributions = self._approx_shapley(init_answers, None) # reference 
        
        # 2) for each agent i, choose helpful peers by top-k (and threshold if set) 
        helpful = []
        for i in range(self.num_agents):
            scored = [] 
            for j in range(self.num_agents):
                if j == i:
                    continue
                base = sims[i][j] 
                # bias toward listening to high-contribution agents 
                adj = base * (1.0 + self.num_agents * (contributions[j] - contributions[i])) # with a multiplier of N [since the contributions are scaled by N] 
                scored.append((j, adj, base))
                    
            scored = [p for p in scored if p[2] >= self.sim_threshold]
                        
            scored.sort(key=lambda x: (x[1], contributions[x[0]]), reverse=True)
            keep = {j for (j, _, _) in scored[:self.top_k]}
            
            helpful.append(sorted(list(keep)))
        
        # 3) directed graph 
        edges = set() 
        edge_w = {}
        
        for i in range(self.num_agents):
            for j in helpful[i]:
                edges.add((j, i))
                adj = sims[i][j] * (1.0 + self.num_agents * (contributions[j] - contributions[i])) # with a multiplier of N [since the contributions are scaled by N] 
                edge_w[(j, i)] = max(0.0, adj)
        
        # 4) DAG enforcement
        if self.enforce_dag:
            edges, edge_w = self._dagify(edges, edge_w)
        
        # 5) contribution valuation 
        final_answers = self._propagate_on_dag(query, init_answers, edges, rounds=self.max_rounds, contributions=contributions)
        contributions = self._approx_shapley(final_answers, reference) # init_answers, final_answers
        
        # 6) aggregation 
        if self.aggregate_mode == "single": 
            # pick the agent with highest contribution 
            best_idx = max(range(self.num_agents), key=lambda i: contributions[i])
            response = final_answers[best_idx]
        else:
            final_embs = self._embed_many(final_answers) # [self._embed(ans) for ans in final_answers]
            weights = contributions
            if sum(weights) <= 1e-9:
                weights = [1.0 / self.num_agents] * self.num_agents 
            
            agg = self._weighted_centroid(final_embs, weights)
            nearest = max(range(self.num_agents), key=lambda i: self._cosine(agg, final_embs[i]))
            response = final_answers[nearest]
        
        return {"response": response} 
    
    # helper functions 
    def _init_prompt(self, query):
        instr = (
            "You will independently attempt the user's task first. Let's think step by step.\n" 
            "Be precise and complete.\n" # If the problem is math, compute carefully. 
        )
        
        return f"{instr}\n[Task]\n{query}\n" 
    
    def _update_prompt(self, query, own_response, incoming):
        block = "" 
        for nid, txt in incoming:
            block += f"\n[Peer {nid} answer]\n{txt}\n"
                
        return (
            "Update your answer by critically evaluating the peer answers below. "
            "They may contain errors - do not copy blindly.\n" 
            f"[Task]\n{query}\n"
            f"\n[Your previous answer]\n{own_response}\n"
            f"{block}\n"
            "Now provide your improved answer (Be sure to include the steps in the response):"
        )
    
    def _update_prompt_leader_agent(self, query, own_response):
        """
        Self-refinement only, no peer answers
        """
        return (
            "You are the current lead agent. No peer answers are available for this round.\n"
            "Review your previous answer and improve it if needed.\n"
            f"[Task]\n{query}\n"
            f"\n[Your previous answer]\n{own_response}\n"
            "Now provide your updated answer with concise reasoning:"
        )
    
    def _get_role_map(self):
        return {
            "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
            "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
            "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
            "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
            "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
            "Doctor": "You are a medical expert. You recommend evidence-based options and note caveats. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient's age, lifestyle and medical history when providing your recommendations.",
            "Programmer": "You are a programmer skilled in software design and debugging. You have experience in designing and developing computer software and hardware.",
            "Historian": "You are a historian. You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.",
        }
    
    def _call_llm(self, prompt, system_prompt, temperature):
        try:
            return self.call_llm(prompt=prompt, system_prompt=system_prompt, temperature=temperature)
        except TypeError:
            messages = [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": prompt}
            ]
            return self.call_llm(messages=messages)
    
    def _ensure_emb_model(self):
        if self._emb_model is not None:
            return 
        
        if SelfOrg_Main._GLOBAL_EMB_MODEL is not None:
            self._emb_model = SelfOrg_Main._GLOBAL_EMB_MODEL
            return
        
        with SelfOrg_Main._GLOBAL_EMB_LOCK:
            if SelfOrg_Main._GLOBAL_EMB_MODEL is None:
                if self.model_name in ['qwen2.5-1.5b-instruct', 'qwen2.5-7b-instruct', 'qwen2.5-14b-instruct']: 
                    SelfOrg_Main._GLOBAL_EMB_MODEL = SentenceTransformer(self.emb_model_name, trust_remote_code=True)
                else: # qwen-2.5-72b-instruct, llama-3.3-70b-instruct 
                    SelfOrg_Main._GLOBAL_EMB_MODEL = SentenceTransformer(self.emb_model_name, device="cpu", trust_remote_code=True) 
            
            self._emb_model = SelfOrg_Main._GLOBAL_EMB_MODEL
    
    @torch.inference_mode()
    def _embed(self, text):
        self._ensure_emb_model()
        with self._forward_lock: 
            emb = self._emb_model.encode([text], batch_size=8, normalize_embeddings=True)
        return emb[0].astype(float).tolist()
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), reraise=True, retry_error_callback=handle_retry_error)
    @torch.inference_mode()
    def _embed_many(self, texts):
        if not texts:
            return [] 
        
        self._ensure_emb_model()
        
        with self._forward_lock:
            embs = self._emb_model.encode(texts, batch_size=8, normalize_embeddings=True)
        return [row.astype(float).tolist() for row in embs] 
    
    def _l2norm(self, v):
        n = math.sqrt(sum(x*x for x in v)) or 1.0 
        return [x/n for x in v] 
    
    def _cosine(self, a, b):
        return sum(x * y for x, y in zip(a, b))
    
    def _dagify(self, edges, edge_w):
        from collections import defaultdict

        def build_adj(E):
            adj = defaultdict(list)
            for a, b in E:
                adj[a].append(b)
            return adj

        E = set(edges)
        W = dict(edge_w)

        while True:
            adj = build_adj(E)
            color = {u: 0 for u in range(self.num_agents)}  # 0=white,1=gray,2=black
            parent = {u: None for u in range(self.num_agents)}
            removed_any = False

            def dfs(u):
                nonlocal removed_any
                color[u] = 1
                for v in adj.get(u, []):
                    if removed_any:
                        return  # stop early; we’ll rebuild and restart
                    if color[v] == 0:
                        parent[v] = u
                        dfs(v)
                    elif color[v] == 1:
                        # Found a back edge (u -> v). Extract the cycle edges: v -> ... -> u -> v
                        cycle_nodes = [u]
                        x = u
                        while x != v and parent[x] is not None:
                            x = parent[x]
                            cycle_nodes.append(x)
                        cycle_nodes.reverse()  # v ... u
                        cycle_edges = []
                        for i in range(len(cycle_nodes) - 1):
                            cycle_edges.append((cycle_nodes[i], cycle_nodes[i+1]))
                        cycle_edges.append((u, v))  # close the cycle

                        # Remove the minimum-weight edge on this cycle
                        weakest = min(
                            cycle_edges,
                            key=lambda e: (W.get(e, 0.0), e[0], e[1])  # stable tie-break
                        )
                        # DEBUG: print the actual weakest edge
                        # print("remove weakest on cycle:", weakest, "w=", W.get(weakest, 0.0))
                        E.remove(weakest)
                        W.pop(weakest, None)
                        removed_any = True
                        return
                color[u] = 2

            for u in range(self.num_agents):
                if removed_any:
                    break
                if color[u] == 0:
                    dfs(u)

            if not removed_any:
                break  # no cycles left

        return E, W
    
    def _topo_order_by_contributions(self, edges, contributions):
        """
        Topo sort when there are multiple valid orders:
        tie-breakers:
            - higher contribution first
            - smaller node id first
        """
        
        indeg = [0] * self.num_agents 
        adj = defaultdict(list)
        
        for u ,v in edges:
            adj[u].append(v)
            indeg[v] += 1 
        
        heap = []
        for i in range(self.num_agents):
            if indeg[i] == 0:
                heapq.heappush(heap, (-contributions[i], i)) 
        
        
        order = [] 
        while heap:
            neg_c, u = heapq.heappop(heap)
            order.append(u)
            
            for v in adj[u]:
                indeg[v] -= 1 
                if indeg[v] == 0:
                    heapq.heappush(heap, (-contributions[v], v))
        
        return order if len(order) == self.num_agents else [] 
    
    def _propagate_on_dag(self, query, init_answers, edges, rounds=2, contributions=None):
        current = list(init_answers)
        adj_in = defaultdict(list)
        for u, v in edges:
            adj_in[v].append(u)
            
        order = list(range(self.num_agents))
        if contributions is not None:
            order = self._topo_order_by_contributions(edges, contributions) or order
        
        final_idx = order[-1]
        
        for r in range(max(1, rounds)):
            for i in order:
                preds = adj_in.get(i, [])
                
                use_feedback = (i == order[0]) # (r > 0 and i == order[0])
                
                if not preds and not use_feedback:
                    continue

                incoming = [(pid, current[pid]) for pid in preds]
                
                own_response = current[i]
                system_prompt = self.role_map.get(self.roles[i], self.role_map["Assistant"])
                
                if use_feedback:
                    prompt = self._update_prompt_leader_agent(query, own_response)
                else:
                    prompt = self._update_prompt(query, own_response, incoming)
                
                # dff = "=====" * 40 
                # print(f"\n{dff}\n[Agent {i}] -> Updated prompt: \n{prompt}\n{dff}\n")
                
                current[i] = self._call_llm(prompt=prompt, system_prompt=system_prompt, temperature=self.temperature)

            # check for consensus [whether to early stop]
            sims = self._pairwise_sims(current)
            
            contributions = self._approx_shapley(current, None) # reference 
            
            if self._check_for_consensus(sims):
                break 
            
            if self.reform and (r + 1) < rounds:
                # rebuild the DAG
                helpful = [] 
                for i in range(self.num_agents):
                    scored = [] 
                    for j in range(self.num_agents):
                        if j == i:
                            continue
                        base = sims[i][j] 
                        # bias toward listening to high-contribution agents 
                        adj = base * (1.0 + self.num_agents * (contributions[j] - contributions[i]))
                        scored.append((j, adj, base))

                    scored = [p for p in scored if p[2] >= self.sim_threshold]
                    
                    scored.sort(key=lambda x: (x[1], contributions[x[0]]), reverse=True)
                    keep = {j for (j, _, _) in scored[:self.top_k]}
                    
                    helpful.append(sorted(list(keep)))
                
                edges = set()
                edge_w = {}
                
                # print(helpful)
                
                for i in range(self.num_agents):
                    for j in helpful[i]:
                        edges.add((j, i))
                        adj = sims[i][j] * (1.0 + self.num_agents * (contributions[j] - contributions[i])) # with a multiplier of N [since the contributions are scaled by N] 
                        edge_w[(j, i)] = max(0.0, adj)
                
                if self.enforce_dag:
                    edges, edge_w = self._dagify(edges, edge_w)
                
                adj_in = defaultdict(list)
                for u, v in edges:
                    adj_in[v].append(u)
                
                order = self._topo_order_by_contributions(edges, contributions) or list(range(self.num_agents)) 
            
            final_idx = order[-1]
        
        return current 
    
    def _approx_shapley(self, answers, reference):
        """
        cosine sim.-based contribution weights: 
            - with reference: w_i = cos(ei, eref)
            - wo reference: cos(ei, centroid)
        """
        embs = self._embed_many(answers) # [self._embed(a) for a in answers]
        if reference:
            e_ref = self._embed(reference)
            raw = [max(0.0, self._cosine(e, e_ref)) for e in embs]
        else:
            centroid = self._weighted_centroid(embs, [1.0] * len(embs))
            raw = [max(0.0, self._cosine(e, centroid)) for e in embs]
        
        s = sum(raw)
        if s <= 1e-12:
            return [1.0 / len(answers)] * len(answers)
        
        t = max(1e-6, 1.0 / float(10.0))
        mx = max(raw)
        ex = [math.exp((r - mx) / t) for r in raw] 
        s = sum(ex) or 1.0 
        return [e / s for e in ex] 
        # return [r / s  for r in raw] 
    
    def _weighted_centroid(self, vecs, weights):
        d = len(vecs[0])
        s = [0.0] * d 
        wsum = sum(weights) or 1.0 
        for v, w in zip(vecs, weights):
            for k in range(d):
                s[k] += w * v[k]

        # for k in range(d):
        #     s[k] /= wsum

        return self._l2norm(s)
    
    def _pairwise_sims(self, answers):
        # embs = [self._embed(a) for a in answers]
        embs = self._embed_many(answers)  # batching 
        # print(">>>>", embs)
        n = len(embs)
        sims = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    sims[i][j] = self._cosine(embs[i], embs[j])
        
        return sims 

    def _check_for_consensus(self, sims):
        vals = []
        n = len(sims)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    vals.append(sims[i][j])
        
        if not vals:
            return True 
        
        vmin, vmax = min(vals), max(vals) 
        
        if vmin >= self.consensus_min_sim and (vmax - vmin) <= self.consensus_range_eps:
            return True
        return False 
