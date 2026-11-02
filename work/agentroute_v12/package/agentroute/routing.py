"""Fixed baseline communication. Paid routing was removed in v9."""
import time
METHODS=('broadcast','no_comm')
class Router:
    def __init__(self,settings,client,embedding=None):
        self.s=settings
    def select(self,method,question,role,messages,seed,rng,ledger):
        start=time.perf_counter()
        if method not in METHODS:
            raise ValueError('Only fixed baseline routing is exposed here; v9 AgentRoute uses local_decision and paid routing is disabled')
        selected=list(messages) if method=='broadcast' else []
        return selected,{'candidates':len(messages),'kept':len(selected),'budget':None,
            'fallback':False,'llm_candidates':0,'lexical_resolved':0,'embedding_resolved':0,
            'lexical_s':0.,'embedding_s':0.,'llm_s':0.,'wall_s':time.perf_counter()-start}
