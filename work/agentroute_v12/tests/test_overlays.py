import json,pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'package')]
from telemetry import resolved_config,policy_overlay,Spans
from agentroute.runtime import Runtime,public_task,run_cascade
from agentroute.evidence_policy import select_reports

class Tokens:
    def encode(self,text,**kw):return list(text)
class Client:
    def __init__(self,outputs):self.outputs=iter(outputs)
    def call(self,messages,seed,max_tokens,category,ledger):
        text=next(self.outputs);ledger.append(dict(messages=messages,seed=seed,max_tokens=max_tokens,category=category,status='ok',prompt_tokens=10,completion_tokens=4,response_text=text));return text
def runtime(outputs,method):
    base=json.loads((ROOT/'package/configs/compact.json').read_text());c=resolved_config(base,{'method':method})
    rt=Runtime(c,c['models'][0],'fixture','http://fixture.invalid',Tokens(),None,17,'fixture');rt.client=Client(outputs);return rt
TASK=public_task({'question':'Public fixture','metric':'mcq','answers':['HIDDEN_GOLD']})
VALUES=['reason\nFINAL: '+x for x in 'ABCDBBAD']
class OverlayTests(unittest.TestCase):
    def test_observer_preserves_calls_and_output(self):
        a=runtime(VALUES,'cascade');b=runtime(VALUES,'cascade');oa=run_cascade(a,TASK,'cascade')
        with Spans() as spans:ob=run_cascade(b,TASK,'cascade')
        self.assertEqual(oa,ob);self.assertEqual(a.ledger,b.ledger);self.assertTrue(spans.events)
    def test_paid_control_matches_fixed_worker_inputs(self):
        a=runtime(VALUES,'route_fixed');b=runtime(VALUES+['FINAL: A'],'route_paid')
        run_cascade(a,TASK,'route_fixed');run_cascade(b,TASK,'route_paid')
        self.assertEqual(a.ledger,b.ledger[:-1]);self.assertEqual(b.ledger[-1]['category'],'aggregate')
    def test_no_comm_isolated_and_gold_not_in_prompts(self):
        a=runtime(VALUES,'no_comm');run_cascade(a,TASK,'no_comm')
        self.assertFalse(any(d['kind']=='peer' for d in a.deliveries))
        self.assertNotIn('Previous answer record',str(a.raw_calls));self.assertNotIn('HIDDEN_GOLD',str(a.raw_calls))
    def test_extra_stages_have_distinct_observable_effects(self):
        candidates=[{'sender':1,'text':'long reasoning text\nFINAL: B'},{'sender':2,'text':'short\nFINAL: B'}]
        self.assertEqual(select_reports('FINAL: A',candidates,'mcq')[0]['sender'],2)
        with policy_overlay('remove_representative_rank'):
            import agentroute.evidence_policy as p
            self.assertEqual(p.select_reports('FINAL: A',candidates,'mcq')[0]['sender'],1)
        self.assertEqual(len(select_reports('FINAL: A',candidates,'mcq',group_answers=False)),2)
        a=runtime(VALUES,'remove_vote_summary');run_cascade(a,TASK,'remove_vote_summary')
        self.assertNotIn('Previous answer record',str(a.raw_calls));self.assertTrue(any(d['kind']=='peer' for d in a.deliveries))
if __name__=='__main__':unittest.main()
