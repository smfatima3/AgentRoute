import importlib.util,json,pathlib,sys,tempfile,threading,time,types,unittest
from unittest.mock import patch
try:import resource
except ImportError:
 sys.modules['resource']=types.SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda _:types.SimpleNamespace(ru_maxrss=0))
path=pathlib.Path(__file__).resolve().parents[1]/'physical_node.py'
spec=importlib.util.spec_from_file_location('physical_node',path);node=importlib.util.module_from_spec(spec);spec.loader.exec_module(node)

class PhysicalTests(unittest.TestCase):
 def test_real_http_integrity_and_reused_connection(self):
  server=node.http.server.ThreadingHTTPServer(('127.0.0.1',0),node.Handler)
  thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with node.Session() as session:
    for body in ['数学 Δ'.encode(),b'a'*200000]:
     result=node.rpc(session,f'http://127.0.0.1:{server.server_port}/',body)
     self.assertEqual(result['request_body_bytes'],len(body))
  finally:server.shutdown();server.server_close()
 def test_control_payloads_retained_with_declared_scaling(self):
  record={'deliveries':[{'sender':'0','recipient':'1','kind':'peer','text':'original'},
    {'sender':'coordinator','recipient':'2','kind':'control','text':'summary'}]}
  messages=[json.loads(x) for x in node.expand(record,8)]
  self.assertEqual(len(messages),6)
  self.assertEqual(sum(m['kind']=='control' for m in messages),2)
  self.assertEqual({m['text'] for m in messages},{'original','summary'})
 def test_open_loop_saturation_keeps_denominator(self):
  def slow(*a,**k):time.sleep(.08);return {'request_body_bytes':1,'response_body_bytes':1}
  plan={'records':[{'deliveries':[{'sender':'0','recipient':'1','kind':'peer','text':'x'}]}],
        'agents':4,'url':'unused','rate':1000,'warmup_s':0,'steady_s':.12}
  with tempfile.TemporaryDirectory() as temp,patch.object(node,'rpc',slow),patch.object(node.subprocess,'run'):
   out=pathlib.Path(temp)/'output';node.run(plan,out);metrics=json.loads((out/'metrics.json').read_text())
   self.assertEqual(metrics['scheduled'],120)
   self.assertGreater(metrics['counts']['dropped'],0)
   self.assertEqual(sum(metrics['counts'].values()),120)
   self.assertEqual(metrics['unfinished'],0)
if __name__=='__main__':unittest.main()
