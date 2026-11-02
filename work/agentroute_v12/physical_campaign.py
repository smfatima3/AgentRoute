"""Single-owner, budget-limited physical transport controller, after quality completion."""
import concurrent.futures,hashlib,json,math,os,pathlib,random,sqlite3,subprocess,sys,threading,time,uuid
import modal
ROOT=pathlib.Path('/workspace/work/agentroute_v12');OUT=pathlib.Path('/workspace/outputs/agentroute_v12_completion')
sys.path.insert(0,str(ROOT))
from coordinator_store import atomic_json,hash_file
from run_campaign import Coordinator,execute
RATE=(.1419+2*.024)*1.75  # Conservative maximum placement multiplier for explicitly selected regions.

def load_sources(co):
 methods=['cascade','dense','safesieve','card','selforg','agentnet'];rows={m:{} for m in methods};provenance=[]
 with co.store.connection() as db:tasks=list(db.execute("SELECT spec,receipt FROM tasks WHERE group_name='short_bbh' AND state='accepted' ORDER BY id"))
 for row in tasks:
  task=json.loads(row['spec'])
  if task['method'] not in rows:continue
  receipt=json.loads(row['receipt']);ev=next(e for e in receipt['evidence'] if e['role']=='raw');p=OUT/ev['path']
  if hash_file(p)!=ev['sha256']:raise RuntimeError('Source raw hash mismatch')
  if p.suffix=='.jsonl':candidates=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
  else:
   v=json.loads(p.read_text());candidates=v if isinstance(v,list) else [v]
  raw=next((r for r in candidates if r.get('id')==task['question_id'] and r.get('method')==task['method']),None)
  if raw is None:raise RuntimeError('Source record identity missing '+task['task_id'])
  rows[task['method']][task['question_id']]=raw
  provenance.append({'task_id':task['task_id'],'question_id':task['question_id'],'method':task['method'],**ev})
 common=sorted(set.intersection(*(set(r) for r in rows.values())))[:10]
 if len(common)!=10:raise RuntimeError('Ten common source question traces unavailable')
 # No outcome filtering; fixed lexical question order, selected before transport observations.
 return {m:[{'id':q,'deliveries':rows[m][q]['deliveries']} for q in common] for m in methods},[p for p in provenance if p['question_id'] in common]

def claim(co,task):
 now=time.time();lease='transport-'+uuid.uuid4().hex
 with co.store.connection() as db:
  db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT * FROM tasks WHERE id=?',(task['task_id'],)).fetchone()
  if row['state']!='pending':raise RuntimeError('Transport task not pending')
  epoch=db.execute('SELECT COALESCE(MAX(epoch),0)+1 FROM leases').fetchone()[0]
  db.execute('INSERT INTO leases(id,epoch,worker,state,started,heartbeat,expires) VALUES(?,?,?,?,?,?,?)',(lease,epoch,'CPU-NET-A','active',now,now,now+7200))
  db.execute("UPDATE tasks SET state='leased',lease_id=?,epoch=?,attempts=attempts+1 WHERE id=?",(lease,epoch,task['task_id']))
  db.execute('INSERT INTO events(unix_s,kind,details) VALUES(?,?,?)',(now,'physical_transport_claim',json.dumps({'lease':lease,'task_id':task['task_id'],'epoch':epoch})))
 return lease,epoch

def main():
 co=Coordinator();folder=co.folder/'physical_transport';folder.mkdir()
 summary={'status':'STARTING','accepted_repetitions':0,'evidence_class':'physical_payload_replay','gpu_allocated':False}
 atomic_json(folder/'REPORT.json',summary)
 for filename in ['physical_campaign.py','physical_node.py']:
  (folder/filename).write_bytes(pathlib.Path('/opt',filename).read_bytes())
 atomic_json(folder/'IMPLEMENTATION_LINEAGE.json',{'reason':'Disable receiver Nagle buffering so HTTP acknowledgement packetization does not add an unintended delayed-ACK round trip to the frozen 25ms one-way kernel delay. Preserve failed first shaping validation; tolerance unchanged. Randomize trace rotation by declared repetition seed. Apply conservative region price multiplier.','prior_validation':'20261102T012137Z: RTT increment 75.92ms, bandwidth 9.35Mbit/s; no accepted transport tasks.'})
 identity=json.loads(pathlib.Path('/opt/transport_identity.json').read_text())
 # The launcher supplies its actual identity using the mounted volume after creation.
 assigned=OUT/'TRANSPORT_ASSIGNED.json';until=time.time()+120
 while not assigned.exists() or json.loads(assigned.read_text()).get('nonce')!=identity['nonce']:
  if time.time()>until:raise RuntimeError('Coordinator identity not supplied')
  time.sleep(1)
 assigned=json.loads(assigned.read_text());nodes=[];capture=None;lease=None
 for r in co.registry['resources']:
  if r['state']!='terminated':
   code=modal.Sandbox.from_id(r['id']).poll()
   if code is None:raise RuntimeError('Existing owned resource running')
   r.update(state='terminated',provider_exit_code=code,stopped_unix_s=min(time.time(),r['created_unix_s']+r['timeout_s']))
 co.registry['resources'].append({'id':assigned['id'],'role':'coordinator','state':'running','owned':True,'created_unix_s':assigned['created_unix_s'],'timeout_s':identity['timeout_s'],'rate_usd_per_hour':.1899})
 co.persist()
 budget_token='transport-egress';co.pending_reservations[budget_token]=.5;egress_reserved=.0
 # A half-dollar egress allowance is conservatively charged even when plan free allowance applies.
 def checkpoint_loop():
  while not co.stop.wait(300):
   try:co.checkpoint()
   except Exception as e:atomic_json(folder/'CHECKPOINT_ERROR.json',{'error':str(e)});co.stop.set()
 threading.Thread(target=checkpoint_loop,daemon=True).start()
 try:
  with co.store.connection() as db:
   missing=[r['id'] for r in db.execute("SELECT id FROM tasks WHERE group_name IN ('short_bbh','seed_sensitivity') AND state NOT IN ('accepted','cancelled_by_user')")]
  if missing:raise RuntimeError('Quality dependency incomplete: '+str(len(missing)))
  if co.store.status()['active_leases']:raise RuntimeError('Unresolved prior leases')
  sources,provenance=load_sources(co)
  atomic_json(folder/'SOURCE_FREEZE.json',{'sources':provenance,'selected_question_ids':[r['id'] for r in sources['dense']],
    'mapping':'All recorded logical deliveries, including control, replayed in original order over A-to-B HTTP. For N=8/16 clone numeric 0..3 identities modulo4; peer endpoints expand as Cartesian clone pairs; coordinator names remain singleton; text unchanged. No new agent reasoning. Transport acknowledgements are additional recorded bytes.',
    'source_sha256':hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),'node_source_sha256':hash_file('/opt/physical_node.py')})
  image=modal.Image.debian_slim(python_version='3.12').apt_install('iproute2','tcpdump','iputils-ping','procps').add_local_file('/opt/physical_node.py','/opt/physical_node.py',copy=True).build(co.app)
  timeout=min(7200,int((28-co.cost(True)-.05)/(2*RATE)*3600),int(assigned['created_unix_s']+identity['timeout_s']-time.time()-180))
  if timeout<300:raise RuntimeError('Credit cannot reserve transport validation')
  def create(region):
   token=co.reserve_check(RATE,timeout);vol=modal.Volume.from_name('agentroute-v12-physical-'+co.operation.lower()+'-'+region,create_if_missing=True,version=2);start=time.time()
   sb=modal.Sandbox.create('python','/opt/physical_node.py','server',app=co.app,image=image,runtime='vm',region=region,cpu=(1,1),memory=(2048,2048),timeout=timeout,encrypted_ports=[8000],volumes={'/evidence':vol},tags={'campaign':'agentroute-v12-completion-20261031','role':'physical-transport','operation':co.operation})
   r=co.register(sb,'CPU-NET-'+region,start,timeout,RATE,vol,token);r['requested_region']=region;co.persist()
   node=(sb,vol,r)
   with co.lock:nodes.append(node)
   return node
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs=[pool.submit(create,r) for r in ['us-east','us-west']];a,b=[f.result() for f in fs]
  aa,av,ar=a;bb,bv,br=b;url=bb.tunnels()[8000].url
  facts=[]
  for sb,vol,r in [a,b]:
   fact=json.loads(execute(sb,'python','-c',"import os,json,pathlib;print(json.dumps({'region':os.getenv('MODAL_REGION'),'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text(),'kernel':os.uname().release}))"));fact['sandbox_id']=sb.object_id;facts.append(fact)
  if [f['region'] for f in facts]!=['us-east','us-west']:raise RuntimeError('Actual distinct region residency unverified')
  atomic_json(folder/'HOSTS.json',{'nodes':facts,'host_separation_evidence':'Provider-reported distinct geographical regions; boot IDs alone are not physical-host proof.'})
  def shape(mode):
   for sb,_,_ in [a,b]:
    if mode=='constrained':execute(sb,'tc','qdisc','replace','dev','eth0','root','netem','delay','25ms','rate','10mbit')
    else:
     p=sb.exec('tc','qdisc','del','dev','eth0','root',timeout=20);p.wait()
   atomic_json(folder/('qdisc-'+mode+'-'+str(time.time_ns())+'.json'),[execute(sb,'tc','-s','qdisc','show','dev','eth0') for sb,_,_ in [a,b]])
  checks={}
  for mode in ['ordinary','constrained','ordinary_after']:
   shape('ordinary' if mode.startswith('ordinary') else mode)
   execute(aa,'python','/opt/physical_node.py','validate','--url',url,'--out','/evidence/validation-'+mode+'.json',timeout=240)
   checks[mode]=json.loads(aa.filesystem.read_text('/evidence/validation-'+mode+'.json'))
  delta=(checks['constrained']['rtt_median_s']-(checks['ordinary']['rtt_median_s']+checks['ordinary_after']['rtt_median_s'])/2)*1000
  passed=abs(delta-50)<=15 and 8.5<=checks['constrained']['upload_mbit_s_median']<=11.5
  atomic_json(folder/'SHAPING_VALIDATION.json',{'passed':passed,'rtt_delta_ms':delta,'measurements':checks,'required_bandwidth_mbit_s':[8.5,11.5],'required_added_round_trip_ms':[35,65]})
  if not passed:raise RuntimeError('Achieved physical shaping outside frozen tolerance')
  def execute_plan(plan,label,limit):
   aa.filesystem.write_text(json.dumps(plan),'/tmp/plan.json')
   execute(aa,'python','/opt/physical_node.py','run','--plan','/tmp/plan.json','--out','/evidence/'+label,timeout=limit)
   return json.loads(aa.filesystem.read_text('/evidence/'+label+'/metrics.json'))
  from importlib.util import spec_from_file_location,module_from_spec
  spec=spec_from_file_location('node','/opt/physical_node.py');node=module_from_spec(spec);spec.loader.exec_module(node)
  egress_reserved=.01
  def reserve_payload(records,n,rate,seconds):
   nonlocal egress_reserved
   expanded=[node.expand(r,n) for r in records]
   max_body=max(sum(len(x)+2048 for x in trace) for trace in expanded)
   charge=3*max_body*math.ceil(seconds*rate)/2**30*.04
   if egress_reserved+charge>.45:raise RuntimeError('Conservative physical network egress reservation exhausted')
   egress_reserved+=charge
   atomic_json(folder/'EGRESS_RESERVATION.json',{'usd':egress_reserved,'cap_usd':.5,'note':'Three times maximal recorded payload plus 2048 bytes per request, charged at .04/GiB; usage allowance not assumed'})
  rates={}
  for n in [4,8,16]:
   for mode in ['ordinary','constrained']:
    shape(mode);ramp=[];high=128;bracket=False
    for rate in [1,4,12,32,64,128]:
     if time.time()+60>ar['created_unix_s']+timeout:raise RuntimeError('Transport allocation cap reached during calibration')
     label=f'calibration-{n}-{mode}-{rate}'
     plan={'records':sources['dense'],'agents':n,'url':url,'rate':rate,'warmup_s':2,'steady_s':8,'calibration':True}
     reserve_payload(sources['dense'],n,rate,10)
     m=execute_plan(plan,label,60);ramp.append(m)
     if m['failure_fraction']>.05:high=rate;bracket=True;break
    rates[str(n)+'/'+mode]={'high':high,'low':max(.25,high/4),'saturation_bracket_found':bracket,'ramp':ramp}
    atomic_json(folder/'CALIBRATION_IN_PROGRESS.json',rates)
  atomic_json(folder/'RATES_FROZEN.json',rates)
  with co.store.connection() as db:tasks=[json.loads(r['spec']) for r in db.execute("SELECT spec FROM tasks WHERE group_name IN ('transport_anchors','transport_scaling') AND state='pending'")]
  random.Random(20261031).shuffle(tasks);atomic_json(folder/'TASK_ORDER.json',[t['task_id'] for t in tasks])
  for task in tasks:
   rate=rates[str(task['agents'])+'/'+task['mode']][task['load']];steady=max(60,math.ceil(1000/rate));required=steady+10+60
   if co.stop.is_set() or time.time()+required>ar['created_unix_s']+timeout:
    summary.update(status='INCOMPLETE_RESOURCE_CAP',next_task=task['task_id']);break
   reserve_payload(sources[task['method']],task['agents'],rate,steady+10)
   shape(task['mode']);lease,epoch=claim(co,task);directory=folder/'tasks'/task['task_id'];directory.mkdir(parents=True)
   plan={'records':sources[task['method']],'agents':task['agents'],'url':url,'rate':rate,'warmup_s':10,'steady_s':steady,'task':task,'epoch':epoch,'scheduling_seed':20261031+task['repetition']}
   # Capture only the receiver experiment port, excluding Modal control RPCs.
   capture=bb.exec('tcpdump','-i','eth0','-U','-s','128','-w','/evidence/'+lease+'.pcap','tcp','port','8000',timeout=required)
   metrics=execute_plan(plan,lease,required)
   execute(bb,'pkill','-INT','tcpdump');capture.wait();capture_log=capture.stderr.read();capture=None
   execute(bb,'sync','/evidence')
   for remote,local in [(lease+'/raw.json','raw.json'),(lease+'/metrics.json','metrics.json')]:
    (directory/local).write_bytes(b''.join(av.read_file(remote)))
   (directory/'transport.pcap').write_bytes(b''.join(bv.read_file(lease+'.pcap')))
   (directory/'packet_capture.log').write_text(capture_log)
   evidence=[{'role':role,'path':(directory/name).relative_to(OUT).as_posix(),'sha256':hash_file(directory/name)} for role,name in [('raw','raw.json'),('metrics','metrics.json'),('packet_capture','transport.pcap')]]
   receipt={'task_id':task['task_id'],'group':task['group'],'fingerprint':task['fingerprint'],'attempt_id':co.operation+'/'+lease,'epoch':epoch,'status':'measured','outcome':'transport_completed','evidence':evidence}
   atomic_json(directory/'receipt.json',receipt);subprocess.run(['sync','/workspace'],check=True);co.store.accept(receipt,OUT);co.store.confirmed_dead(lease,epoch,0);lease=None
   summary['accepted_repetitions']+=1;atomic_json(folder/'REPORT.json',summary);co.checkpoint()
  else:summary['status']='EXECUTION_COMPLETE_PENDING_ANALYSIS'
 except Exception as exc:summary.update(status='BLOCKED',error_type=type(exc).__name__,error=str(exc)[:3000])
 finally:
  co.stop.set()
  for sb,vol,r in nodes:
   if r['state']!='terminated':co.terminate(sb,r)
   co.download_volume(vol,folder/'node_evidence'/r['id'])
  if lease:co.store.confirmed_dead(lease,epoch,137,retry=False)
  co.pending_reservations.pop(budget_token,None)
  co.registry['resources'].append({'id':'egress-reservation-'+co.operation,'role':'conservative_network_charge','state':'terminated','owned':True,'created_unix_s':0,'stopped_unix_s':3600,'timeout_s':3600,'rate_usd_per_hour':.5,'cost_mode':'Conservative maximum network reserve including validation and failed requests; not provider invoice'})
  co.persist();co.checkpoint(force=True)
  summary.update(finished_unix_s=time.time(),cost_estimate_usd=co.cost(),all_transport_workers_stopped=all(r['state']=='terminated' for _,_,r in nodes))
  atomic_json(folder/'REPORT.json',summary);atomic_json(OUT/'PHYSICAL_TRANSPORT_LATEST.json',{'path':str(folder),'report':summary});subprocess.run(['sync','/workspace'],check=True)
if __name__=='__main__':main()
