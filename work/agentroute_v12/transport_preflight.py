"""Bounded physical-network prerequisite check; never a measured experiment receipt."""
import concurrent.futures,json,pathlib,subprocess,time
import modal

PROBE=r'''
import json,os,pathlib,platform,socket,subprocess,time
def cmd(args):
 try:
  p=subprocess.run(args,capture_output=True,text=True,timeout=15)
  return {'argv':args,'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except Exception as e:return {'argv':args,'error':type(e).__name__+': '+str(e)}
d={'unix_s':time.time(),'hostname':socket.gethostname(),'kernel':platform.release(),
   'region':os.getenv('MODAL_REGION'),'cloud':os.getenv('MODAL_CLOUD_PROVIDER')}
for name in ['/proc/sys/kernel/random/boot_id','/sys/class/dmi/id/product_uuid','/proc/self/status']:
 try:d[name]=pathlib.Path(name).read_text()
 except OSError as e:d[name]=str(e)
d['routes']=cmd(['ip','-j','route','show','default'])
try:dev=json.loads(d['routes']['stdout'])[0]['dev']
except Exception:dev=None
d['interface']=dev
if dev:
 d['before']=cmd(['tc','-j','qdisc','show','dev',dev])
 d['shaping']=cmd(['tc','qdisc','replace','dev',dev,'root','netem','delay','25ms','rate','10mbit'])
 d['after']=cmd(['tc','-j','qdisc','show','dev',dev])
 d['reset']=cmd(['tc','qdisc','del','dev',dev,'root'])
d['packet_capture']=cmd(['timeout','2','tcpdump','-i','any','-c','1','-nn'])
print(json.dumps(d))
'''

def run(co,folder):
 from coordinator_store import atomic_json
 out=folder/'transport_preflight';out.mkdir(exist_ok=False)
 report={'status':'CHECKING','evidence_class':'infrastructure_preflight_only','started_unix_s':time.time(),
         'experiment_receipts_created':0,'requested_regions':['us-east','us-west'],'nodes':[]}
 atomic_json(out/'REPORT.json',report)
 records=[]
 try:
  image=modal.Image.debian_slim(python_version='3.12').apt_install('iproute2','tcpdump','iputils-ping').build(co.app)
  def check(region):
   rate=.1419+2*.024;timeout=300;reservation=co.reserve_check(rate,timeout)
   vol=modal.Volume.from_name('agentroute-v12-netcheck-'+co.operation.lower()+'-'+region,create_if_missing=True,version=2)
   sb=None;r=None
   try:
    start=time.time()
    sb=modal.Sandbox.create('sleep','infinity',app=co.app,image=image,runtime='vm',region=region,
       cpu=(1,1),memory=(2048,2048),timeout=timeout,volumes={'/evidence':vol},
       tags={'campaign':'agentroute-v12-completion-20261031','role':'transport-preflight','operation':co.operation})
    r=co.register(sb,'transport-preflight',start,timeout,rate,vol,reservation)
    with co.lock:records.append((sb,r))
    proc=sb.exec('python','-c',PROBE,timeout=90)
    raw=proc.stdout.read();err=proc.stderr.read();code=proc.wait()
    result={'sandbox_id':sb.object_id,'requested_region':region,'exit_code':code,'stderr':err,'probe':json.loads(raw) if code==0 else raw}
    atomic_json(out/(region+'.json'),result)
    return result
   except Exception as exc:
    if sb is None:
     with co.lock:co.pending_reservations.pop(reservation,None)
    result={'requested_region':region,'error_type':type(exc).__name__,'error':str(exc)[:2000]}
    atomic_json(out/(region+'-ERROR.json'),result);return result
   finally:
    if sb and r:co.terminate(sb,r)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:report['nodes']=list(pool.map(check,['us-east','us-west']))
  passed=all(isinstance(n.get('probe'),dict) and n['probe'].get('shaping',{}).get('code')==0 for n in report['nodes'])
  report.update(status='PREREQUISITES_PARTIAL' if passed else 'BLOCKED_INFRASTRUCTURE',
    explanation='Kernel shaping tested; achieved bandwidth/delay and distinct physical hosts still require validation before any experiment.' if passed else 'Requested VM and/or kernel traffic shaping unavailable. No physical transport repetitions counted.',
    required_remedy='Two verified distinct hosts with kernel traffic shaping, packet capture and validated 10 Mbit/s plus 25 ms one-way delay.')
 except Exception as exc:report.update(status='BLOCKED_INFRASTRUCTURE',error_type=type(exc).__name__,error=str(exc)[:2000])
 finally:
  for sb,r in records:
   if r['state']!='terminated':co.terminate(sb,r)
  report.update(finished_unix_s=time.time(),estimated_campaign_cost_usd=co.cost(),all_preflight_workers_stopped=all(r['state']=='terminated' for _,r in records))
  atomic_json(out/'REPORT.json',report);co.checkpoint(force=True);subprocess.run(['sync','/workspace'],check=True)
