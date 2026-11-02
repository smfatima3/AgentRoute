"""Single bounded coordinator: two L40S replicas, isolated 10-question clients.

Closed task receipts are read back from committed Volumes and hash-verified.
Cloud reservations bound worst-case charges even if the coordinator disappears.
"""
import argparse,concurrent.futures,hashlib,json,os,secrets,shutil,sys,threading,time,urllib.request,uuid,zipfile,subprocess
from pathlib import Path
import modal
from coordinator_store import Store,atomic_json,hash_file
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1];OUT=BASE/'outputs/agentroute_v12_completion'
sys.path.insert(0,str(BASE/'work/v9_local/clean_release_check_final/agentroute_modal_codex_v9'))
from modal_runner.images import images
QUAL=['main_mmlu','transfer_svamp','short_math','short_bbh','component_ablations','seed_sensitivity','aggregation_controls','self_consistency','selforg_variant','stage_ablations']
RATE={'gpu':1.95+4*.1419+16*.024,'client':2*.1419+8*.024}

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def execute(sb,*args,timeout=300,secret=None):
    kwargs={'timeout':timeout}
    if secret:kwargs['secrets']=[secret]
    p=sb.exec(*args,**kwargs);out=p.stdout.read();err=p.stderr.read();code=p.wait()
    if code:raise RuntimeError('Command failed: '+out[-1500:]+err[-1500:])
    return out

class Coordinator:
    def __init__(self):
        self.manifest=read(OUT/'campaign.lock.json');self.store=Store(OUT/'coordinator.sqlite');self.lock=threading.RLock();self.stop=threading.Event()
        self.operation=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());self.folder=OUT/'runs'/self.operation;self.folder.mkdir(parents=True,exist_ok=False)
        self.pending_reservations={};self.checkpoint_lock=threading.Lock()
        self.registry=read(OUT/'resource_registry_live.json') if (OUT/'resource_registry_live.json').exists() else {'campaign_id':self.manifest['campaign_id'],'resources':[]}
        if os.getenv('V12_COORDINATOR_ID') and not any(r['id']==os.environ['V12_COORDINATOR_ID'] for r in self.registry['resources']):
            self.registry['resources'].append({'id':os.environ['V12_COORDINATOR_ID'],'role':'coordinator','owned':True,'state':'running',
              'created_unix_s':float(os.environ['V12_COORDINATOR_STARTED']),'timeout_s':43200,'rate_usd_per_hour':.1899,'operation':'cloud-coordinator','volume_id':os.environ['V12_COORDINATOR_VOLUME']})
        self.gpus=[];self.live={};self.export_counter=0;self.last_export=0;self.last_status=0
        self.app=modal.App.lookup('agentroute-v12-completion',create_if_missing=True)
        self.key=secrets.token_urlsafe(32);self.initial_billing=None
    def persist(self):
        with self.lock:atomic_json(OUT/'resource_registry_live.json',self.registry)
    def cost(self,future=False):
        now=time.time();total=0.
        for r in self.registry['resources']:
            end=r.get('stopped_unix_s',r['created_unix_s']+r['timeout_s'] if future else now)
            total+=max(0,end-r['created_unix_s'])*r['rate_usd_per_hour']/3600
        return total+(sum(self.pending_reservations.values()) if future else 0)
    def register(self,sb,role,start,timeout,rate,vol,reservation=None):
        r={'id':sb.object_id,'role':role,'owned':True,'state':'running','created_unix_s':start,'timeout_s':timeout,
           'rate_usd_per_hour':rate,'cost_mode':'conservative fixed requested resource limits; provider bill separate','operation':self.operation,'volume_id':vol.object_id}
        with self.lock:
            self.registry['resources'].append(r);self.live[sb.object_id]=(sb,vol,r)
            if reservation:self.pending_reservations.pop(reservation)
            self.persist()
        return r
    def reserve_check(self,rate,timeout):
        with self.lock:
            if self.cost(future=True)+rate*timeout/3600>28:raise RuntimeError('CREDIT_RESERVATION_LIMIT')
            token=uuid.uuid4().hex;self.pending_reservations[token]=rate*timeout/3600;return token
    def terminate(self,sb,r):
        if r['state']=='terminated':return
        sb.terminate(wait=True);code=sb.poll()
        if code is None:raise RuntimeError('Provider did not confirm shutdown')
        with self.lock:r.update(state='terminated',provider_exit_code=code,stopped_unix_s=time.time());self.persist()
    def recover(self):
        for r in self.registry['resources']:
            if r['role']=='coordinator':continue
            if r['state']!='terminated':
                sb=modal.Sandbox.from_id(r['id']);self.terminate(sb,r)
        for lease in self.store.status()['active_leases']:
            r=next((x for x in self.registry['resources'] if x.get('lease_id')==lease['id']),None)
            if r is None:raise RuntimeError('Unresolved lease ownership; no automatic reassignment')
            self.store.confirmed_dead(lease['id'],lease['epoch'],r.get('provider_exit_code'))
    def build(self):
        original_cpu=read(Path(read(OUT/'preparation_latest.json')['path'])/'state.json')['cpu_image_id']
        ci=modal.Image.from_id(original_cpu)
        for name in ['shard.py','telemetry.py']:ci=ci.add_local_file(ROOT/name,'/opt/v12/'+name,copy=True)
        ci=ci.env({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
        _,gi=images(ROOT/'package')
        # The compact package has its pinned model manifest under compact_data.
        # Supply the byte-identical compatibility path required by the unchanged serving preflight.
        gi=gi.add_local_file(ROOT/'package/compact_data/manifest.json','/opt/package/data/manifest.json',copy=True)
        for name in ['serve_v12.py','gpu_entry.py']:gi=gi.add_local_file(ROOT/name,'/opt/v12/'+name,copy=True)
        # Building the images allocates no GPU. Identical image/settings for A/B.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(im.build,self.app) for im in [ci,gi]]
            for f in fs:f.result()
        self.ci=ci;self.gi=gi
        atomic_json(self.folder/'images.json',{'cpu_image':ci.object_id,'gpu_image':gi.object_id,'model':self.manifest['model']})
        print('Both immutable images built; no GPU allocated yet.',flush=True)
    def launch_gpu(self,label,timeout):
        reservation=self.reserve_check(RATE['gpu'],timeout)
        name='agentroute-v12-'+self.operation.lower()+'-'+label.lower()
        vol=modal.Volume.from_name(name+'-logs',create_if_missing=True,version=2)
        cache=modal.Volume.from_name('agentroute-v12-model-cache-'+label.lower(),create_if_missing=True,version=2)
        plan={'operation_id':label,'model':self.manifest['model'],'revision':self.manifest['split_manifest']['models'][self.manifest['model']],
              'max_model_len':32768,'tensor_parallel_size':1,'gpu_timeout_s':timeout}
        start=time.time()
        sb=modal.Sandbox.create('/opt/gpu/bin/python','/opt/v12/serve_v12.py','--plan-json',json.dumps(plan),app=self.app,image=self.gi,
          gpu='L40S',cpu=(4,4),memory=(16384,16384),timeout=timeout,encrypted_ports=[8000],
          secrets=[modal.Secret.from_dict({'VLLM_API_KEY':self.key})],volumes={'/gpu-logs':vol,'/hf':cache},
          tags={'campaign':'agentroute-v12-completion-20261031','role':label,'operation':self.operation})
        r=self.register(sb,label,start,timeout,RATE['gpu'],vol,reservation);url=sb.tunnels()[8000].url.rstrip('/')
        item={'sb':sb,'vol':vol,'resource':r,'label':label,'url':url,'start':start,'timeout':timeout}
        with self.lock:self.gpus.append(item)
        until=time.time()+900
        while time.time()<until and not self.stop.is_set():
            if sb.poll() is not None:raise RuntimeError('GPU server exited before readiness')
            try:
                req=urllib.request.Request(url+'/v1/models',headers={'Authorization':'Bearer '+self.key})
                with urllib.request.urlopen(req,timeout=10) as response:models=json.loads(response.read())
                if any(x['id']==self.manifest['model'] for x in models['data']):break
            except Exception:time.sleep(5)
        else:raise RuntimeError('GPU readiness timeout')
        identity=json.loads(sb.filesystem.read_text('/gpu-logs/'+label+'/identity.json'))
        atomic_json(self.folder/(label+'-identity.json'),identity)
        print('Ready '+label+' '+sb.object_id,flush=True);return item
    def download_volume(self,volume,dest,immutable=False):
        dest.mkdir(parents=True,exist_ok=True);count=0
        for entry in volume.iterdir('/',recursive=True):
            if entry.type!=modal.volume.FileEntryType.FILE:continue
            rel=entry.path.lstrip('/');path=(dest/rel).resolve()
            if not path.is_relative_to(dest.resolve()):raise RuntimeError('Export path traversal')
            raw=b''.join(volume.read_file(rel));path.parent.mkdir(parents=True,exist_ok=True)
            if immutable and path.exists() and path.read_bytes()!=raw:raise RuntimeError('Immutable committed output changed')
            temp=path.with_suffix(path.suffix+'.download');temp.write_bytes(raw);temp.replace(path);count+=1
        return count
    def accept_downloaded(self,dest):
        for path in dest.rglob('receipt.json'):
            row=read(path)
            if row['status']!='measured':continue
            # Provider hashes refer to the evidence-volume root; local receipts use its immutable imported root.
            for ev in row['evidence']:ev['path']=(dest/ev['path']).relative_to(OUT).as_posix()
            if row['task_id'].startswith('probe-'):continue
            if any(not (OUT/e['path']).exists() or hash_file(OUT/e['path'])!=e['sha256'] for e in row['evidence']):continue
            self.store.accept(row,OUT)
    def checkpoint(self,force=False):
        with self.checkpoint_lock:self._checkpoint(force)
    def _checkpoint(self,force=False):
        now=time.time()
        with self.lock:
            resources=[(sb,v,r.copy()) for sb,v,r in self.live.values()]
        if force or now-self.last_export>=1200:
            stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(self.export_counter);self.export_counter+=1
            checkpoint=OUT/'checkpoints'/stamp;checkpoint.mkdir(parents=True)
            for sb,vol,r in resources:
                if r['state']=='running':
                    try:execute(sb,'sync','/gpu-logs' if r['role'].startswith('GPU') else '/evidence',timeout=60)
                    except Exception as exc:atomic_json(checkpoint/(r['id']+'-commit-error.json'),{'error_type':type(exc).__name__})
                self.download_volume(vol,checkpoint/'volumes'/r['id'])
            self.store.snapshot(checkpoint/'coordinator.sqlite')
            for name in ['campaign.lock.json','source_lock.json','protocol.json','resource_registry_live.json','REUSE_AUDIT.json']:
                shutil.copyfile(OUT/name,checkpoint/name)
            manifest={p.relative_to(checkpoint).as_posix():hash_file(p) for p in checkpoint.rglob('*') if p.is_file()}
            atomic_json(checkpoint/'MANIFEST.json',manifest)
            archive=checkpoint.with_suffix('.zip')
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for p in checkpoint.rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(checkpoint).as_posix())
            with zipfile.ZipFile(archive) as z:
                for name,h in manifest.items():
                    if hashlib.sha256(z.read(name)).hexdigest()!=h:raise RuntimeError('Checkpoint archive hash mismatch')
            atomic_json(OUT/'checkpoints/LATEST.json',{'created_unix_s':time.time(),'path':str(checkpoint),'archive':str(archive),'sha256':hash_file(archive),'verified_files':len(manifest),'contains_actual_remote_results':bool(resources)})
            self.last_export=time.time()
        status=self.store.status();status.update(status='RUNNING' if not self.stop.is_set() else 'STOPPING',resource_cost_estimate_usd=self.cost(),worst_case_reserved_usd=self.cost(True),credit_execution_ceiling_usd=28,resources=self.registry['resources'])
        atomic_json(OUT/'STATUS.json',status)
        rows=self.store.accepted();tmp=OUT/'results_index.jsonl.tmp'
        with tmp.open('w',encoding='utf8') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        tmp.replace(OUT/'results_index.jsonl');self.last_status=time.time()
        if os.name!='nt':subprocess.run(['sync','/workspace'],check=True,timeout=90)
    def worker(self,gpu,slot,probe=None,crash=False):
        label=gpu['label']+'-slot'+str(slot)
        while not self.stop.is_set():
            # Preserve enough time for a bounded question; shrink late shards instead of starting a full cell at expiry.
            available=gpu['start']+gpu['timeout']-time.time()-180
            if available<300:return
            if probe:
                lease={'id':'probe-'+uuid.uuid4().hex,'epoch':0,'tasks':probe,'worker':label}
            else:
                lease=self.store.claim(label,QUAL,size=min(10,max(1,int(available/240))))
                if not lease:return
            timeout=min(3600,int(available))
            with self.lock:
                try:reservation=self.reserve_check(RATE['client'],timeout)
                except RuntimeError:
                    if not probe:
                        # No provider process was created: explicit launch-not-started proof.
                        self.store.confirmed_dead(lease['id'],lease['epoch'],-999,retry=True)
                    return
            vol=modal.Volume.from_name('agentroute-v12-'+lease['id'],create_if_missing=True,version=2)
            prefix=f"{self.manifest['campaign_id']}/{label}/{lease['id']}";attempt=self.operation+'/'+lease['id'];fence=prefix+'/lease.json'
            job={'tasks':lease['tasks'],'attempt':attempt,'epoch':lease['epoch'],'prefix':prefix,'fence_file':fence,'crash_after_first_model_call':crash,
              'runtime_lineage':read(OUT/'RUNTIME_LINEAGE.json') if (OUT/'RUNTIME_LINEAGE.json').exists() else None,
              'gpu':{'id':gpu['sb'].object_id,'label':gpu['label']},'source_lock':self.manifest['source_lock']['package'],
              'source_lock_sha256':read(OUT/'protocol.json')['source_lock_sha256'],
              'scheduling':{'query_concurrency_per_gpu':2,'slot':slot,'lease_assigned_unix_s':time.time(),'warmup_calls':2}}
            sb=None;r=None;dest=OUT/'evidence'/lease['id'];failed=False
            try:
                start=time.time()
                sb=modal.Sandbox.create('sleep','infinity',app=self.app,image=self.ci,cpu=(2,2),memory=(8192,8192),timeout=timeout,
                  volumes={'/evidence':vol,'/hf':modal.Volume.from_name('agentroute-v12-client-cache').read_only()},
                  secrets=[modal.Secret.from_dict({'VLLM_API_KEY':self.key,'AGENTROUTE_BASE_URL':gpu['url']+'/v1'})],
                  tags={'campaign':'agentroute-v12-completion-20261031','role':'quality-client','operation':self.operation})
                r=self.register(sb,'client',start,timeout,RATE['client'],vol,reservation);r.update(lease_id=lease['id'],epoch=lease['epoch'],task_ids=[t['task_id'] for t in lease['tasks']]);self.persist()
                execute(sb,'mkdir','-p','/work',timeout=30)
                execute(sb,'cp','-a','/opt/package','/work/project',timeout=120)
                execute(sb,'mkdir','-p','/work/project/recovery','/evidence/'+prefix,timeout=30)
                execute(sb,'/opt/client/bin/python','/opt/restore_git_metadata.py',timeout=120)
                job['client_sandbox_id']=sb.object_id
                sb.filesystem.write_text(json.dumps(job),'/tmp/job.json');sb.filesystem.write_text(json.dumps({'state':'active','epoch':lease['epoch']}),'/evidence/'+fence)
                execute(sb,'sync','/evidence',timeout=60)
                proc=sb.exec('/opt/client/bin/python','/opt/v12/shard.py','--job','/tmp/job.json',timeout=timeout)
                def drain(stream,path):
                    with path.open('w',encoding='utf8') as f:
                        for line in stream:f.write(line);f.flush()
                self.folder.mkdir(exist_ok=True)
                threads=[threading.Thread(target=drain,args=(stream,self.folder/(lease['id']+suffix)),daemon=True) for stream,suffix in [(proc.stdout,'.log'),(proc.stderr,'.err')]]
                for t in threads:t.start()
                started=time.time();last_read=0
                while proc.poll() is None and not self.stop.wait(10):
                    if not probe:self.store.heartbeat(lease['id'],lease['epoch'])
                    if crash:
                        try:barrier=json.loads(sb.filesystem.read_text('/evidence/'+prefix+'/CRASH_READY.json'))
                        except FileNotFoundError:barrier=None
                        if barrier and barrier['completed_model_calls']>0:
                            self.terminate(sb,r);break
                    if time.time()-last_read>120:
                        execute(sb,'sync','/evidence',timeout=60)
                        # Only closed receipts are accepted; partial journals remain in cost/recovery snapshots.
                        self.download_volume(vol,dest);self.accept_downloaded(dest);last_read=time.time()
                if self.stop.is_set() and r['state']!='terminated':self.terminate(sb,r)
                code=proc.poll();failed=code not in (0,None)
                for t in threads:t.join(timeout=5)
            except Exception as exc:
                failed=True;atomic_json(self.folder/(lease['id']+'-error.json'),{'type':type(exc).__name__,'message':str(exc)[:2000]})
                print('Preserved worker failure '+lease['id']+' '+type(exc).__name__,flush=True)
            finally:
                if sb and r:
                    if r['state']!='terminated':
                        try:execute(sb,'sync','/evidence',timeout=60)
                        except Exception:pass
                    self.terminate(sb,r)
                    self.download_volume(vol,dest);self.accept_downloaded(dest)
                    if not probe:self.store.confirmed_dead(lease['id'],lease['epoch'],r['provider_exit_code'],retry=not failed)
                elif not probe:self.store.confirmed_dead(lease['id'],lease['epoch'],-999,retry=False)
                with self.lock:self.pending_reservations.pop(reservation,None)
                self.checkpoint()
            if probe:return {'lease':lease,'resource':r,'path':str(dest)}
            if failed:return
    def run(self):
        self.recover();self.persist();self.checkpoint(force=True)
        checkpoint_errors=[]
        def checkpoint_loop():
            while not self.stop.wait(300):
                try:self.checkpoint()
                except Exception as exc:
                    checkpoint_errors.append(type(exc).__name__)
                    atomic_json(OUT/'CHECKPOINT_FAILURE.json',{'unix_s':time.time(),'error_type':type(exc).__name__})
                    self.stop.set()
        writer=threading.Thread(target=checkpoint_loop,daemon=True);writer.start()
        self.build()
        remaining=28-self.cost(True);timeout=min(7200,int(max(0,remaining-4)/(2*RATE['gpu'])*3600))
        if timeout<900:raise RuntimeError('Insufficient credit reservation for two GPU startup/recovery tests')
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                gs=[pool.submit(self.launch_gpu,label,timeout) for label in ['GPU-A','GPU-B']]
                gpu=[f.result() for f in gs]
            # Recovery probe is separate supplementary development evidence, never counted as extra benchmark success.
            original=next(t for t in self.manifest['tasks'] if t['group']=='development' and t['dataset']=='bbh' and t['method']=='original')
            probe=dict(original,task_id='probe-'+original['task_id'])
            killed=self.worker(gpu[0],0,[probe],crash=True)
            if killed['resource']['state']!='terminated' or not list(Path(killed['path']).rglob('CRASH_READY.json')):raise RuntimeError('Crash test did not reach a model-call barrier and confirmed worker death')
            recovered=self.worker(gpu[1],0,[probe]);receipts=list(Path(recovered['path']).rglob('receipt.json'))
            if len(receipts)!=1 or read(receipts[0])['status']!='measured':raise RuntimeError('Development recovery probe did not produce verified result')
            # Require real per-request server timestamps from the instrumented engine before held-out execution.
            execute(gpu[1]['sb'],'sync','/gpu-logs',timeout=60)
            spans=b''.join(gpu[1]['vol'].read_file('GPU-B/request_spans.jsonl'))
            parsed=[json.loads(x) for x in spans.splitlines()]
            if not any(x['finished'] and x['state']['scheduled_ts']>=x['state']['queued_ts']>0 for x in parsed):raise RuntimeError('Per-request serving queue instrumentation absent')
            atomic_json(self.folder/'RECOVERY_TEST.json',{'status':'PASSED','interrupted_development_worker':killed,'recovered_development_worker':recovered,'server_spans_sha256':hashlib.sha256(spans).hexdigest(),'heldout_questions_used':0})
            print('Development crash recovery and real serving queue timestamps verified. Starting held-out queue.',flush=True)
            self.checkpoint(force=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                jobs=[pool.submit(self.worker,g,s) for s in range(2) for g in gpu]
                while any(not f.done() for f in jobs):
                    if self.cost()>=27.5:self.stop.set()
                    if time.time()-self.last_status>=300:self.checkpoint()
                    time.sleep(5)
                for f in jobs:f.result()
        finally:
            self.stop.set()
            writer.join(timeout=5)
            for sb,vol,r in list(self.live.values()):
                if r['state']!='terminated':
                    try:execute(sb,'sync','/gpu-logs' if r['role'].startswith('GPU') else '/evidence',timeout=60)
                    except Exception:pass
                    self.terminate(sb,r)
            self.checkpoint(force=True)
            atomic_json(self.folder/'RUN_END.json',{'status':'BOUNDED_ALLOCATION_ENDED','queue':self.store.status(),'estimated_resource_cost_usd':self.cost(),'all_registered_resources_stopped':all(r['state']=='terminated' for r in self.registry['resources'])})

def main():
    from filelock import FileLock
    with FileLock(str(OUT/'controller.lock'),timeout=0):
        co=Coordinator();co.run()
if __name__=='__main__':main()
