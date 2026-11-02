"""User-authorized short scheduling window; no change to inference or scoring."""
import concurrent.futures,hashlib,json,os,pathlib,sqlite3,sys,threading,time,subprocess
ROOT=pathlib.Path('/workspace/work/agentroute_v12');OUT=pathlib.Path('/workspace/outputs/agentroute_v12_completion')
sys.path.insert(0,str(ROOT))
def main():
    deadline=float(os.environ['SHORT_DEADLINE']);stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());folder=OUT/'short_windows'/stamp;folder.mkdir(parents=True)
    from coordinator_store import atomic_json
    import modal
    registry=json.loads((OUT/'resource_registry_live.json').read_text())
    for r in registry['resources']:
        if r['state']!='terminated':
            code=modal.Sandbox.from_id(r['id']).poll()
            if code is None:raise RuntimeError('Previous owned resource remains active')
            r.update(state='terminated',provider_exit_code=code,stopped_unix_s=time.time())
    atomic_json(OUT/'resource_registry_live.json',registry)
    with sqlite3.connect(OUT/'coordinator.sqlite') as db:
        db.execute("UPDATE tasks SET state='cancelled_by_user' WHERE group_name='transfer_svamp' AND state!='accepted'")
        db.execute('INSERT INTO events(unix_s,kind,details) VALUES(?,?,?)',(time.time(),'user_scope_amendment','SVAMP removed by explicit user request; original manifest and evidence retained'))
    atomic_json(OUT/'ACTIVE_SCOPE.json',{'excluded_datasets':['svamp'],'original_expected_tasks':3013,'revised_expected_tasks':2533})
    # Reuse the already verified development recovery test, not a new science result.
    proofs=[p for p in (OUT/'runs').glob('*/RECOVERY_TEST.json') if json.loads(p.read_text()).get('status')=='PASSED']
    if not proofs:raise RuntimeError('No prior successful recovery validation')
    import run_campaign as module
    source=(ROOT/'run_campaign.py').read_text()
    source=source.replace("available=gpu['start']+gpu['timeout']-time.time()-180","available=gpu['start']+gpu['timeout']-time.time()-15")
    source=source.replace('if available<300:return','if available<25:return')
    source=source.replace("self.store.claim(label,QUAL,size=min(10,max(1,int(available/240))))","self.store.claim(label,self.slot_groups[(gpu['label'],slot)],size=1)")
    amended=ROOT/('short_controller_'+stamp+'.py');amended.write_text(source)
    import importlib.util
    spec=importlib.util.spec_from_file_location('short_controller',amended);runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    co=runtime.Coordinator();co.slot_groups={('GPU-A',0):['short_bbh'],('GPU-B',0):['short_math'],('GPU-A',1):['component_ablations'],('GPU-B',1):['stage_ablations']}
    before=co.store.status();atomic_json(folder/'PLAN.json',{'deadline_unix_s':deadline,'before':before,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'reused_recovery_validation':str(proofs[-1]),'physical_transport':'BLOCKED_NOT_IMPLEMENTED_VALIDATED','live_anchors':'BLOCKED_NOT_IMPLEMENTED_VALIDATED'})
    cutoff=threading.Timer(max(0,deadline-time.time()-10),co.stop.set);cutoff.daemon=True;cutoff.start()
    try:
        co.build()
        seconds=int(deadline-time.time())
        if seconds<75:raise RuntimeError('Insufficient time left for model readiness and a valid attempt')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(co.launch_gpu,label,seconds) for label in ['GPU-A','GPU-B']]
            gpu=[f.result() for f in futures]
        if os.getenv('SHORT_FROM_READY')=='1':
            measurement_deadline=min(deadline,time.time()+300)
            window=threading.Timer(max(0,measurement_deadline-time.time()),co.stop.set);window.daemon=True;window.start()
            atomic_json(folder/'MEASUREMENT_WINDOW.json',{'started_unix_s':time.time(),'deadline_unix_s':measurement_deadline,'parallel_slots':4,'groups':['short_bbh','short_math','component_ablations','stage_ablations']})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(co.worker,g,s) for s in range(2) for g in gpu]
            atomic_json(folder/'WORKERS_DISPATCHED.json',{'unix_s':time.time(),'slots':4,'note':'Dispatch submitted; actual sandbox creation is recorded in resource_registry_live.json'})
            for f in jobs:f.result()
    except Exception as exc:atomic_json(folder/'ERROR.json',{'type':type(exc).__name__,'message':str(exc)[:2000]})
    finally:
        co.stop.set()
        for sb,v,r in list(co.live.values()):
            if r['state']!='terminated':co.terminate(sb,r)
        co.checkpoint(force=True)
        atomic_json(folder/'REPORT.json',{'before':before,'after':co.store.status(),'estimated_cost_usd':co.cost(),'deadline_unix_s':deadline,'finished_unix_s':time.time(),'all_experiment_workers_stopped':all(r['state']=='terminated' for _,_,r in co.live.values())})
        subprocess.run(['sync','/workspace'],check=True)
if __name__=='__main__':main()
