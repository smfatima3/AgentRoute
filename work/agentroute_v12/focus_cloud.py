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
        excluded=[r[0] for r in db.execute('SELECT id,spec FROM tasks') if json.loads(r[1]).get('dataset') in ['svamp','math']]
        db.executemany("UPDATE tasks SET state='cancelled_by_user' WHERE id=? AND state!='accepted'",[(x,) for x in excluded])
        db.execute('INSERT INTO events(unix_s,kind,details) VALUES(?,?,?)',(time.time(),'user_scope_amendment',json.dumps({'user_request':'Complete BBH, ablations and remaining experiments; leave MATH. Prior SVAMP cancellation retained.','excluded_task_ids':excluded})))
    atomic_json(OUT/'ACTIVE_SCOPE.json',{'excluded_datasets':['svamp','math'],'excluded_task_ids':excluded,'original_expected_tasks':3013,'revised_expected_tasks':3013-len(excluded),'preserved_prior_results':True,'math_based_live_anchors':'excluded with MATH; not counted complete'})
    # Reuse the already verified development recovery test, not a new science result.
    proofs=[p for p in (OUT/'runs').glob('*/RECOVERY_TEST.json') if json.loads(p.read_text()).get('status')=='PASSED']
    if not proofs:raise RuntimeError('No prior successful recovery validation')
    import run_campaign as module
    source=(ROOT/'run_campaign.py').read_text()
    source=source.replace("available=gpu['start']+gpu['timeout']-time.time()-180","available=gpu['start']+gpu['timeout']-time.time()-15")
    source=source.replace('if available<300:return','if available<25:return')
    source=source.replace("self.store.claim(label,QUAL,size=min(10,max(1,int(available/240))))","self.store.claim(label,self.slot_groups[(gpu['label'],slot)],size=min(5,max(1,int(available/120))))")
    amended=ROOT/('short_controller_'+stamp+'.py');amended.write_text(source)
    import importlib.util
    spec=importlib.util.spec_from_file_location('short_controller',amended);runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    controls=['component_ablations','stage_ablations','aggregation_controls','seed_sensitivity','self_consistency','selforg_variant']
    co=runtime.Coordinator();co.slot_groups={('GPU-A',0):['short_bbh'],('GPU-B',0):['short_bbh'],('GPU-A',1):controls,('GPU-B',1):controls}
    # Account for all short-window coordinator allocations, including their failed startup costs.
    assigned=pathlib.Path('/workspace/FOCUS_ASSIGNED.json')
    until=time.time()+120
    while not assigned.exists() or json.loads(assigned.read_text()).get('deadline_unix_s')!=deadline:
        if time.time()>until:raise RuntimeError('Focus coordinator identity not supplied')
        time.sleep(1)
    identity=json.loads(assigned.read_text())
    for r in json.loads(pathlib.Path('/opt/prior_coordinators.json').read_text()):
        if r['id'].startswith('focus-coordinator-reservation-'):r['id']=identity['id']
        if not any(x['id']==r['id'] for x in co.registry['resources']):co.registry['resources'].append(r)
    co.persist();co.checkpoint(force=True)
    before=co.store.status();atomic_json(folder/'PLAN.json',{'deadline_unix_s':deadline,'before':before,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'reused_recovery_validation':str(proofs[-1]),'physical_transport':'BLOCKED_NOT_IMPLEMENTED_VALIDATED','live_anchors':'BLOCKED_NOT_IMPLEMENTED_VALIDATED'})
    cutoff=threading.Timer(max(0,deadline-time.time()-10),co.stop.set);cutoff.daemon=True;cutoff.start()
    def checkpoints():
        while not co.stop.wait(300):
            try:co.checkpoint()
            except Exception as exc:
                atomic_json(folder/'CHECKPOINT_FAILURE.json',{'error':str(exc)});co.stop.set()
    threading.Thread(target=checkpoints,daemon=True).start()
    try:
        co.build()
        seconds=min(3600,int(deadline-time.time()),int(max(0,28-co.cost(True)-.5)/(2*runtime.RATE['gpu']+4*runtime.RATE['client'])*3600))
        if seconds<300:raise RuntimeError('Remaining credit cannot safely reserve startup and concurrent work')
        allocation_cutoff=threading.Timer(seconds-20,co.stop.set);allocation_cutoff.daemon=True;allocation_cutoff.start()
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
            # Freed slots can finish the other ready family without changing inference or task identity.
            if not co.stop.is_set():
                co.slot_groups={key:['short_bbh']+controls for key in co.slot_groups}
                for f in [pool.submit(co.worker,g,s) for s in range(2) for g in gpu]:f.result()
    except Exception as exc:atomic_json(folder/'ERROR.json',{'type':type(exc).__name__,'message':str(exc)[:2000]})
    finally:
        co.stop.set()
        for sb,v,r in list(co.live.values()):
            if r['state']!='terminated':co.terminate(sb,r)
        co.checkpoint(force=True)
        atomic_json(folder/'REPORT.json',{'before':before,'after':co.store.status(),'estimated_cost_usd':co.cost(),'deadline_unix_s':deadline,'finished_unix_s':time.time(),'all_experiment_workers_stopped':all(r['state']=='terminated' for _,_,r in co.live.values())})
        subprocess.run(['sync','/workspace'],check=True)
if __name__=='__main__':main()
