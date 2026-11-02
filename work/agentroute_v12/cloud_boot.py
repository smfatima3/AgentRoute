"""Cloud-resident supervisor; no Codex automation or local polling dependency."""
import hashlib,json,os,pathlib,shutil,subprocess,sys,time,zipfile
ROOT=pathlib.Path('/workspace');OUT=ROOT/'outputs/agentroute_v12_completion'
def save(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def sync():subprocess.run(['sync','/workspace'],check=True,timeout=120)
def safe_destination(root,name):
    root=pathlib.Path(root).resolve()
    p=(root/name).resolve()
    if not p.is_relative_to(root):raise RuntimeError('Seed path traversal')
    return p
def main():
    marker=ROOT/'SEED_RESTORED.json'
    if not marker.exists():
        with zipfile.ZipFile('/opt/campaign_seed.zip') as z:
            manifest=json.loads(z.read('SEED_MANIFEST.json'))
            for name,h in manifest.items():
                p=safe_destination(ROOT,name)
                raw=z.read(name)
                if hashlib.sha256(raw).hexdigest()!=h:raise RuntimeError('Seed hash mismatch')
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        save(marker,{'restored_unix_s':time.time(),'verified_files':len(manifest)});sync()
    # The output is a single-writer Volume. Raw imported evidence is never rewritten.
    latest=json.loads((OUT/'preparation_latest.json').read_text())
    name=latest['path'].replace('\\','/').split('/')[-1]
    save(OUT/'preparation_latest.json',{'path':str(OUT/'preparation'/name),'original_local_path':latest['path']})
    assigned=ROOT/'COORDINATOR_ASSIGNED.json'
    deadline=time.monotonic()+180
    while not assigned.exists() or json.loads(assigned.read_text()).get('launched_unix_s')!=float(os.environ['V12_COORDINATOR_STARTED']):
        if time.monotonic()>deadline:raise RuntimeError('Launcher did not persist coordinator identity; no GPU started')
        time.sleep(1)
    os.environ['V12_COORDINATOR_ID']=json.loads(assigned.read_text())['sandbox_id']
    os.environ['V12_COORDINATOR_STARTED']=os.environ.get('V12_COORDINATOR_STARTED',str(time.time()))
    patchpath=pathlib.Path('/opt/runtime_patch.json')
    if patchpath.exists():
        patch=json.loads(patchpath.read_text());version=hashlib.sha256(patchpath.read_bytes()).hexdigest()
        history=OUT/'runtime_versions'/version;history.mkdir(parents=True,exist_ok=True)
        for name,entry in patch['files'].items():
            if '/' in name or '\\' in name or not name.endswith('.py'):raise RuntimeError('Invalid operational patch path')
            target=ROOT/'work/agentroute_v12'/name
            if hashlib.sha256(entry['text'].encode()).hexdigest()!=entry['sha256']:raise RuntimeError('Operational patch hash mismatch')
            before=history/(name+'.before')
            if target.exists() and not before.exists():shutil.copyfile(target,before)
            (history/name).write_text(entry['text']);target.write_text(entry['text'])
        save(history/'PATCH.json',patch)
        previous=json.loads((OUT/'RUNTIME_LINEAGE.json').read_text()) if (OUT/'RUNTIME_LINEAGE.json').exists() else None
        save(OUT/'RUNTIME_LINEAGE.json',{'runtime_version_sha256':version,'reason':patch['reason'],'previous':previous,
          'scientific_policy_changed':False,'source_files':{n:e['sha256'] for n,e in patch['files'].items()},'applied_unix_s':time.time()})
        registry=pathlib.Path('/opt/provider_registry.json')
        if registry.exists():save(OUT/'resource_registry_live.json',json.loads(registry.read_text()))
        sync()
    scopepath=pathlib.Path('/opt/scope_change.json')
    if scopepath.exists():
        import sqlite3
        scope=json.loads(scopepath.read_text())
        if scope['excluded_datasets']!=['svamp']:raise RuntimeError('Unexpected scope amendment')
        with sqlite3.connect(OUT/'coordinator.sqlite',timeout=60) as db:
            db.execute('BEGIN IMMEDIATE')
            excluded=[r[0] for r in db.execute("SELECT id FROM tasks WHERE group_name='transfer_svamp'")]
            db.execute("UPDATE tasks SET state='cancelled_by_user' WHERE group_name='transfer_svamp' AND state!='accepted'")
            scope.update(excluded_task_ids=excluded,original_expected_tasks=3013,revised_expected_tasks=2533,applied_unix_s=time.time())
            db.execute('INSERT INTO events(unix_s,kind,details) VALUES(?,?,?)',(time.time(),'user_scope_amendment',json.dumps(scope)))
        save(OUT/'ACTIVE_SCOPE.json',scope);sync()
    sys.path.insert(0,str(ROOT/'work/agentroute_v12'))
    from coordinator_store import Store
    from run_campaign import QUAL
    save(OUT/'CLOUD_STATUS.json',{'state':'STARTING','coordinator_id':os.environ['V12_COORDINATOR_ID'],'checkpoint_interval_s':1200,'status_interval_s':300,
      'independent_of_codex':True,'volume_id':os.environ['V12_COORDINATOR_VOLUME'],'user_budget_usd':30,'execution_ceiling_usd':28,'reserved_usd':2})
    sync()
    for epoch in range(12):
        with (OUT/'cloud_controller.log').open('a',buffering=1) as log:
            proc=subprocess.Popen([sys.executable,str(ROOT/'work/agentroute_v12/run_campaign.py')],stdout=log,stderr=subprocess.STDOUT)
            save(OUT/'CLOUD_STATUS.json',{'state':'RUNNING','epoch':epoch,'controller_pid':proc.pid,'coordinator_id':os.environ['V12_COORDINATOR_ID'],
               'checkpoint_interval_s':1200,'status_interval_s':300,'independent_of_codex':True,'volume_id':os.environ['V12_COORDINATOR_VOLUME']});sync()
            code=proc.wait()
        sync();status=Store(OUT/'coordinator.sqlite').status()
        pending=sum(r['count'] for r in status['counts'] if r['group_name'] in QUAL and r['state']=='pending')
        if code!=0 or pending==0:
            save(OUT/'CLOUD_STATUS.json',{'state':'INCOMPLETE_REQUIRES_ATTENTION' if code else 'QUALITY_QUEUE_DRAINED_OTHER_BLOCKS_PENDING',
              'epoch':epoch,'controller_exit_code':code,'pending_quality_tasks':pending,'queue':status,'last_update_unix_s':time.time(),
              'coordinator_id':os.environ['V12_COORDINATOR_ID'],'checkpoint_interval_s':1200,'independent_of_codex':True});sync();return
        # Normal lifetime expiry can resume missing tasks; credit reservations still apply to every new resource.
    save(OUT/'CLOUD_STATUS.json',{'state':'INCOMPLETE_BOUNDED_EPOCH_LIMIT','unix_s':time.time()});sync()
if __name__=='__main__':main()
