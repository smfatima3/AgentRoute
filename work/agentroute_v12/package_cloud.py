"""Version the unused local-controller prototype, freeze cloud execution, package seed."""
import hashlib,json,pathlib,shutil,subprocess,sys,time,zipfile
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parents[1];OUT=BASE/'outputs/agentroute_v12_completion'
sys.path.insert(0,str(ROOT))
from coordinator_store import Store,atomic_json,hash_file
def main():
    receipts=[json.loads(x) for x in (OUT/'results_index.jsonl').read_text().splitlines()]
    if any(x['status']!='verified_reuse' for x in receipts):raise RuntimeError('New model results exist; explicit scientific lineage required')
    failed=None
    if (OUT/'CLOUD_COORDINATOR.json').exists():
        if '--after-failed-bootstrap' not in sys.argv:raise RuntimeError('Cloud already launched; do not reseed its queue')
        old=json.loads((OUT/'CLOUD_COORDINATOR.json').read_text())
        observations=[json.loads(p.read_text()) for p in sorted((OUT/'cloud_observations').glob('*/observation.json'))]
        failed=next((r for r in reversed(observations) if r['coordinator_id']==old['sandbox_id']),None)
        if not failed or failed['provider_exit_code'] is None or failed.get('STATUS.json') is not None or failed.get('resource_registry_live.json') is not None:
            raise RuntimeError('Only a verified pre-controller bootstrap failure permits this reseed')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());history=OUT/'protocol_history'/('precloud-'+stamp);history.mkdir(parents=True)
    if failed:
        seed_history=BASE/'work/cloud_seed_history';seed_history.mkdir(exist_ok=True)
        seed=ROOT/'campaign_seed.zip';dest=seed_history/(hash_file(seed)+'.zip')
        if not dest.exists():shutil.copyfile(seed,dest)
        old=json.loads((OUT/'CLOUD_COORDINATOR.json').read_text())
        import calendar
        start=calendar.timegm(time.strptime(old['volume_name'].split('coordinator-')[1].upper(),'%Y%m%dT%H%M%SZ'))
        atomic_json(OUT/'resource_registry_live.json',{'campaign_id':'agentroute-v12-completion-20261031','resources':[
          {'id':old['sandbox_id'],'role':'coordinator','owned':True,'state':'terminated','provider_exit_code':failed['provider_exit_code'],
           'created_unix_s':start,'stopped_unix_s':failed['observed_unix_s'],'timeout_s':43200,'rate_usd_per_hour':.1899,
           'cost_mode':'conservative creation-to-confirmed-observation upper bound','volume_id':old['volume_id'],'operation':'failed-cloud-bootstrap'}]})
        (OUT/'CLOUD_COORDINATOR.json').replace(history/'original_registration.json')
    for name in ['campaign.lock.json','source_lock.json','protocol.json','STAGE_INVENTORY.json','results_index.jsonl','STATUS.json']:
        src=OUT/name;dest=history/name
        assert dest.resolve().is_relative_to(OUT.resolve())
        src.replace(dest)
    # No controller or worker has started; retain the earlier coordinator database as immutable history.
    Store(OUT/'coordinator.sqlite').snapshot(history/'coordinator_snapshot.sqlite')
    for name in ['coordinator.sqlite','coordinator.sqlite-wal','coordinator.sqlite-shm']:
        src=OUT/name
        if src.exists():src.replace(history/name)
    atomic_json(history/'LINEAGE.json',{'reason':'Repair mounted-path containment, explicit cloud identity registration and Python spawn entry guard after verified pre-controller bootstrap failure. No inference or GPUs started.' if failed else 'User explicitly requested cloud-resident coordinator/checkpoints, not Codex. No new inference or GPUs were started under prior prototype.',
       'prior_manifest_sha256':hash_file(history/'campaign.lock.json'),'policy_change':False,'new_heldout_runs':0})
    subprocess.run([sys.executable,str(ROOT/'freeze_campaign.py')],check=True)
    outzip=ROOT/'campaign_seed.zip';files=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='campaign_seed.zip':files.append(p)
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix not in ['.log','.err'] and not p.name.endswith(('-wal','-shm')):files.append(p)
    control=BASE/'work/v9_local/clean_release_check_final/agentroute_modal_codex_v9/modal_runner'
    files.extend(p for p in control.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest={}
    with zipfile.ZipFile(outzip,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            name=p.relative_to(BASE).as_posix();data=p.read_bytes();manifest[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        z.writestr('SEED_MANIFEST.json',json.dumps(manifest,sort_keys=True))
    with zipfile.ZipFile(outzip) as z:
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    atomic_json(OUT/'CLOUD_SEED_VERIFICATION.json',{'archive':str(outzip),'sha256':hash_file(outzip),'bytes':outzip.stat().st_size,'verified_files':len(manifest),
       'user_reported_credit_balance_usd':30,'execution_ceiling_usd':28,'checkpoint_location':'Modal persistent Volume; every 1200 seconds','no_codex_automation':True})
    print(json.dumps({'seed_bytes':outzip.stat().st_size,'verified_files':len(manifest),'sha256':hash_file(outzip)}))
if __name__=='__main__':main()
