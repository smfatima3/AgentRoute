"""Apply an explicit user scope change to the live coordinator's local transactional store."""
import json,time
from pathlib import Path
import modal
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/agentroute_v12_completion'
REMOTE=r'''
import json,sqlite3,time,pathlib,hashlib,subprocess
import modal
p=pathlib.Path('/workspace/outputs/agentroute_v12_completion')
stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());folder=p/'scope_amendments'/('remove-svamp-'+stamp);folder.mkdir(parents=True)
db=sqlite3.connect(p/'coordinator.sqlite',timeout=60);db.row_factory=sqlite3.Row
db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
rows=[dict(r) for r in db.execute("SELECT id,state,lease_id,spec FROM tasks WHERE group_name='transfer_svamp'")]
if len(rows)!=480:raise RuntimeError('Unexpected SVAMP scope')
before={s:sum(r['state']==s for r in rows) for s in {r['state'] for r in rows}}
leases={r['lease_id'] for r in rows if r['state']=='leased'}
amendment={'authorization':'User: stop SVAMP dataset, BBH and MATH with MMLU are enough.',
 'created_unix_s':time.time(),'original_manifest_sha256':hashlib.sha256((p/'campaign.lock.json').read_bytes()).hexdigest(),
 'removed_dataset':'svamp','excluded_task_ids':[r['id'] for r in rows],
 'original_expected_tasks':3013,'revised_expected_tasks':2533,'before':before,
 'preserve_completed_evidence_and_all_costs':True,'active_lease_ids':sorted(leases)}
(folder/'AMENDMENT.json').write_text(json.dumps(amendment,indent=2))
db.execute("UPDATE tasks SET state='cancelled_by_user' WHERE group_name='transfer_svamp' AND state!='accepted'")
db.execute('INSERT INTO events(unix_s,kind,details) VALUES(?,?,?)',(time.time(),'user_scope_amendment',json.dumps(amendment)))
db.commit();db.close()
original=json.loads((p/'campaign.lock.json').read_text());effective=dict(original)
effective['tasks']=[t for t in original['tasks'] if t['dataset']!='svamp']
effective['exact_counts']=dict(original['exact_counts']);effective['exact_counts'].pop('transfer_svamp')
effective['scope_amendment']=amendment;effective['original_manifest_retained']='campaign.lock.json'
(folder/'campaign.effective.json').write_text(json.dumps(effective,indent=2))
(p/'ACTIVE_SCOPE.json').write_text(json.dumps({'amendment':str(folder/'AMENDMENT.json'),'effective_manifest':str(folder/'campaign.effective.json'),'excluded_datasets':['svamp'],'expected_tasks':2533},indent=2))
stopped=[];deadline=time.time()+150
while leases and time.time()<deadline:
 registry=json.loads((p/'resource_registry_live.json').read_text())
 for r in registry['resources']:
  if r.get('lease_id') not in leases:continue
  if not set(r.get('task_ids',[])).issubset(set(amendment['excluded_task_ids'])):raise RuntimeError('Mixed dataset worker: do not terminate')
  sb=modal.Sandbox.from_id(r['id'])
  if sb.poll() is None:
   try:
    proc=sb.exec('sync','/evidence',timeout=30);proc.wait()
   except Exception:pass
   sb.terminate(wait=True)
  code=sb.poll()
  if code is None:raise RuntimeError('SVAMP worker shutdown unconfirmed')
  stopped.append({'id':r['id'],'lease_id':r['lease_id'],'provider_exit_code':code,'volume_id':r['volume_id']});leases.remove(r['lease_id'])
 if leases:time.sleep(2)
report={'scope_change':amendment,'stopped_svamp_workers':stopped,'unresolved_leases':sorted(leases),'other_datasets_interrupted':False}
db=sqlite3.connect(p/'coordinator.sqlite',timeout=60)
report['svamp_states']=[list(r) for r in db.execute("SELECT state,count(*) FROM tasks WHERE group_name='transfer_svamp' GROUP BY state")]
report['current_counts']=[list(r) for r in db.execute('SELECT group_name,state,count(*) FROM tasks GROUP BY group_name,state')];db.close()
(folder/'VERIFICATION.json').write_text(json.dumps(report,indent=2));subprocess.run(['sync','/workspace'],check=True,timeout=60)
print(json.dumps(report),flush=True)
if leases:raise RuntimeError('SVAMP worker creation race needs further inspection')
'''
def main():
    state=json.loads((OUT/'CLOUD_COORDINATOR.json').read_text());sb=modal.Sandbox.from_id(state['sandbox_id'])
    if sb.poll() is not None:
        from inspect_cloud import main as inspect
        inspect()
        print('Coordinator stopped; scope change recorded locally for next resume.',flush=True)
        (OUT/'PENDING_SCOPE_CHANGE.json').write_text(json.dumps({'excluded_datasets':['svamp'],'authorization':'User requested SVAMP removal','no_gpu_restart_authorized_by_this_status_check':True},indent=2))
        return
    proc=sb.exec('python','-c',REMOTE,timeout=300)
    out=proc.stdout.read();err=proc.stderr.read();code=proc.wait()
    if out:
        result=json.loads(out);(OUT/'SVAMP_STOP_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if code:raise RuntimeError(err[-2500:])
if __name__=='__main__':main()
