"""Single coordinator SQLite leases and verified receipts. No shared-volume DB."""
import contextlib,hashlib,json,os,sqlite3,time,uuid
from pathlib import Path

def hash_file(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

class Store:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, group_name TEXT, fingerprint TEXT, spec TEXT, bucket TEXT, priority TEXT, state TEXT DEFAULT 'pending', epoch INTEGER DEFAULT 0, attempts INTEGER DEFAULT 0, lease_id TEXT, receipt TEXT);
            CREATE TABLE IF NOT EXISTS leases(id TEXT PRIMARY KEY, epoch INTEGER, worker TEXT, state TEXT, started REAL, heartbeat REAL, expires REAL, provider_exit_code INTEGER, ended REAL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, unix_s REAL, kind TEXT, details TEXT);
            CREATE TABLE IF NOT EXISTS attempt_receipts(task_id TEXT, lease_id TEXT, receipt TEXT, accepted INTEGER, PRIMARY KEY(task_id,lease_id));
            ''')
    @contextlib.contextmanager
    def connection(self):
        db=sqlite3.connect(self.path,timeout=60);db.row_factory=sqlite3.Row;db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
        try:yield db;db.commit()
        except:db.rollback();raise
        finally:db.close()
    def initialize(self,tasks):
        with self.connection() as db:
            for task in tasks:
                old=db.execute('SELECT fingerprint FROM tasks WHERE id=?',(task['task_id'],)).fetchone()
                if old and old[0]!=task['fingerprint']:raise ValueError('Manifest task changed')
                bucket='/'.join(str(task.get(k,'')) for k in ['group','dataset','method','seed','agents'])
                tier='0' if task['group']=='main_mmlu' else '1' if task['group'] in ['transfer_svamp','short_math','short_bbh'] else '2'
                priority=tier+hashlib.sha256(('20261031/'+bucket).encode()).hexdigest()
                db.execute('INSERT OR IGNORE INTO tasks(id,group_name,fingerprint,spec,bucket,priority) VALUES(?,?,?,?,?,?)',
                  (task['task_id'],task['group'],task['fingerprint'],json.dumps(task),bucket,priority))
    def claim(self,worker,allowed_groups,size=10,ttl=900):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE');marks=','.join('?' for _ in allowed_groups)
            first=db.execute(f"SELECT * FROM tasks WHERE state='pending' AND group_name IN ({marks}) ORDER BY priority,id LIMIT 1",allowed_groups).fetchone()
            if not first:return None
            rows=db.execute("SELECT * FROM tasks WHERE state='pending' AND bucket=? ORDER BY id LIMIT ?",(first['bucket'],size)).fetchall()
            lease='lease-'+uuid.uuid4().hex;epoch=db.execute('SELECT COALESCE(MAX(epoch),0)+1 FROM leases').fetchone()[0];now=time.time()
            db.execute('INSERT INTO leases(id,epoch,worker,state,started,heartbeat,expires) VALUES(?,?,?,?,?,?,?)',(lease,epoch,worker,'active',now,now,now+ttl))
            for row in rows:db.execute("UPDATE tasks SET state='leased',lease_id=?,epoch=?,attempts=attempts+1 WHERE id=?",(lease,epoch,row['id']))
            db.execute('INSERT INTO events(unix_s,kind,details) VALUES(?,?,?)',(now,'claim',json.dumps({'lease':lease,'worker':worker,'epoch':epoch,'tasks':[r['id'] for r in rows]})))
            return {'id':lease,'epoch':epoch,'worker':worker,'tasks':[json.loads(r['spec']) for r in rows]}
    def heartbeat(self,lease,epoch,ttl=900):
        with self.connection() as db:
            cur=db.execute("UPDATE leases SET heartbeat=?,expires=? WHERE id=? AND epoch=? AND state='active'",(time.time(),time.time()+ttl,lease,epoch))
            if cur.rowcount!=1:raise ValueError('Lease fenced')
    def accept(self,receipt,root,reuse=False):
        root=Path(root).resolve()
        for evidence in receipt['evidence']:
            path=(root/evidence['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file() or hash_file(path)!=evidence['sha256']:raise ValueError('Receipt artifact missing/corrupt')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE');task=db.execute('SELECT * FROM tasks WHERE id=?',(receipt['task_id'],)).fetchone()
            if not task or task['fingerprint']!=receipt['fingerprint'] or task['group_name']!=receipt['group']:raise ValueError('Receipt identity mismatch')
            if task['state']=='accepted':
                if json.loads(task['receipt'])==receipt:return False
                raise ValueError('Duplicate scientific outcome; retain in attempt ledger only')
            if not reuse:
                if receipt['epoch']!=task['epoch']:raise ValueError('Stale epoch receipt')
                lease=db.execute('SELECT state FROM leases WHERE id=?',(task['lease_id'],)).fetchone()
                if not lease or lease['state'] not in ['active','terminated']:raise ValueError('Invalid lease')
            if receipt['status'] not in ['measured','verified_reuse']:raise ValueError('Infrastructure attempt cannot be accepted as a quality outcome')
            raw=json.dumps(receipt);db.execute("UPDATE tasks SET state='accepted',receipt=? WHERE id=?",(raw,task['id']))
            db.execute('INSERT INTO events(unix_s,kind,details) VALUES(?,?,?)',(time.time(),'accept',raw))
            return True
    def confirmed_dead(self,lease,epoch,provider_exit_code,retry=True):
        if provider_exit_code is None:raise ValueError('Expired heartbeat is not proof of death')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE');r=db.execute('SELECT * FROM leases WHERE id=? AND epoch=?',(lease,epoch)).fetchone()
            if not r:raise ValueError('Unknown lease epoch')
            db.execute("UPDATE leases SET state='terminated',provider_exit_code=?,ended=? WHERE id=?",(provider_exit_code,time.time(),lease))
            db.execute("UPDATE tasks SET state=CASE WHEN attempts<3 AND ? THEN 'pending' ELSE 'blocked' END WHERE lease_id=? AND state='leased'",(int(retry),lease))
    def status(self):
        with self.connection() as db:
            counts=[dict(r) for r in db.execute('SELECT group_name,state,count(*) AS count FROM tasks GROUP BY group_name,state')]
            active=[dict(r) for r in db.execute("SELECT * FROM leases WHERE state='active'")]
            return dict(observed_unix_s=time.time(),counts=counts,active_leases=active)
    def accepted(self):
        with self.connection() as db:return [json.loads(r[0]) for r in db.execute("SELECT receipt FROM tasks WHERE state='accepted' ORDER BY id")]
    def snapshot(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as src,sqlite3.connect(path) as dst:src.backup(dst)
