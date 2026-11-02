"""Durable process-safe reservations; unknown costs remain charged conservatively."""
import json,sqlite3,time
from pathlib import Path

class BudgetExceeded(RuntimeError):pass

class Budget:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,phase TEXT,status TEXT,started REAL,finished REAL);
            CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY,attempt_id TEXT,phase TEXT,category TEXT,status TEXT,
            reserved INTEGER,used INTEGER,started REAL,finished REAL,ledger TEXT);''')
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA busy_timeout=30000');return db
    def initialize(self,settings):
        with self.connect() as db:
            for k,v in settings.items():
                old=db.execute('SELECT value FROM settings WHERE key=?',(k,)).fetchone();value=json.dumps(v,sort_keys=True)
                if old and old[0]!=value:raise RuntimeError('Budget settings are immutable')
                db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)',(k,value))
    def settings(self,db):return {k:json.loads(v) for k,v in db.execute('SELECT key,value FROM settings')}
    def _deadline(self,s):
        if time.time()>=s['admission_deadline_unix']:raise BudgetExceeded('Admission deadline reached; reserve shutdown/export time')
    def start_attempt(self,id,phase):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');s=self.settings(db);self._deadline(s)
            if db.execute('SELECT id FROM attempts WHERE id=?',(id,)).fetchone():raise RuntimeError('Attempt already recorded; no implicit retry')
            n=db.execute("SELECT COUNT(*) FROM attempts WHERE phase!='fit' AND phase!='warmup'").fetchone()[0]
            if phase not in ['fit','warmup'] and n>=s['scored_attempt_cap']:raise BudgetExceeded('Scored-attempt cap')
            if phase=='smoke' and db.execute("SELECT COUNT(*) FROM attempts WHERE phase='smoke'").fetchone()[0]>=60:raise BudgetExceeded('Smoke attempt cap')
            db.execute('INSERT INTO attempts VALUES (?,?,?, ?,NULL)',(id,phase,'started',time.time()))
    def finish_attempt(self,id,status):
        with self.connect() as db:db.execute('UPDATE attempts SET status=?,finished=? WHERE id=?',(status,time.time(),id))
    def reserve_call(self,id,attempt,phase,category,reservation=32768):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');s=self.settings(db);self._deadline(s)
            total=db.execute('SELECT COALESCE(SUM(COALESCE(used,reserved)),0) FROM calls').fetchone()[0]
            fit=db.execute("SELECT COALESCE(SUM(COALESCE(used,reserved)),0) FROM calls WHERE phase='fit'").fetchone()[0]
            if total+reservation>s['total_token_cap'] or (phase=='fit' and fit+reservation>s['fit_token_cap']):raise BudgetExceeded('Token reservation cap')
            db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,NULL,?,NULL,NULL)',(id,attempt,phase,category,'reserved',reservation,time.time()))
    def finish_call(self,id,record):
        used=None
        if record.get('prompt_tokens') is not None and record.get('completion_tokens') is not None:
            used=record['prompt_tokens']+record['completion_tokens']
        with self.connect() as db:
            reservation=db.execute('SELECT reserved FROM calls WHERE id=?',(id,)).fetchone()[0]
            db.execute('UPDATE calls SET status=?,used=?,finished=?,ledger=? WHERE id=?',
                (record['status'],used,time.time(),json.dumps(record,ensure_ascii=False),id))
        if used is not None and used>reservation:raise BudgetExceeded('Authoritative usage exceeded reservation; halt campaign')
    def snapshot(self):
        with self.connect() as db:
            return {'settings':self.settings(db),'attempts_by_phase_status':[dict(phase=p,status=s,count=n) for p,s,n in db.execute('SELECT phase,status,COUNT(*) FROM attempts GROUP BY phase,status')],
                'calls':db.execute('SELECT COUNT(*) FROM calls').fetchone()[0],
                'authoritative_tokens':db.execute('SELECT COALESCE(SUM(used),0) FROM calls').fetchone()[0],
                'charged_tokens_including_outstanding':db.execute('SELECT COALESCE(SUM(COALESCE(used,reserved)),0) FROM calls').fetchone()[0],
                'unknown_or_outstanding_calls':db.execute('SELECT COUNT(*) FROM calls WHERE used IS NULL').fetchone()[0],
                'fit_charged_tokens':db.execute("SELECT COALESCE(SUM(COALESCE(used,reserved)),0) FROM calls WHERE phase='fit'").fetchone()[0]}
