"""Cooperative application-state migration and measured HTTP mailbox delivery.

This is a transport prototype. It migrates mailbox/workflow state, not model
weights, processes, CUDA contexts, or KV caches. It provides idempotent receipt
within retained state, not exactly-once external side effects or crash recovery.
"""
import hashlib,json,sqlite3,threading,time,urllib.error,urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def encoded(value):return json.dumps(value,sort_keys=True,ensure_ascii=False).encode()
def checksum(value):return hashlib.sha256(encoded(value)).hexdigest()

class Worker:
    def __init__(self,path):
        self.db=sqlite3.connect(path,check_same_thread=False);self.lock=threading.RLock()
        self.db.execute('CREATE TABLE IF NOT EXISTS actors (id TEXT PRIMARY KEY, state TEXT)');self.db.commit()
    def operate(self,op,data):
        with self.lock,self.db:
            id=data['agent_id'];row=self.db.execute('SELECT state FROM actors WHERE id=?',(id,)).fetchone()
            state=json.loads(row[0]) if row else None
            if op=='install':
                if state:raise ValueError('Actor already exists on this worker')
                state={'epoch':data.get('epoch',0),'status':'ACTIVE','inbox':{},'receipts':{},
                       'workflow':data.get('workflow',{}),'redirect':None}
                result={'status':'INSTALLED'}
            elif op=='prepare':
                if state and state['status']=='ACTIVE':raise ValueError('Destination actor is active')
                state=data['snapshot'];state['epoch']=data['epoch'];state['status']='READY';state['redirect']=None
                result={'status':'READY','state_hash':checksum(state)}
            elif not state:raise KeyError('Unknown actor')
            elif op=='deliver':
                if state.get('redirect'):return {'status':'REDIRECT',**state['redirect']}
                if state['status']!='ACTIVE':return {'status':'RETRY','reason':state['status']}
                if data['epoch']!=state['epoch']:return {'status':'STALE','epoch':state['epoch']}
                message=data['message'];mid=message['id'];h=checksum(message)
                if mid in state['receipts']:
                    if state['receipts'][mid]!=h:raise ValueError('Message ID reused with a different payload')
                    return {'status':'RECEIVED','id':mid,'duplicate':True}
                state['inbox'][mid]=message;state['receipts'][mid]=h
                result={'status':'RECEIVED','id':mid,'duplicate':False}
            elif op=='freeze':
                if state['status']!='ACTIVE' or state['epoch']!=data['epoch']:raise ValueError('Invalid freeze epoch')
                state['status']='FROZEN';result={'status':'FROZEN','snapshot':state.copy()}
            elif op=='activate':
                if state['status']!='READY' or state['epoch']!=data['epoch']:raise ValueError('Invalid activation')
                state['status']='ACTIVE';result={'status':'ACTIVE'}
            elif op=='redirect':
                if state['status']!='FROZEN':raise ValueError('Old worker must remain fenced')
                state['redirect']={'endpoint':data['endpoint'],'epoch':data['epoch']};result={'status':'FENCED'}
            elif op=='inspect':return state
            else:raise ValueError('Unknown operation')
            self.db.execute('INSERT OR REPLACE INTO actors VALUES (?,?)',(id,encoded(state).decode()))
            return result

def serve_worker(path,host,port):
    worker=Worker(path)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length>32*1024*1024:raise ValueError('Body too large')
                request=json.loads(self.rfile.read(length));result=worker.operate(self.path.strip('/'),request)
                body=encoded(result);self.send_response(200)
            except Exception as exc:
                body=encoded({'error':type(exc).__name__});self.send_response(400)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer((host,port),Handler);server.serve_forever()

class Manager:
    def __init__(self,path,location_aware=True,buffering=True,max_queue=1000,max_hops=4,timeout=30):
        self.db=sqlite3.connect(path,check_same_thread=False);self.lock=threading.RLock()
        self.db.execute('CREATE TABLE IF NOT EXISTS directory (id TEXT PRIMARY KEY, endpoint TEXT, epoch INT, status TEXT, origin TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS queue (id TEXT PRIMARY KEY, agent TEXT, message TEXT, deadline REAL)');self.db.commit()
        self.location_aware=location_aware;self.buffering=buffering;self.max_queue=max_queue;self.max_hops=max_hops;self.timeout=timeout
        self.events=[];self.sender_pending={}
    def rpc(self,endpoint,op,data):
        body=encoded(data);start=time.perf_counter();record={'operation':op,'endpoint':endpoint,
            'request_body_bytes':len(body),'response_body_bytes':None,'status':'error'}
        try:
            request=urllib.request.Request(endpoint.rstrip('/')+'/'+op,body,{'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=self.timeout) as response:raw=response.read()
            record['response_body_bytes']=len(raw);record['status']='ok';return json.loads(raw)
        finally:record['wall_s']=time.perf_counter()-start;self.events.append(record)
    def register(self,agent_id,endpoint,workflow=None):
        if self.db.execute('SELECT 1 FROM directory WHERE id=?',(agent_id,)).fetchone():raise ValueError('Already registered')
        self.rpc(endpoint,'install',{'agent_id':agent_id,'epoch':0,'workflow':workflow or {}})
        with self.db:self.db.execute('INSERT INTO directory VALUES (?,?,?,?,?)',(agent_id,endpoint,0,'ACTIVE',endpoint))
    def location(self,agent_id):
        row=self.db.execute('SELECT endpoint,epoch,status,origin FROM directory WHERE id=?',(agent_id,)).fetchone()
        if not row:raise KeyError(agent_id)
        return dict(zip(['endpoint','epoch','status','origin'],row))
    def send(self,agent_id,message,deadline=None):
        deadline=deadline if deadline is not None else time.time()+self.timeout
        if time.time()>deadline:raise TimeoutError('Message deadline expired')
        with self.lock:
            loc=self.location(agent_id)
            if loc['status']=='MIGRATING':
                if self.buffering:
                    existing=self.db.execute('SELECT message,agent FROM queue WHERE id=?',(message['id'],)).fetchone()
                    if existing:
                        if existing!=(encoded(message).decode(),agent_id):raise ValueError('Conflicting queued message ID')
                        return {'status':'BUFFERED','id':message['id']}
                    if self.db.execute('SELECT COUNT(*) FROM queue').fetchone()[0]>=self.max_queue:raise BufferError('Queue limit')
                    with self.db:self.db.execute('INSERT INTO queue VALUES (?,?,?,?)',(message['id'],agent_id,encoded(message).decode(),deadline))
                    return {'status':'BUFFERED','id':message['id']}
                self.sender_pending[message['id']]=(agent_id,message,deadline)
                return {'status':'RETRY','reason':'MIGRATING'}
            if loc['status']!='ACTIVE':raise RuntimeError('Agent unavailable')
        endpoint=loc['endpoint'] if self.location_aware else loc['origin'];epoch=loc['epoch']
        for _ in range(self.max_hops):
            if time.time()>deadline:raise TimeoutError('Delivery deadline expired')
            result=self.rpc(endpoint,'deliver',{'agent_id':agent_id,'epoch':epoch,'message':message})
            if result['status']=='RECEIVED':return result
            if result['status']=='REDIRECT':endpoint=result['endpoint'];epoch=result['epoch']
            elif result['status']=='STALE':
                loc=self.location(agent_id);endpoint=loc['endpoint'];epoch=loc['epoch']
            else:raise RuntimeError('Receiver unavailable; retain original message for retry')
        raise RuntimeError('Routing hop bound exceeded')
    def migrate(self,agent_id,destination,after_freeze=None):
        with self.lock,self.db:
            loc=self.location(agent_id)
            if loc['status']!='ACTIVE' or loc['endpoint']==destination:raise ValueError('Invalid migration')
            self.db.execute('UPDATE directory SET status=? WHERE id=?',('MIGRATING',agent_id))
        start=time.perf_counter();new_epoch=loc['epoch']+1
        try:
            snapshot=self.rpc(loc['endpoint'],'freeze',{'agent_id':agent_id,'epoch':loc['epoch']})['snapshot']
            if after_freeze:after_freeze()
            self.rpc(destination,'prepare',{'agent_id':agent_id,'epoch':new_epoch,'snapshot':snapshot})
            self.rpc(destination,'activate',{'agent_id':agent_id,'epoch':new_epoch})
            # Destination READY/ACTIVE plus old FROZEN precede publishing ownership.
            with self.lock,self.db:
                changed=self.db.execute('UPDATE directory SET endpoint=?,epoch=?,status=? WHERE id=? AND epoch=? AND status=?',
                    (destination,new_epoch,'ACTIVE',agent_id,loc['epoch'],'MIGRATING')).rowcount
                if changed!=1:raise RuntimeError('Migration ownership conflict')
            self.rpc(loc['endpoint'],'redirect',{'agent_id':agent_id,'endpoint':destination,'epoch':new_epoch})
            self.drain(agent_id)
            return {'agent_id':agent_id,'from':loc['endpoint'],'to':destination,'epoch':new_epoch,
                    'migration_wall_s':time.perf_counter()-start,'state_bytes':len(encoded(snapshot))}
        except Exception:
            # Conservative: no automatic rollback once readiness/activation may
            # have occurred. Operator recovery is outside this prototype's scope.
            with self.db:self.db.execute('UPDATE directory SET status=? WHERE id=?',('UNAVAILABLE',agent_id))
            raise
    def drain(self,agent_id):
        pending=self.db.execute('SELECT id,message,deadline FROM queue WHERE agent=? ORDER BY rowid',(agent_id,)).fetchall()
        for mid,raw,deadline in pending:
            receipt=self.send(agent_id,json.loads(raw),deadline)
            if receipt['status']!='RECEIVED':raise RuntimeError('Buffer drain did not receive acknowledgment')
            with self.db:self.db.execute('DELETE FROM queue WHERE id=?',(mid,))
        for mid,(target,message,deadline) in list(self.sender_pending.items()):
            if target==agent_id:
                receipt=self.send(target,message,deadline)
                if receipt['status']=='RECEIVED':del self.sender_pending[mid]
