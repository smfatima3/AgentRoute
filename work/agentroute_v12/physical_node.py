"""Physical HTTP payload replay. No model inference and no artificial network sleeps."""
import argparse,concurrent.futures,hashlib,http.server,json,math,os,pathlib,random,resource,socket,statistics,subprocess,threading,time
import http.client,urllib.parse

class Session:
 def __init__(self):self.connection=None;self.origin=None
 def __enter__(self):return self
 def __exit__(self,*a):self.close()
 def close(self):
  if self.connection:self.connection.close()
  self.connection=None
 def post(self,url,body,timeout):
  parsed=urllib.parse.urlsplit(url);origin=(parsed.scheme,parsed.hostname,parsed.port)
  if self.connection is None or self.origin!=origin:
   self.close();self.origin=origin
   cls=http.client.HTTPSConnection if parsed.scheme=='https' else http.client.HTTPConnection
   self.connection=cls(parsed.hostname,parsed.port,timeout=timeout)
  self.connection.timeout=timeout
  if self.connection.sock:self.connection.sock.settimeout(timeout)
  try:
   self.connection.request('POST',parsed.path or '/',body,{'Content-Type':'application/octet-stream'})
   r=self.connection.getresponse();raw=r.read()
   if r.status!=200:raise RuntimeError('HTTP status '+str(r.status))
   return raw
  except Exception:self.close();raise

def save(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix('.partial');q.write_text(json.dumps(x));q.replace(p)

class Handler(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 disable_nagle_algorithm=True
 def log_message(self,*a):pass
 def do_POST(self):
  body=self.rfile.read(int(self.headers.get('Content-Length','0')))
  response=json.dumps({'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),'received_unix_ns':time.time_ns()}).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(response)));self.end_headers();self.wfile.write(response)

def rpc(session,url,body,timeout=30):
 start=time.perf_counter();raw=session.post(url,body,timeout);ack=json.loads(raw)
 if ack['sha256']!=hashlib.sha256(body).hexdigest() or ack['bytes']!=len(body):raise RuntimeError('Payload integrity failure')
 return {'wall_s':time.perf_counter()-start,'request_body_bytes':len(body),'response_body_bytes':len(raw),'sha256':ack['sha256']}

def validate(url,out):
 with Session() as s:
  rpc(s,url,b'warmup')
  rtts=[rpc(s,url,b'ping')['wall_s'] for _ in range(30)]
  body=b'0'*(8*1024*1024)
  bulk=[rpc(s,url,body,60) for _ in range(3)]
 save(out,{'rtt_s':rtts,'rtt_median_s':statistics.median(rtts),'bulk':bulk,
    'upload_mbit_s_median':statistics.median(8*x['request_body_bytes']/x['wall_s']/1e6 for x in bulk),
    'note':'Round-trip and upload measured on caller monotonic clock. One-way delay is validated by paired RTT increment, not unsynchronized host clocks.'})

def expand(record,n):
 """Clone each of four agent identities; expand peer endpoints by Cartesian clone pairs."""
 factor=n//4;out=[]
 for d in record['deliveries']:
  a=str(d['sender']);b=str(d['recipient'])
  # Native non-integer names stay singleton; only registered numeric 0..3 identities are scaled.
  sa=[str(int(a)+4*k) for k in range(factor)] if a in ['0','1','2','3'] else [a]
  sb=[str(int(b)+4*k) for k in range(factor)] if b in ['0','1','2','3'] else [b]
  for aa in sa:
   for bb in sb:
    out.append(json.dumps({'sender':aa,'recipient':bb,'kind':d['kind'],'text':d['text']},ensure_ascii=False,separators=(',',':')).encode())
 return out

def run(plan,out):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=False)
 rate=plan['rate'];warm=plan.get('warmup_s',10);duration=plan.get('steady_s',max(60,math.ceil(1000/rate)))
 traces=[expand(r,plan['agents']) for r in plan['records']]
 if not traces or any(not t for t in traces):raise ValueError('Missing communication trace; cannot infer zero traffic')
 scheduled=math.ceil((duration+warm)*rate);zero=time.perf_counter()+.2;slots=threading.BoundedSemaphore(32)
 order=list(range(len(traces)));random.Random(plan.get('scheduling_seed',20261031)).shuffle(order)
 lock=threading.Lock();events=[];tls=threading.local();cpu0=time.process_time();peak0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 journal=(out/'events.jsonl').open('a',encoding='utf-8');stop_commit=threading.Event()
 save(out/'PLAN.json',plan)
 def record(item):
  with lock:
   events.append(item);journal.write(json.dumps(item)+'\n');journal.flush()
 def commit_partial():
  while not stop_commit.wait(60):
   subprocess.run(['sync','/evidence'],check=True,timeout=45)
 threading.Thread(target=commit_partial,daemon=True).start()
 def query(i,due,steady):
  source_index=order[i%len(order)]
  now=time.perf_counter();item={'id':i,'source_index':source_index,'scheduled_monotonic_s':due,'started_monotonic_s':now,'steady':steady,'state':'accepted','deliveries':[]}
  try:
   if not hasattr(tls,'session'):tls.session=Session()
   for body in traces[source_index]:
    left=due+30-time.perf_counter()
    if left<=0:raise TimeoutError('Request deadline expired')
    item['deliveries'].append(rpc(tls.session,plan['url'],body,min(30,left)))
   item.update(state='completed',latency_s=time.perf_counter()-due)
   if item['latency_s']>30:item['state']='timed_out'
  except Exception as exc:item.update(state='timed_out' if isinstance(exc,TimeoutError) else 'transport_error',error_type=type(exc).__name__,latency_s=time.perf_counter()-due)
  finally:
   item['finished_monotonic_s']=time.perf_counter()
   record(item)
   slots.release()
 # Admission is open-loop; saturation drops arrivals instead of delaying the generator.
 with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
  for i in range(scheduled):
   due=zero+i/rate;remaining=due-time.perf_counter()
   if remaining>0:time.sleep(remaining)
   steady=i/rate>=warm
   if slots.acquire(blocking=False):pool.submit(query,i,due,steady)
   else:record({'id':i,'scheduled_monotonic_s':due,'steady':steady,'state':'dropped','admission_lag_s':time.perf_counter()-due})
 stop_commit.set();journal.close()
 events.sort(key=lambda r:r['id'])
 rows=[x for x in events if x['steady']];ok=[x for x in rows if x['state']=='completed']
 counts={k:sum(x['state']==k for x in rows) for k in ['completed','dropped','timed_out','transport_error']}
 metrics={'evidence_class':'physical_two_region_payload_replay','counts':counts,'scheduled':len(rows),'accepted':len(rows)-counts['dropped'],
  'unfinished':len(rows)-sum(counts.values()),'steady_s':duration,'rate':rate,'completed_per_s':len(ok)/duration,
  'delivery_goodput_queries_per_s':sum(x['latency_s']<=5 for x in ok)/duration,
  'failure_fraction':1-len(ok)/len(rows) if rows else None,'cpu_s':time.process_time()-cpu0,
  'rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'all_http_body_bytes':sum(d['request_body_bytes']+d['response_body_bytes'] for x in events for d in x.get('deliveries',[])),
  'latencies_s':[x['latency_s'] for x in ok],
  'limitations':'Transport goodput only, no live accuracy or LLM latency. Preserves recorded delivery order; removes model compute waits. All replay envelopes cross the same fixed A-to-B path. Acknowledgements include hashes. Captured IP bytes are distinct from HTTP bytes and exclude Ethernet overhead.'}
 save(out/'raw.json',{'plan':plan,'events':events});save(out/'metrics.json',metrics)
 subprocess.run(['sync','/evidence'],check=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['server','validate','run']);p.add_argument('--url');p.add_argument('--out');p.add_argument('--plan');a=p.parse_args()
 if a.mode=='server':http.server.ThreadingHTTPServer(('0.0.0.0',8000),Handler).serve_forever()
 elif a.mode=='validate':validate(a.url,a.out)
 else:run(json.loads(pathlib.Path(a.plan).read_text()),a.out)
