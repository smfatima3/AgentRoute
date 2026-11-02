"""Replay measured peer payloads over actual HTTP; this is not live LLM execution."""
import argparse,json,sys,time,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentroute.transport import Manager,checksum
from agentroute.core import write_json,file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True)
    p.add_argument('--worker-a',required=True);p.add_argument('--worker-b',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--location-aware',type=int,choices=[0,1],default=1)
    p.add_argument('--buffering',type=int,choices=[0,1],default=1)
    p.add_argument('--deployment',choices=['same-host','cross-host'],required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh replay output directory')
    record=json.loads(a.record.read_text())
    if record.get('status')!='ok' or not record.get('usage_complete'):raise ValueError('A successful metered model-run record is required')
    messages=[d for d in record['deliveries'] if d['kind']=='peer']
    if not messages:raise ValueError('No peer payloads to replay')
    a.output.mkdir(parents=True);manager=Manager(a.output/'directory.sqlite',bool(a.location_aware),bool(a.buffering))
    namespace=uuid.uuid4().hex;agents=sorted({d['recipient'] for d in messages})
    for agent in agents:manager.register(namespace+agent,a.worker_a,{'query_id':record['id'],'coordination_position':'replay-start'})
    receipts=[];moves=[];started=time.perf_counter()
    for index,delivery in enumerate(messages):
        target=namespace+delivery['recipient'];message={'id':f'{namespace}/{index}','sender':delivery['sender'],'text':delivery['text']}
        # A real migration snapshot is in progress when this selected message
        # arrives. There is no synthetic delay or fabricated network-time saving.
        if not any(move['agent_id']==target for move in moves):
            moves.append(manager.migrate(target,a.worker_b,after_freeze=lambda:receipts.append(manager.send(target,message))))
        else:receipts.append(manager.send(target,message))
    received=0
    for agent in agents:
        id=namespace+agent;loc=manager.location(id);state=manager.rpc(loc['endpoint'],'inspect',{'agent_id':id})
        expected={f'{namespace}/{i}' for i,d in enumerate(messages) if d['recipient']==agent}
        if set(state['inbox'])!=expected:raise RuntimeError('Missing or extra receipts after migration')
        received+=len(expected)
    write_json(a.output/'transport.json',{'evidence_class':'measured_http_replay','deployment':a.deployment,
        'source_record_sha256':file_hash(a.record),'location_aware':bool(a.location_aware),'buffering':bool(a.buffering),
        'selected_messages':len(messages),'unique_received_messages':received,'migration_count':len(moves),
        'wall_s':time.perf_counter()-started,'migrations':moves,'http_events':manager.events,
        'http_body_bytes':sum(e['request_body_bytes']+(e['response_body_bytes'] or 0) for e in manager.events),
        'http_attempts':len(manager.events),'note':'HTTP JSON body bytes exclude headers/TCP overhead. Receipts are not LLM processing acknowledgments.'})

if __name__=='__main__':main()
