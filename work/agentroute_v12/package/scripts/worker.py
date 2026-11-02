import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentroute.transport import serve_worker
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--db',required=True);p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,required=True);a=p.parse_args();serve_worker(a.db,a.host,a.port)
