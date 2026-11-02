"""Streaming loopback proxy to authenticated GPU inference; no retries or response cache."""
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit

def start_proxy(endpoint,token,port=8000,allow_local_http=False):
    target=urlsplit(endpoint)
    if target.scheme!='https' and not (allow_local_http and target.scheme=='http' and target.hostname=='127.0.0.1'):
        raise ValueError('Remote inference requires HTTPS')
    if target.username or target.password or target.query or target.fragment or target.path not in ['', '/']:
        raise ValueError('Expected a plain HTTPS tunnel origin')
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.0'
        def log_message(self,*args):pass
        def do_GET(self):self.forward()
        def do_POST(self):self.forward()
        def forward(self):
            if (self.command,self.path) not in [('GET','/v1/models'),('POST','/v1/chat/completions')]:
                self.send_error(404);return
            if self.headers.get('Transfer-Encoding'):
                self.send_error(400);return
            size=int(self.headers.get('Content-Length','0'))
            if not 0<=size<=8*1024*1024:self.send_error(413);return
            body=self.rfile.read(size) if size else None
            connection=(http.client.HTTPSConnection if target.scheme=='https' else http.client.HTTPConnection)(
                target.hostname,target.port,timeout=310)
            started=False
            try:
                connection.request(self.command,self.path,body=body,
                    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
                response=connection.getresponse()
                self.send_response(response.status)
                self.send_header('Content-Type',response.getheader('Content-Type','application/json'))
                self.send_header('Connection','close');self.end_headers();started=True
                while data:=response.read1(65536):
                    self.wfile.write(data);self.wfile.flush()
            except (OSError,http.client.HTTPException):
                if not started:
                    self.send_error(502,'Inference transport failed')
                # On a broken stream, close the connection. The original client rejects missing usage.
            finally:
                connection.close();self.close_connection=True
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server
