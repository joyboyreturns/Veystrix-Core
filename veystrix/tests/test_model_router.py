import http.client
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from veystrix.model_coordinator import ModelCoordinator
from veystrix.model_registry import InstalledModel
from veystrix.model_router import RouterServer, session_owner


def model(name):
    return InstalledModel(name, "flashnext", name, "/models/" + name, "qwen4_exp", {}, None, "test")


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls.append(body["model"])
        response = (b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n' if body.get("stream") else json.dumps({"model": body["model"], "choices": []}).encode())
        self.send_response(200)
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(response)


class Driver:
    def __init__(self, port):
        self.port, self.api_key, self.current = port, "internal", None
        self.events = []
    def load(self, m, cancelled):
        assert self.current is None
        self.current = m.id
        self.events.append(("load", m.id))
    def unload(self):
        self.events.append(("unload", self.current))
        self.current = None
    def ready(self, m): return self.current == m.id
    def health(self): return {"ready": True, "installed_model_id": self.current}


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        self.backend.calls = []
        self.bthread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.bthread.start()
        self.models = {x: model(x) for x in ("a", "b")}
        self.driver = Driver(self.backend.server_port)
        self.coordinator = ModelCoordinator(self.models.get, self.driver)
        self.server = RouterServer(("127.0.0.1", 0), self.coordinator, self.driver, api_key="secret", inventory=lambda: list(self.models.values()))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.backend.shutdown(); self.backend.server_close(); self.bthread.join()
    def call(self, path, payload=None, key="secret"):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        body = None if payload is None else json.dumps(payload)
        conn.request("GET" if payload is None else "POST", path, body=body, headers=headers)
        r = conn.getresponse(); status = r.status; result = json.loads(r.read()); conn.close()
        return status, result
    def test_picker_and_same_session_switch(self):
        status, result = self.call("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual({x["id"] for x in result["data"]}, {"a", "b"})
        msg = [{"role": "user", "content": "same coding session"}]
        for mid in ("a", "b", "a"):
            status, response = self.call("/v1/chat/completions", {"model": mid, "messages": msg})
            self.assertEqual((status, response["model"]), (200, mid))
        self.assertEqual(self.backend.calls, ["a", "b", "a"])
        self.assertEqual(self.driver.events, [("load", "a"), ("unload", "a"), ("load", "b"), ("unload", "b"), ("load", "a")])
    def test_other_session_cannot_evict(self):
        self.call("/v1/chat/completions", {"model": "a", "messages": [{"role": "user", "content": "main"}]})
        status, response = self.call("/v1/chat/completions", {"model": "b", "messages": [{"role": "user", "content": "background"}]})
        self.assertEqual((status, response["error"]["code"]), (409, "model_handoff_required"))
        self.assertEqual(self.driver.current, "a")
    def test_stream_sends_loading_heartbeat_without_splitting_sse(self):
        import time
        from unittest.mock import patch
        old_load = self.driver.load
        def slow(model, cancelled):
            time.sleep(0.18)
            old_load(model, cancelled)
        self.driver.load = slow
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        data = json.dumps({"model":"a", "messages":[{"role":"user", "content":"hello"}], "stream":True})
        with patch("veystrix.model_router.HEARTBEAT_SECONDS", 0.03):
            conn.request("POST", "/v1/chat/completions", data, {"Authorization":"Bearer secret", "Content-Type":"application/json"})
            response = conn.getresponse(); raw = response.read().decode(); conn.close()
        self.assertEqual(response.status, 200)
        self.assertIn(": generating\n\n", raw)
        self.assertIn('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n', raw)

    def test_legacy_family_alias_rejects_ambiguity(self):
        status, body = self.call("/v1/chat/completions", {"model": "macqwen-flashnext", "messages": []})
        self.assertEqual((status, body["error"]["code"]), (404, "model_not_found"))
        self.assertEqual(self.driver.events, [])

    def test_bad_model_and_key_do_not_load(self):
        self.assertEqual(self.call("/v1/models", key="wrong")[0], 401)
        status, body = self.call("/v1/chat/completions", {"model": "bad", "messages": []})
        self.assertEqual((status, body["error"]["code"]), (404, "model_not_found"))
        self.assertEqual(self.driver.events, [])

if __name__ == "__main__": unittest.main()

CHILD_ROUTER = r'''import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def do_GET(self):
  self.reply({'ready':True,'instance_id':os.environ['VEYSTRIX_ENGINE_INSTANCE_ID'],'installed_model_id':sys.argv[2]})
 def do_POST(self):
  d=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  self.reply({'model':d['model'],'process_model':sys.argv[2]})
 def reply(self,obj):
  b=json.dumps(obj).encode(); self.send_response(200); self.send_header('Content-Length',str(len(b))); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(b)
HTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()
'''

class ProcessRouterTests(unittest.TestCase):
 def test_two_owned_processes_switch_and_serve_exact_model(self):
  import sys
  from veystrix.model_process import ProcessDriver
  models = {x: model(x) for x in ('a','b')}
  d = ProcessDriver(startup_timeout=5, stop_timeout=5, command_factory=lambda m,p:[sys.executable,'-c',CHILD_ROUTER,str(p),m.id])
  c = ModelCoordinator(models.get,d)
  server = RouterServer(('127.0.0.1',0),c,d,inventory=lambda:list(models.values()))
  t = threading.Thread(target=server.serve_forever,daemon=True); t.start()
  pids=[]
  try:
   for name in ('a','b','a'):
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=10)
    data=json.dumps({'model':name,'messages':[{'role':'user','content':'same'}]})
    conn.request('POST','/v1/chat/completions',data,{'Content-Type':'application/json'})
    response=conn.getresponse(); body=json.loads(response.read()); conn.close()
    self.assertEqual((response.status,body['model'],body['process_model']),(200,name,name))
    pids.append(d.process.pid)
   self.assertEqual(len(set(pids)),3)
  finally:
   server.shutdown(); server.server_close(); t.join()
   d.unload()
