import os
import sys
import unittest
from pathlib import Path
from veystrix.model_process import ProcessDriver
from veystrix.model_registry import InstalledModel

# A real child process and HTTP endpoint; no model or MLX work.
CHILD = r'''import json,os,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
class H(BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def do_GET(self):
  b=json.dumps({'ready':True,'instance_id':os.environ['VEYSTRIX_ENGINE_INSTANCE_ID'],'installed_model_id':'test-model'}).encode()
  self.send_response(200); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
HTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()
'''

class ModelProcessTests(unittest.TestCase):
 def test_real_child_is_verified_and_unloaded(self):
  m=InstalledModel('test-model','flashnext','test','/tmp/test','qwen4_exp',{},None,'test')
  driver=ProcessDriver(startup_timeout=5,stop_timeout=5,command_factory=lambda model,port:[sys.executable,'-c',CHILD,str(port)])
  try:
   driver.load(m,lambda:False)
   self.assertTrue(driver.ready(m))
   pid=driver.process.pid
   driver.unload()
   self.assertIsNone(driver.process)
   with self.assertRaises(ProcessLookupError): os.kill(pid,0)
  finally:
   if driver.process is not None:
    driver.process.kill(); driver.process.wait()
 def test_wrong_identity_fails_and_cleanup_works(self):
  m=InstalledModel('wrong-model','flashnext','test','/tmp/test','qwen4_exp',{},None,'test')
  driver=ProcessDriver(startup_timeout=0.5,stop_timeout=5,command_factory=lambda model,port:[sys.executable,'-c',CHILD,str(port)])
  try:
   with self.assertRaises(TimeoutError): driver.load(m,lambda:False)
   driver.unload()
   self.assertIsNone(driver.process)
  finally:
   if driver.process is not None:
    driver.process.kill(); driver.process.wait()

class GuardTests(unittest.TestCase):
 def test_guard_terminates_child_when_owner_pipe_closes(self):
  import subprocess, time
  read_fd, write_fd = os.pipe()
  parent = subprocess.Popen([sys.executable, '-m', 'veystrix.engine_guard', str(read_fd),
                             sys.executable, '-c', 'import time; time.sleep(30)'],
                            pass_fds=(read_fd,), start_new_session=True)
  os.close(read_fd)
  try:
   time.sleep(0.3)
   self.assertIsNone(parent.poll())
   os.close(write_fd)
   parent.wait(timeout=5)
   self.assertIsNotNone(parent.returncode)
  finally:
   if parent.poll() is None:
    os.killpg(parent.pid, 9); parent.wait()
