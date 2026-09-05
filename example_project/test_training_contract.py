"""CP-86 host refusals and real shell/HTTP sync probes; no GPU."""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train
import train_loop

@pytest.mark.parametrize('flags',[['--steps','0'],['--steps','-1'],['--episodes','0'],['--timeout','0'],['--sync-timeout','0'],['--entropy-coeff','0.5'],['--use-kl-loss']])
def test_loop_refuses_before_import_or_collection(tmp_path,flags):
    # -S removes installed packages: a late refusal would fail on dependencies.
    start=time.monotonic()
    result=subprocess.run([sys.executable,'-S',str(HERE/'train_loop.py'),'--snapshot','missing','--steps','1','--run-dir',str(tmp_path/'run'),*flags],capture_output=True,text=True)
    elapsed=time.monotonic()-start
    assert result.returncode==2,(result.stdout,result.stderr)
    assert 'found' in result.stderr and 'expected' in result.stderr and 'use' in result.stderr.lower()
    assert not (tmp_path/'run').exists()
    assert 'Traceback' not in result.stderr
    print(f'{flags}: refused in {elapsed:.6f}s; no run directory; python -S')

def test_loop_stale_before_import(tmp_path):
    (tmp_path/'step1/collected').mkdir(parents=True)
    (tmp_path/'step1/collected/old.json').write_text('{}')
    r=subprocess.run([sys.executable,'-S',str(HERE/'train_loop.py'),'--snapshot','missing','--steps','1','--run-dir',str(tmp_path)],capture_output=True,text=True)
    assert r.returncode and 'nonempty' in r.stderr and 'fresh' in r.stderr
    assert 'Traceback' not in r.stderr

def test_collect_refuses_before_client(tmp_path,monkeypatch):
    (tmp_path/'old.json').write_text('{}')
    def forbidden(*a): pytest.fail('client constructed before freshness refusal')
    monkeypatch.setattr(train,'RolloutClient',forbidden)
    with pytest.raises(SystemExit,match='nonempty'):
        train.collect(None,[],tmp_path,2,1)

@pytest.mark.parametrize('flags',[['--collect-only'],['--snapshot','missing']])
def test_single_step_stale_before_import(tmp_path,flags):
    (tmp_path/'old.json').write_text('{}')
    r=subprocess.run([sys.executable,'-S',str(HERE/'train.py'),'--out',str(tmp_path),*flags],capture_output=True,text=True)
    assert r.returncode and 'nonempty' in r.stderr and 'fresh' in r.stderr
    assert 'Traceback' not in r.stderr

def test_sync_timer_includes_shell(monkeypatch):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from threading import Thread
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path=='/v1/models'
            time.sleep(.08)
            self.send_response(200);self.end_headers();self.wfile.write(b'{"data":[{"id":"model"}]}')
        def log_message(self,*a): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    t=Thread(target=server.serve_forever,daemon=True);t.start()
    try:
        start=time.monotonic()
        measured=train_loop.run_sync('sleep 0.25','unused',f'http://127.0.0.1:{server.server_port}','model',2)
        wall=time.monotonic()-start
        assert measured>=.33 and abs(wall-measured)<.05
        print(f'sync: reported={measured:.6f}s wall={wall:.6f}s')
    finally: server.shutdown();server.server_close();t.join()

def test_sync_checkpoint_is_one_literal_argument(tmp_path,monkeypatch):
    import shlex
    script=tmp_path/'args.py';out=tmp_path/'args.json'
    script.write_text('import sys,json;open(sys.argv[1],"w").write(json.dumps(sys.argv[2:]))')
    checkpoint=str(tmp_path/"checkpoint with spaces ' quote $(touch INJECTED) `false`")
    monkeypatch.setattr(train_loop,'wait_for_engine',lambda *a:0.)
    train_loop.run_sync(f'{shlex.quote(sys.executable)} {shlex.quote(str(script))} {shlex.quote(str(out))} {{ckpt}}',checkpoint,'unused','model',2)
    assert json.loads(out.read_text())==[checkpoint]
    assert not (Path.cwd()/'INJECTED').exists()

@pytest.mark.parametrize('template',["echo '{ckpt}'",'echo "{ckpt}"','echo prefix{ckpt}'])
def test_quoted_placeholder_refused(template):
    with pytest.raises(SystemExit,match='unquoted'):
        train_loop.run_sync(template,'x','unused','model',1)

def test_worker_controls_refuse_before_cuda():
    spec=importlib.util.spec_from_file_location('contract_loop',HERE.parent/'verl_bridge/loop.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for kwargs in [{'entropy_coeff':.5},{'use_kl_loss':True}]:
        with pytest.raises(ValueError,match='unsupported'):mod.make_worker('missing',**kwargs)

@pytest.mark.parametrize('flags',[['--episodes','0'],['--timeout','nan'],['--timeout','inf']])
def test_single_step_positive_arguments(flags):
    r=subprocess.run([sys.executable,'-S',str(HERE/'train.py'),*flags],capture_output=True,text=True)
    assert r.returncode==2 and 'expected a positive finite value' in r.stderr
    assert 'Traceback' not in r.stderr


def test_sync_failure_time(capsys):
    with pytest.raises(SystemExit,match=r'exited 7 after 0\.[12]\d\ds from command start'):
        train_loop.run_sync('sleep 0.12; exit 7','unused','unused','model',1)


def test_sync_readiness_failure_time(capsys,monkeypatch):
    def fail(*args):
        time.sleep(.05)
        raise SystemExit('not ready')
    monkeypatch.setattr(train_loop,'wait_for_engine',fail)
    with pytest.raises(SystemExit,match='not ready'):
        train_loop.run_sync('sleep 0.12','unused','unused','model',1)
    assert 'sync failed after 0.' in capsys.readouterr().err
