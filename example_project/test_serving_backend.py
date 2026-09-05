"""CP-87: execute the sync launch with a fake vLLM; no SSH or GPU."""

import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def capture_launch(script, tmp_path):
    """Run the actual launch stanza, excluding lifecycle and remote commands."""
    lines = script.read_text().splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.startswith('CUDA_VISIBLE_DEVICES=')]
    assert len(starts) == 1, script
    start = starts[0]
    end = next(i for i in range(start, len(lines))
               if lines[i].strip().endswith('2>&1 &'))
    (tmp_path / 'venv/bin').mkdir(parents=True)
    (tmp_path / 'run').mkdir()
    capture = tmp_path / 'capture.py'
    capture.write_text(
        'import json, os, sys\n'
        'keys = ["CUDA_VISIBLE_DEVICES", "VLLM_LOGGING_LEVEL", '
        '"VLLM_ATTENTION_BACKEND", "VLLM_USE_FLASHINFER_SAMPLER"]\n'
        'with open("captured.json", "w") as out:\n'
        '    json.dump({"argv": sys.argv[1:], "env": '
        '{key: os.environ[key] for key in keys if key in os.environ}}, out)\n'
    )
    fake = tmp_path / 'venv/bin/vllm'
    fake.write_text('#!/bin/sh\nexec "$CP87_TEST_PYTHON" '
                    '"$CP87_TEST_CAPTURE_SCRIPT" "$@"\n')
    fake.chmod(0o755)
    env = os.environ.copy()
    env.pop('VLLM_ATTENTION_BACKEND', None)
    env.update(CP87_TEST_PYTHON=sys.executable,
               CP87_TEST_CAPTURE_SCRIPT=str(capture),
               RDIR='cp87-launch-fixture', GPU='0', PORT='18000',
               GPU_FRAC='0.30', CKPT='/synthetic/checkpoint with spaces',
               GSJ_MODEL_ID='synthetic/model', MODEL_ID='synthetic/model',
               GSJ_MODEL_REVISION='synthetic-revision')
    body = 'set -euo pipefail\n' + '\n'.join(lines[start:end + 1]) + '\nwait $!\n'
    result = subprocess.run(['bash', '-s'], input=body, cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, (script, result.stdout, result.stderr)
    return json.loads((tmp_path / 'captured.json').read_text())


def assert_automatic_backend(captured):
    assert 'VLLM_ATTENTION_BACKEND' not in captured['env'], captured['env']
    assert captured['env'] == {
        'CUDA_VISIBLE_DEVICES': '0', 'VLLM_LOGGING_LEVEL': 'DEBUG',
        'VLLM_USE_FLASHINFER_SAMPLER': '0',
    }
    assert not any(arg.startswith('--attention-backend')
                   for arg in captured['argv'])
    assert '--enforce-eager' in captured['argv']


def test_sync_launch_uses_automatic_backend_and_preserves_serving_contract(tmp_path):
    captured = capture_launch(HERE / 'sync_engine_local.sh', tmp_path)
    assert_automatic_backend(captured)
    serving_dir = Path.home() / 'cp87-launch-fixture'
    assert captured['argv'] == [
        'serve', '/synthetic/checkpoint with spaces',
        '--served-model-name', 'synthetic/model',
        '--host', '127.0.0.1', '--port', '18000',
        '--max-model-len', '32768', '--gpu-memory-utilization', '0.30',
        '--enable-auto-tool-choice', '--tool-call-parser', 'hermes',
        '--reasoning-parser', 'qwen3',
        '--default-chat-template-kwargs', '{"enable_thinking": false}',
        '--chat-template', str(serving_dir / 'qwen3_training.jinja'),
        '--generation-config', str(serving_dir / 'genconfig'),
        '--enable-log-requests', '--enforce-eager',
    ]
