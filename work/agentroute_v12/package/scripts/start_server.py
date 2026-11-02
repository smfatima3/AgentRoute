"""Validate dependencies, start/reuse a GPU server, and verify real generation."""
import argparse, json, os, signal, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentroute.server import model_status, wait_ready
from check_server import check_generation

def version_probe(python):
    return subprocess.run([str(python), str(ROOT/'scripts/check_serving_env.py'), '--versions-only'],
                          capture_output=True, text=True, timeout=90)

def select_environment(a):
    """Only the package-owned environment is installed/repaired automatically."""
    env_python = ROOT/'serving_env/bin/python'
    selected = a.serving_python or env_python
    probe = version_probe(selected) if selected.is_file() else None
    if probe is not None and probe.returncode == 0: return selected
    if a.serving_python or not a.setup_env:
        detail = (probe.stdout+probe.stderr)[-10000:] if probe else f'Missing Python: {selected}'
        raise RuntimeError('Serving dependencies do not match requirements/serving.lock. '
            'Use --setup-env for the package-owned environment, or repair the selected environment.\n'+detail)
    setup = ROOT/'setup'; setup.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if env_python.is_file():
        before = subprocess.run([str(env_python), '-m', 'pip', 'freeze'], capture_output=True, text=True)
        (setup/f'serving_before_repair_{stamp}.txt').write_text(before.stdout+before.stderr)
        print('Repairing package-owned serving_env from the validated dependency lock.', flush=True)
    else:
        print('Creating isolated serving_env from the validated dependency lock.', flush=True)
        subprocess.run([sys.executable, '-m', 'venv', str(env_python.parent.parent)], check=True)
    subprocess.run([str(env_python), '-m', 'pip', 'install', '-r', str(ROOT/'requirements/serving.lock')], check=True)
    probe = version_probe(env_python)
    if probe.returncode: raise RuntimeError('Dependency verification failed after installation:\n'+probe.stdout+probe.stderr)
    return env_python

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True); p.add_argument('--gpu', default='0')
    p.add_argument('--port', type=int, default=8000); p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--max-model-len', type=int, default=32768); p.add_argument('--serving-python', type=Path)
    p.add_argument('--setup-env', action='store_true', help='Install/repair the locked stack in package-owned serving_env')
    p.add_argument('--wait-seconds', type=float, default=1200); a = p.parse_args()
    if a.model not in json.loads((ROOT/'configs/study.json').read_text())['models']:
        p.error('Model is not enabled in this package')
    if a.tensor_parallel_size < 1: p.error('Tensor parallel size must be positive')
    base_url = f'http://127.0.0.1:{a.port}/v1'
    if model_status(base_url, a.model):
        print('Reusing endpoint; its environment/revision cannot be verified remotely:', base_url, flush=True)
        check_generation(base_url, a.model, ROOT/'setup/generation_check.json')
        return
    selected = select_environment(a)
    probe_path = ROOT/'setup/serving_preflight.json'
    subprocess.run([str(selected), str(ROOT/'scripts/check_serving_env.py'), '--model', a.model,
        '--require-cuda', '--tensor-parallel-size', str(a.tensor_parallel_size), '--report', str(probe_path)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu), cwd=ROOT, check=True, timeout=600)
    log_dir = ROOT/'logs'; log_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_path = log_dir/f'{a.model.split("/")[-1]}_server_{stamp}.log'
    cmd = [str(selected), str(ROOT/'scripts/serve.py'), '--model', a.model, '--gpu', a.gpu,
           '--port', str(a.port), '--tensor-parallel-size', str(a.tensor_parallel_size),
           '--max-model-len', str(a.max_model_len)]
    with log_path.open('w') as log:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run_meta = {'model': a.model, 'port': a.port, 'gpu': a.gpu, 'tensor_parallel_size': a.tensor_parallel_size,
                'command': cmd, 'probe_report': str(probe_path), 'pid': process.pid, 'log': str(log_path)}
    (log_dir/f'server_{a.port}.json').write_text(json.dumps(run_meta, indent=2)+'\n')
    print('Server log:', log_path, flush=True)
    try:
        wait_ready(base_url, a.model, a.wait_seconds, process=process, log_path=log_path)
        check_generation(base_url, a.model, ROOT/'setup/generation_check.json')
    except BaseException:
        if process.poll() is None: os.killpg(process.pid, signal.SIGTERM)
        print(log_path.read_text(errors='replace')[-8000:], file=sys.stderr)
        raise
    print('READY:', a.model, base_url, flush=True)

if __name__ == '__main__': main()
