"""Validate the locked stack and real pinned Qwen tokenizer before model loading."""
import argparse, hashlib, json, platform, subprocess, sys, traceback
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT/'requirements/serving.lock'

def pinned_versions():
    pins = {}
    for line in LOCK.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            name, version = line.split('=='); pins[name] = version
    return pins

def version_status(pins=None, get_version=metadata.version):
    installed, mismatches = {}, []
    for name, expected in (pinned_versions() if pins is None else pins).items():
        try: actual = get_version(name)
        except metadata.PackageNotFoundError: actual = None
        installed[name] = actual
        if actual != expected: mismatches.append(f'{name}: installed={actual!r}, required={expected}')
    return installed, mismatches

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model'); p.add_argument('--versions-only', action='store_true')
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--report', type=Path); a = p.parse_args()
    versions, mismatches = version_status()
    report = {'date': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
              'lock_sha256': hashlib.sha256(LOCK.read_bytes()).hexdigest(),
              'versions': versions, 'mismatches': mismatches, 'gpu_inference_performed': False}
    try:
        if mismatches:
            raise RuntimeError('Serving dependency mismatch:\n'+'\n'.join(mismatches)+
                '\nUse start_server.py --setup-env to repair the managed serving_env, '
                'or install requirements/serving.lock in your explicitly selected GPU environment.')
        if not a.versions_only:
            if not a.model: p.error('--model is required unless --versions-only is used')
            if a.model not in json.loads((ROOT/'configs/study.json').read_text())['models']:
                raise RuntimeError('Model is not enabled in this package')
            check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
            report['pip_check'] = check.stdout+check.stderr
            if check.returncode: raise RuntimeError('pip check failed:\n'+report['pip_check'])
            import torch
            report.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
            if a.require_cuda:
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA is unavailable in the serving environment; check the NVIDIA driver and GPU access.')
                if torch.cuda.device_count() < a.tensor_parallel_size:
                    raise RuntimeError('Fewer visible GPUs than --tensor-parallel-size')
                report['gpus'] = [torch.cuda.get_device_name(i) for i in range(a.tensor_parallel_size)]
            from transformers import AutoConfig
            from vllm.transformers_utils.tokenizer import get_tokenizer
            revision = json.loads((ROOT/'data/manifest.json').read_text())['models'][a.model]
            if AutoConfig.from_pretrained(a.model, revision=revision).model_type != 'qwen3':
                raise RuntimeError('Unexpected model architecture')
            tokenizer = get_tokenizer(a.model, revision=revision)
            text = 'Tokenizer compatibility check.'
            if tokenizer.decode(tokenizer.encode(text)) != text:
                raise RuntimeError('Tokenizer encode/decode check failed')
            chat = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if '<think>\n\n</think>' not in chat: raise RuntimeError('Qwen non-thinking chat template check failed')
            report.update(model=a.model, revision=revision, tokenizer=type(tokenizer).__name__,
                          special_token_count=len(tokenizer.all_special_tokens_extended),
                          tokenizer_cache_encode_decode_chat_template='passed')
            import vllm.entrypoints.openai.api_server
            report['api_server_import'] = 'passed'
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1

if __name__ == '__main__': raise SystemExit(main())
