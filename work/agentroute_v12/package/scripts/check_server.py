"""One genuine API generation check; stored separately from benchmark results."""
import argparse, json, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentroute.server import wait_ready

def check_generation(base_url, model, report_path):
    request = {'model': model, 'messages': [{'role': 'user', 'content': 'Reply with the single word READY.'}],
               'temperature': 0, 'max_tokens': 32, 'seed': 97, 'stream': False,
               'chat_template_kwargs': {'enable_thinking': False}}
    report = {'date': datetime.now(timezone.utc).isoformat(), 'purpose': 'startup_smoke_test_not_benchmark',
              'base_url': base_url, 'request': request}
    try:
        wait_ready(base_url, model, timeout_s=60)
        req = urllib.request.Request(base_url.rstrip('/')+'/chat/completions',
            data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response: payload = json.load(response)
        report['response'] = payload
        if payload.get('model') != model: raise RuntimeError('Generation response has the wrong model ID')
        content = payload['choices'][0]['message']['content']
        if not isinstance(content, str) or not content.strip(): raise RuntimeError('Generation returned no text')
        if payload.get('usage', {}).get('completion_tokens', 0) <= 0:
            raise RuntimeError('Generation response has no positive completion-token accounting')
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise RuntimeError(f'Generation smoke test failed: {exc}. Details: {report_path}') from exc
    finally:
        report_path = Path(report_path); report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2)+'\n')
    print('GENERATION CHECK PASSED:', model, 'Report:', report_path, flush=True)
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True); p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--report', type=Path, default=ROOT/'setup/generation_check.json'); a=p.parse_args()
    if a.model not in json.loads((ROOT/'configs/study.json').read_text())['models']:
        p.error('Model is not enabled in this package')
    check_generation(a.base_url, a.model, a.report)

if __name__ == '__main__': main()
