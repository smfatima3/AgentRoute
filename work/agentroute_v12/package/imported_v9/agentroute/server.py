"""Readiness checks only: never submits a generation request."""
import json
import time
import urllib.error
import urllib.request

def model_status(base_url, model, headers=None, request_timeout=5):
    url = base_url.rstrip('/') + '/models'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}),
                                    timeout=request_timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(f'Server authentication failed (HTTP {exc.code}) at {url}; check the configured API key.') from exc
        return None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f'{url} did not return a valid model list.') from exc
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
        raise RuntimeError(f'{url} did not return a valid model list.')
    ids = [row.get('id') for row in payload['data'] if isinstance(row, dict)]
    if model not in ids:
        raise RuntimeError(f'Endpoint {url} serves {ids}, not the required {model}. Use the matching port/model.')
    return payload

def wait_ready(base_url, model, timeout_s=0, headers=None, process=None, log_path=None):
    deadline = time.monotonic() + timeout_s
    next_notice = 0
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'vLLM launcher exited with code {process.returncode}. Read the server log: {log_path}')
        remaining = max(0, deadline-time.monotonic())
        status = model_status(base_url, model, headers, 5 if timeout_s <= 0 else min(5, max(.1, remaining)))
        if status is not None: return status
        if time.monotonic() >= deadline:
            raise RuntimeError(f'No ready {model} server at {base_url}. Run the notebook server-start cell '
                f'(START_SERVER=True), or start scripts/serve.py in your GPU environment. '
                f'Check its log for missing vLLM, CUDA errors, or model loading failures. Log: {log_path}')
        if time.monotonic() >= next_notice:
            print('Waiting for vLLM to load the model; no inference calls are being made.', flush=True)
            next_notice = time.monotonic()+30
        time.sleep(min(2, max(.01, deadline-time.monotonic())))
