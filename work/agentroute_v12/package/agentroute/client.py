"""Metered synchronous Chat Completions client, compatible with vLLM.

No invisible retries, response cache, fabricated token counts, or error text logs.
Every attempted HTTP call receives a ledger entry, even on transport failure.
"""
import json
import os
import time
import urllib.request


class ChatClient:
    def __init__(self, settings):
        self.s = settings

    def call(self, messages, seed, max_tokens, category, ledger):
        s = self.s
        payload = {'model': s['model'], 'messages': messages,
                   'temperature': s['temperature'], 'max_tokens': max_tokens,
                   'seed': seed, 'stream': s.get('stream', True)}
        if payload['stream']:
            payload['stream_options'] = {'include_usage': True}
        # For servers requiring reasoning/template settings, expose them in config.
        for key, val in s.get('extra_body', {}).items():
            if key in payload:
                raise ValueError(f'extra_body cannot overwrite {key}')
            payload[key] = val
        headers = {'Content-Type': 'application/json'}
        if s.get('api_key_env'):
            key = os.environ.get(s['api_key_env'])
            if not key:
                raise ValueError(f'Missing environment variable {s["api_key_env"]}')
            headers['Authorization'] = 'Bearer ' + key
        record = {'category': category, 'model': s['model'], 'status': 'error',
                  'prompt_tokens': None, 'completion_tokens': None, 'cached_tokens': None,
                  'usd': None, 'seed': seed, 'max_tokens': max_tokens}
        record['messages'] = messages
        start = time.perf_counter()
        record['started_unix_s']=time.time()
        record['started_monotonic_s']=start
        record['inference_http_response_body_bytes']=0
        try:
            body=json.dumps(payload).encode()
            record['inference_http_request_body_bytes']=len(body)
            request = urllib.request.Request(s['base_url'].rstrip('/') + '/chat/completions',
                                             body, headers)
            with urllib.request.urlopen(request, timeout=s['timeout_s']) as response:
                if payload['stream']:
                    fragments, data, first = [], {'choices': []}, None
                    finish = None
                    for line in response:
                        record['inference_http_response_body_bytes']+=len(line)
                        if not line.startswith(b'data:'): continue
                        raw = line[5:].strip()
                        if raw == b'[DONE]': break
                        event = json.loads(raw)
                        if event.get('usage'): data['usage'] = event['usage']
                        if event.get('id'): data['id'] = event['id']
                        if event.get('system_fingerprint'): data['system_fingerprint'] = event['system_fingerprint']
                        for choice in event.get('choices', []):
                            token = (choice.get('delta') or {}).get('content')
                            if token:
                                if first is None: first = time.perf_counter()-start
                                fragments.append(token)
                            if choice.get('finish_reason'): finish = choice['finish_reason']
                    data['choices'] = [{'message': {'content': ''.join(fragments)}, 'finish_reason': finish}]
                    record['ttft_s'] = first
                else:
                    raw=response.read();record['inference_http_response_body_bytes']=len(raw)
                    data = json.loads(raw)
                    record['ttft_s'] = None
            usage = data.get('usage') or {}
            pin, pout = usage.get('prompt_tokens'), usage.get('completion_tokens')
            if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (pin, pout)):
                raise ValueError('Missing authoritative usage')
            cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
            if cached is not None and (not isinstance(cached, int) or isinstance(cached,bool) or not 0 <= cached <= pin):
                raise ValueError('Invalid cached-token usage')
            record.update(prompt_tokens=pin, completion_tokens=pout, cached_tokens=cached)
            record['raw_usage']=usage
            prices = s.get('prices')
            if prices and cached is not None:
                record['usd'] = ((pin-cached)*prices['input_per_million'] + cached*prices['cached_input_per_million']
                                 + pout*prices['output_per_million']) / 1e6
            choice = data['choices'][0]
            record['finish_reason'] = choice.get('finish_reason')
            text = choice['message'].get('content')
            if not isinstance(text, str):
                raise ValueError('Response has no text content')
            record['response_id'] = data.get('id')
            record['system_fingerprint'] = data.get('system_fingerprint')
            record['status'] = 'ok'
            record['response_text'] = text
            return text
        except Exception as exc:
            # Do not record exception messages, which can contain URLs or credentials.
            record['error_type'] = type(exc).__name__
            raise
        finally:
            record['wall_s'] = time.perf_counter() - start
            record['ended_monotonic_s']=time.perf_counter()
            record['ended_unix_s']=time.time()
            record['byte_scope']='Client-to-inference-server HTTP bodies; excludes headers/TCP and is not peer-agent wire traffic'
            ledger.append(record)
