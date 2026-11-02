"""Fetch pinned sources, preserve existing work, and report every failure."""
import argparse, json, os, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def git(*args, cwd=None, timeout=180):
    result = subprocess.run(['git', *map(str, args)], cwd=cwd,
        env=dict(os.environ, GIT_TERMINAL_PROMPT='0'), capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'git failed')
    return result.stdout.strip()

def check_checkout(dest, commit):
    if git('rev-parse', 'HEAD', cwd=dest) != commit:
        raise RuntimeError(f'{dest}: wrong commit; expected {commit}')
    if git('status', '--porcelain', '--untracked-files=no', cwd=dest):
        raise RuntimeError(f'{dest}: tracked files changed; existing work is preserved')

def fetch_one(name, spec, parent, source=None):
    dest = parent / name
    parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        check_checkout(dest, spec['commit'])
        return {'status':'ready', 'commit':spec['commit'], 'path':str(dest), 'source':'existing_verified_checkout'}
    urls = [str(Path(source).expanduser().resolve())] if source else spec.get('fetch_urls', [spec['url']])
    errors = []
    for url in urls:
        try:
            with tempfile.TemporaryDirectory(prefix=f'.{name}-fetch-', dir=parent) as temp:
                staging = Path(temp) / 'repo'
                git('init', staging)
                git('remote', 'add', 'origin', url, cwd=staging)
                git('fetch', '--depth', '1', 'origin', spec['commit'], cwd=staging)
                git('checkout', '--detach', spec['commit'], cwd=staging)
                check_checkout(staging, spec['commit'])
                staging.rename(dest)
            return {'status':'ready', 'commit':spec['commit'], 'path':str(dest), 'source':url}
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            errors.append({'source':url, 'error':str(exc)})
    return {'status':'unavailable', 'commit':spec['commit'], 'path':str(dest), 'attempts':errors,
        'action':'Restore an authorized Git checkout containing the pinned commit with --source NAME=/path/to/repo. No replacement baseline is selected.'}

def main():
    specs = json.loads((ROOT / 'configs/upstream.json').read_text())
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dest', type=Path, default=ROOT / 'upstream')
    ap.add_argument('--methods', nargs='+', choices=list(specs), default=list(specs))
    ap.add_argument('--source', action='append', default=[], metavar='NAME=GIT_DIRECTORY')
    ap.add_argument('--report', type=Path, default=ROOT / 'setup/upstream_status.json')
    args = ap.parse_args(); overrides = {}
    for value in args.source:
        name, separator, path = value.partition('=')
        if not separator or name not in specs or not path:
            ap.error('--source must be NAME=/path/to/an/authorized/git/checkout')
        overrides[name] = path
    records = {}
    for name in dict.fromkeys(args.methods):
        try: records[name] = fetch_one(name, specs[name], args.dest.resolve(), overrides.get(name))
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            records[name] = {'status':'blocked', 'commit':specs[name]['commit'], 'error':str(exc)}
        print(name, records[name]['status'], specs[name]['commit'], flush=True)
        if records[name]['status'] != 'ready': print(json.dumps(records[name], indent=2), flush=True)
    report = {'checked_at':datetime.now(timezone.utc).isoformat(), 'methods':records,
              'complete_for_requested_methods':all(r['status']=='ready' for r in records.values())}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print('Setup report:', args.report, flush=True)
    if not report['complete_for_requested_methods']:
        print('Source setup incomplete. AgentRoute/broadcast/no_comm can run independently; full baseline suites remain blocked.', flush=True)
        return 2
    return 0

if __name__ == '__main__': raise SystemExit(main())
