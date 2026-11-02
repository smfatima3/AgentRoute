import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("command", nargs=argparse.REMAINDER)
parser.add_argument("--cwd", required=True)
parser.add_argument("--log", required=True)
args = parser.parse_args()
if not sys.stdin.isatty():
    raise SystemExit("Credential input requires a terminal with echo disabled.")
token_id = getpass.getpass("Modal token ID (hidden): ").strip()
token_secret = getpass.getpass("Modal token secret (hidden): ").strip()
env = os.environ.copy()
temp = Path(__file__).resolve().parent / "temp"
temp.mkdir(exist_ok=True)
env.update(TEMP=str(temp), TMP=str(temp))
env.update(MODAL_TOKEN_ID=token_id, MODAL_TOKEN_SECRET=token_secret,
           PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
log_path = Path(args.log).resolve()
log_path.parent.mkdir(parents=True, exist_ok=True)
print("Credentials loaded in process memory; no credential file written.", flush=True)
with log_path.open("a", encoding="utf-8") as log:
    child = subprocess.Popen([sys.executable, *args.command], cwd=args.cwd, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
    try:
        for line in child.stdout:
            line = line.replace(token_id, "[REDACTED]").replace(token_secret, "[REDACTED]")
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        result = child.wait()
    except KeyboardInterrupt:
        child.terminate()
        result = child.wait()
print("Command exit code:", result, flush=True)
raise SystemExit(result)
