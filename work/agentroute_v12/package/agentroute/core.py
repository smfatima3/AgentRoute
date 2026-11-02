"""Validation, scoring, provenance, and JSON I/O. No generated benchmark data."""
import hashlib
import json
import math
import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp=p.with_name(p.name+'.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(p)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_rows(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def normalize(s):
    s = str(s).lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def extract(text):
    """Return only an explicitly stated final answer.

    The accepted forms were fixed on development data. The parser never infers an
    answer from arbitrary reasoning text. If several explicit markers occur, the
    last one is used because models sometimes self-correct before the final line.
    """
    if not isinstance(text, str):
        return None
    patterns = [
        r'^\s*FINAL\s*:\s*(.+?)\s*$',
        r'^\s*FINAL\s+ANSWER\s*:\s*(.+?)\s*$',
        r'^\s*FINAL\s+RESPONSE\s*:\s*(.+?)\s*$',
        r'^\s*CORRECT\s+ANSWER\s*:\s*(.+?)\s*$',
        r'^\s*ANSWER\s*:\s*(.+?)\s*$',
        r'^\s*ANSWER\s+IS\s*:?\s*(.+?)\s*$',
        r'^\s*THE\s+ANSWER\s+IS\s*:?\s*(.+?)\s*$',
    ]
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I | re.M):
            matches.append((match.start(), match.group(1)))
    if matches:
        _, value = max(matches, key=lambda item: item[0])
        pred = value.strip()
        mcq = re.match(r'^\s*([A-D])(?:[\.\)\]:,-]|\s|$)', pred, flags=re.I)
        return mcq.group(1).upper() if mcq else pred
    stripped = text.strip()
    mcq_only = re.fullmatch(r'([A-D])[\.\)]?', stripped, flags=re.I)
    if mcq_only:
        return mcq_only.group(1).upper()
    numeric_only = re.fullmatch(r'[-+]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?', stripped)
    if numeric_only and stripped:
        return stripped
    return None


def numeric(s):
    try:
        d = Decimal(str(s).replace(',', '').strip())
        return d if d.is_finite() else None
    except InvalidOperation:
        return None


def score(text, answers, metric):
    pred = extract(text)
    if pred is None:
        return {'quality': 0.0, 'exact_match': 0.0, 'format_ok': False}
    if metric == 'numeric':
        p = numeric(pred)
        q = float(p is not None and any(p == numeric(a) for a in answers))
        return {'quality': q, 'exact_match': q, 'format_ok': True}
    p = normalize(pred)
    em = float(any(p == normalize(a) for a in answers))
    if metric == 'exact':
        return {'quality': em, 'exact_match': em, 'format_ok': True}
    scores = []
    for a in answers:
        g = normalize(a)
        if p in ['yes', 'no', 'noanswer'] or g in ['yes', 'no', 'noanswer']:
            scores.append(float(p == g))
            continue
        pt, gt = p.split(), g.split()
        common = sum((Counter(pt) & Counter(gt)).values())
        scores.append(2 * common / (len(pt) + len(gt)) if pt or gt else 1.0)
    return {'quality': max(scores), 'exact_match': em, 'format_ok': True}


def validate_rows(rows, expected_split=None):
    ids, texts = set(), set()
    if not rows:
        raise ValueError('Dataset is empty')
    for row in rows:
        for key in ('id', 'question', 'context', 'answers', 'metric', 'split', 'source'):
            if key not in row:
                raise ValueError(f'Missing dataset field: {key}')
        if row['id'] in ids or digest(row['question'].strip()) in texts:
            raise ValueError('Duplicate id or question')
        ids.add(row['id']); texts.add(digest(row['question'].strip()))
        if not row['question'].strip() or not isinstance(row['context'], list):
            raise ValueError('Invalid question/context')
        if not row['answers'] or not all(isinstance(x, str) for x in row['answers']):
            raise ValueError('Answers must be a nonempty string list')
        if row['metric'] not in ('numeric', 'f1', 'exact'):
            raise ValueError('Unsupported metric')
        if row['split'] not in ('train', 'dev', 'test') or (expected_split and row['split'] != expected_split):
            raise ValueError('Unexpected split')
        if not isinstance(row['source'], dict) or not row['source'].get('dataset') or not row['source'].get('revision'):
            raise ValueError('Missing source dataset/revision provenance')


def validate_config(c):
    if c.get('study')=='AgentRoute-v10-evidence':
        from .evidence_policy import validate_evidence_config
        return validate_evidence_config(c)
    if c.get('models')!=['Qwen/Qwen3-1.7B'] or c.get('datasets')!=['mmlu']:
        raise ValueError('v9 local-vote release supports Qwen3-1.7B and MMLU MCQ only')
    if c.get('agents')!=4 or c.get('rounds')!=2:
        raise ValueError('Use four agents and at most two worker rounds')
    if len(c.get('roles',[]))<4 or len(set(c['roles'][:4]))!=4:
        raise ValueError('Four distinct roles are required')
    if c.get('controls'):
        raise ValueError('Component ablations are disabled for this release')
    if c.get('routing',{}).get('paid_llm_routing') is not False:
        raise ValueError('Paid routing must be disabled')
    if c.get('local_decision',{}).get('paid_aggregator') is not False:
        raise ValueError('Paid final aggregation must be disabled')
    if c['local_decision'].get('tie_break')!='latest_plurality_then_initial_support_then_agent_order':
        raise ValueError('Unknown deterministic tie-breaking policy')
    if type(c['local_decision'].get('unanimous_early_stop')) is not bool:
        raise ValueError('Early-stop setting must be Boolean')
    if not c.get('seeds') or len(set(c['seeds']))!=len(c['seeds']):
        raise ValueError('Provide distinct seeds')
    if type(c.get('max_tokens')) is not int or c['max_tokens']<=0:
        raise ValueError('Positive output limit required')
