#!/usr/bin/env python3
"""Mechanical evidence audit only. Does not launch experiments or certify science."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

SPEC_PATH = Path(__file__).resolve().parents[1] / 'campaign_spec.json'
HASH_RE = re.compile(r'^[0-9a-f]{64}$')
GEN_OUTCOMES = {'correct', 'incorrect', 'invalid_answer', 'protocol_timeout'}


def audit(run_dir, spec=None):
    root = Path(run_dir).resolve()
    spec = spec or json.loads(SPEC_PATH.read_text())
    errors = []
    evidence_cache = {}

    def error(message):
        errors.append(message)

    def read_json(name):
        try:
            return json.loads((root / name).read_text())
        except (OSError, ValueError) as exc:
            error(f'{name}: {type(exc).__name__}: {exc}')
            return {}

    def check_evidence(items, context, required_roles=()):
        if not isinstance(items, list) or not items:
            error(f'{context}: missing evidence list')
            return
        roles = set()
        for item in items:
            if not isinstance(item, dict):
                error(f'{context}: malformed evidence entry')
                continue
            role, rel, expected_hash = item.get('role'), item.get('path'), item.get('sha256')
            if not isinstance(role, str) or not role:
                error(f'{context}: missing evidence role')
            else:
                roles.add(role)
            if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
                error(f'{context}: evidence path must be a nonempty relative path')
                continue
            p = (root / rel).resolve()
            if not p.is_relative_to(root):
                error(f'{context}: evidence path escapes campaign directory: {rel}')
                continue
            if not isinstance(expected_hash, str) or not HASH_RE.fullmatch(expected_hash):
                error(f'{context}: invalid SHA-256 for {rel}')
                continue
            if p not in evidence_cache:
                try:
                    if not p.is_file() or p.stat().st_size == 0:
                        raise ValueError('not a nonempty regular file')
                    with p.open('rb') as f:
                        evidence_cache[p] = hashlib.file_digest(f, 'sha256').hexdigest()
                except (OSError, ValueError) as exc:
                    evidence_cache[p] = None
                    error(f'{context}: unreadable/empty evidence {rel}: {exc}')
            if evidence_cache[p] != expected_hash:
                error(f'{context}: hash mismatch or unavailable file: {rel}')
        for role in sorted(set(required_roles) - roles):
            error(f'{context}: missing evidence role {role}')

    manifest = read_json('campaign.lock.json')
    if not isinstance(manifest, dict):
        error('campaign.lock.json: expected object')
        manifest = {}
    if manifest.get('frozen') is not True:
        error('campaign manifest is not frozen')
    campaign_id = manifest.get('campaign_id')
    if not isinstance(campaign_id, str) or not campaign_id:
        error('campaign_id is missing')
    if manifest.get('model') != spec['model']:
        error('model does not match required scope')
    if not HASH_RE.fullmatch(str(manifest.get('protocol_sha256', ''))):
        error('protocol_sha256 is missing/invalid')
    if manifest.get('stage_inventory_reviewed') is not True:
        error('cascade stage inventory is not reviewed/frozen')
    stages = manifest.get('additional_stage_ids', [])
    if (not isinstance(stages, list) or
            any(not isinstance(s, str) or not s for s in stages) or
            len(set(stages)) != len(stages)):
        error('additional_stage_ids must be distinct nonempty strings')
        stages = []
    note = manifest.get('stage_inventory_note')
    if not stages and (not isinstance(note, str) or not note.strip()):
        error('S=0 requires a source-linked explanation of full stage coverage')
    expected_counts = dict(spec['generation_counts'])
    expected_counts.update(spec['transport_counts'])
    expected_counts.update(spec['fitting_counts'])
    expected_counts['stage_ablations'] = len(stages) * spec['stage_ablation_count_per_additional_stage']
    tasks = manifest.get('tasks', [])
    if not isinstance(tasks, list):
        error('tasks must be a list')
        tasks = []
    expected = {}
    group_counts = Counter()
    for task in tasks:
        if not isinstance(task, dict):
            error('malformed expected task')
            continue
        key = task.get('task_id')
        if not isinstance(key, str) or not key:
            error('expected task missing nonempty task_id')
            continue
        if key in expected:
            error(f'duplicate expected task ID: {key}')
            continue
        group = task.get('group')
        if not isinstance(group, str):
            error(f'{key}: invalid expected group')
            group = '<invalid>'
        if not HASH_RE.fullmatch(str(task.get('fingerprint', ''))):
            error(f'{key}: invalid expected fingerprint')
        expected[key] = task
        group_counts[group] += 1
    for group in sorted(set(group_counts) | set(expected_counts)):
        if group_counts[group] != expected_counts.get(group, 0):
            error(f'manifest scope {group}: {group_counts[group]} tasks, required {expected_counts.get(group, 0)}')

    rows = []
    try:
        with (root / 'results_index.jsonl').open() as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('row is not an object')
                    rows.append(row)
                except ValueError as exc:
                    error(f'results_index.jsonl line {lineno}: {exc}')
    except OSError as exc:
        error(f'results_index.jsonl: {exc}')
    seen = set()
    accepted = Counter()
    for row in rows:
        key = row.get('task_id')
        if not isinstance(key, str) or key not in expected:
            error(f'unexpected or invalid result task ID: {key!r}')
            continue
        if key in seen:
            error(f'duplicate accepted result ID: {key}')
            continue
        seen.add(key)
        task = expected[key]
        group = task.get('group')
        if not isinstance(group, str):
            error(f'{key}: invalid group in expected task')
            continue
        before = len(errors)
        if row.get('group') != group or row.get('fingerprint') != task.get('fingerprint'):
            error(f'{key}: group/fingerprint does not match frozen task')
        status = row.get('status')
        if not isinstance(status, str) or status not in {'measured', 'verified_reuse'}:
            error(f'{key}: non-accepted status {row.get("status")!r}')
        roles = {'raw', 'metrics'}
        if group in spec['generation_counts'] or group == 'stage_ablations':
            roles.update({'score', 'metering'})
            outcome = row.get('outcome')
            if not isinstance(outcome, str) or outcome not in GEN_OUTCOMES:
                error(f'{key}: not a valid measured generation outcome')
        elif group in spec['fitting_counts']:
            roles.update({'checkpoint', 'metering'})
            if row.get('outcome') != 'trained':
                error(f'{key}: fitting has not completed')
        elif group in spec['transport_counts']:
            if row.get('outcome') != 'transport_measured':
                error(f'{key}: transport repetition has not completed')
        if row.get('status') == 'verified_reuse':
            roles.add('reuse_audit')
        check_evidence(row.get('evidence'), key, roles)
        if len(errors) == before:
            accepted[group] += 1
    missing = sorted(set(expected) - seen)
    if missing:
        error(f'{len(missing)} expected task IDs have no result receipt')

    artifact_index = read_json('artifact_index.json')
    artifacts = artifact_index.get('artifacts') if isinstance(artifact_index, dict) else None
    check_evidence(artifacts, 'artifact_index', spec['required_artifact_roles'])
    coverage = read_json('metrics_coverage.json')
    families = coverage.get('families', []) if isinstance(coverage, dict) else []
    ids = []
    if not isinstance(families, list):
        families = []
    for family in families:
        if not isinstance(family, dict):
            error('malformed metric family')
            continue
        fid = family.get('id')
        ids.append(fid)
        if type(fid) is not int:
            error('metric family ID is not an integer')
        if family.get('status') != 'complete' or family.get('unresolved_required_cells') != []:
            error(f'metric family {fid}: incomplete or unresolved required cells')
        check_evidence(family.get('evidence'), f'metric family {fid}')
    if sorted(x for x in ids if type(x) is int) != list(range(1, 15)) or len(ids) != 14:
        error('metric coverage must contain exactly families 1 through 14')

    resources = read_json('resources.json')
    if not isinstance(resources, dict):
        resources = {}
    if resources.get('campaign_id') != campaign_id or resources.get('registry_complete') is not True:
        error('resource registry is not complete or belongs to a different campaign')
    inventory = resources.get('resources', [])
    if not isinstance(inventory, list) or not inventory:
        error('resource registry must enumerate owned campaign resources')
        inventory = []
    resource_ids = set()
    for resource in inventory:
        if not isinstance(resource, dict):
            error('malformed resource entry')
            continue
        rid = resource.get('id')
        if not isinstance(rid, str) or not rid or rid in resource_ids:
            error('missing/duplicate resource ID')
        else:
            resource_ids.add(rid)
        if resource.get('owned') is not True or resource.get('state') != 'terminated':
            error(f'resource {rid}: shutdown not confirmed for owned resource')
        if not isinstance(resource.get('observed_at'), str) or not resource.get('observed_at'):
            error(f'resource {rid}: missing provider observation timestamp')
        check_evidence(resource.get('evidence'), f'resource {rid}')

    return {
        'status': 'MECHANICAL_CHECK_PASSED' if not errors else 'INCOMPLETE_OR_INVALID',
        'experiment_evidence_contract_passed': not errors,
        'scientific_claims_verified': False,
        'scope_note': 'Checks bookkeeping and artifact integrity, not truth of file contents, baseline fidelity or cloud provider state.',
        'expected_by_group': expected_counts,
        'accepted_by_group': {g: accepted[g] for g in expected_counts},
        'missing_task_ids': missing,
        'errors': errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', help='Optional report path; prints JSON in all cases')
    args = parser.parse_args()
    result = audit(args.run_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + '\n')
    print(rendered)
    return 0 if result['experiment_evidence_contract_passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
