"""Artificial fixtures exercise bookkeeping only; they are not experiment data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location('completion', BASE / 'tools/check_completion.py')
completion = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(completion)
SPEC = json.loads((BASE / 'campaign_spec.json').read_text())
H = hashlib.sha256(b'UNIT TEST CONFIGURATION, NOT AN EXPERIMENT').hexdigest()


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = b'UNIT TEST EVIDENCE ONLY; NO SCIENTIFIC OR CLOUD CLAIMS\n'
        (self.root / 'fixture.txt').write_bytes(data)
        self.ev = {'path': 'fixture.txt', 'sha256': hashlib.sha256(data).hexdigest()}
        counts = {**SPEC['generation_counts'], **SPEC['transport_counts'], **SPEC['fitting_counts'], 'stage_ablations':40}
        tasks = []
        self.rows = []
        for group, count in counts.items():
            for number in range(count):
                key = f'{group}:{number}'
                tasks.append({'task_id':key, 'group':group, 'fingerprint':H})
                if group in SPEC['transport_counts']:
                    roles, outcome = ['raw','metrics'], 'transport_measured'
                elif group in SPEC['fitting_counts']:
                    roles, outcome = ['raw','metrics','checkpoint','metering'], 'trained'
                else:
                    roles, outcome = ['raw','metrics','score','metering'], 'incorrect'
                self.rows.append({'task_id':key,'group':group,'fingerprint':H,'status':'measured','outcome':outcome,'evidence':[dict(self.ev,role=role) for role in roles]})
        self.manifest = {'frozen':True,'campaign_id':'unit-test','model':SPEC['model'],'protocol_sha256':H,'stage_inventory_reviewed':True,'additional_stage_ids':['fixture_stage'],'tasks':tasks}
        self.write('campaign.lock.json', self.manifest)
        self.write_rows()
        self.write('artifact_index.json', {'artifacts':[dict(self.ev,role=r) for r in SPEC['required_artifact_roles']]})
        self.coverage = {'families':[{'id':i,'status':'complete','unresolved_required_cells':[],'evidence':[dict(self.ev,role='coverage')]} for i in range(1,15)]}
        self.write('metrics_coverage.json',self.coverage)
        self.resources = {'campaign_id':'unit-test','registry_complete':True,'resources':[{'id':'fixture-resource','owned':True,'state':'terminated','observed_at':'2026-10-31T00:00:00Z','evidence':[dict(self.ev,role='provider_state')]}]}
        self.write('resources.json',self.resources)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, obj):
        (self.root / name).write_text(json.dumps(obj))

    def write_rows(self):
        with (self.root / 'results_index.jsonl').open('w') as f:
            for row in self.rows:
                f.write(json.dumps(row)+'\n')

    def check(self):
        return completion.audit(self.root)

    def assert_incomplete(self, substring):
        result = self.check()
        self.assertFalse(result['experiment_evidence_contract_passed'])
        self.assertTrue(any(substring in e for e in result['errors']), result['errors'][:5])

    def test_complete_contract_all_incorrect_is_accepted_not_scientifically_certified(self):
        result = self.check()
        self.assertTrue(result['experiment_evidence_contract_passed'], result['errors'])
        self.assertFalse(result['scientific_claims_verified'])
        self.assertEqual(sum(result['accepted_by_group'].values()), 2933)

    def test_v11_count_gap_cannot_be_hidden_by_finished_archive(self):
        remove = {f'main_mmlu:{i}' for i in range(710,800)} | {f'transport_scaling:{i}' for i in range(39,72)}
        self.rows = [r for r in self.rows if r['task_id'] not in remove]
        self.write_rows()
        self.write('ARCHIVE_VERIFICATION.json', {'all_members_verified':True,'scope':'archive only'})
        result = self.check()
        self.assertFalse(result['experiment_evidence_contract_passed'])
        self.assertEqual(result['accepted_by_group']['main_mmlu'],710)
        self.assertEqual(result['accepted_by_group']['transport_anchors']+result['accepted_by_group']['transport_scaling'],57)
        self.assertEqual(len(result['missing_task_ids']),123)

    def test_shrinking_manifest_fails_fixed_scope(self):
        self.manifest['tasks'] = self.manifest['tasks'][:-1]
        self.rows = self.rows[:-1]
        self.write('campaign.lock.json',self.manifest)
        self.write_rows()
        self.assert_incomplete('manifest scope stage_ablations')

    def test_duplicate_receipt_is_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.write_rows()
        self.assert_incomplete('duplicate accepted result ID')

    def test_wrong_configuration_is_rejected(self):
        self.rows[0]['fingerprint'] = 'f'*64
        self.write_rows()
        self.assert_incomplete('does not match frozen task')

    def test_infrastructure_error_is_not_an_answer(self):
        self.rows[0]['outcome'] = 'model_initialization_failed'
        self.write_rows()
        self.assert_incomplete('not a valid measured generation outcome')

    def test_missing_metric_family_fails(self):
        self.coverage['families'].pop()
        self.write('metrics_coverage.json',self.coverage)
        self.assert_incomplete('exactly families 1 through 14')

    def test_unresolved_measurement_fails(self):
        self.coverage['families'][5]['unresolved_required_cells'] = ['wire_bytes']
        self.write('metrics_coverage.json',self.coverage)
        self.assert_incomplete('incomplete or unresolved')

    def test_active_resource_fails(self):
        self.resources['resources'][0]['state'] = 'running'
        self.write('resources.json',self.resources)
        self.assert_incomplete('shutdown not confirmed')

    def test_corrupted_evidence_fails(self):
        (self.root / 'fixture.txt').write_text('changed bytes')
        self.assert_incomplete('hash mismatch')

    def test_path_escape_fails(self):
        self.rows[0]['evidence'][0]['path'] = '../outside.txt'
        self.write_rows()
        self.assert_incomplete('escapes campaign directory')

    def test_unverified_reuse_fails(self):
        self.rows[0]['status'] = 'verified_reuse'
        self.write_rows()
        self.assert_incomplete('missing evidence role reuse_audit')

    def test_unfrozen_protocol_fails(self):
        self.manifest['frozen'] = False
        self.write('campaign.lock.json',self.manifest)
        self.assert_incomplete('not frozen')

    def test_missing_fit_fails(self):
        self.rows = [r for r in self.rows if r['task_id'] != 'baseline_fitting:2']
        self.write_rows()
        self.assert_incomplete('1 expected task IDs have no result receipt')

    def test_malformed_status_fails_without_crashing(self):
        self.rows[0]['status'] = {'unexpected':'object'}
        self.write_rows()
        self.assert_incomplete('non-accepted status')

    def test_empty_directory_is_not_complete(self):
        with tempfile.TemporaryDirectory() as other:
            result = completion.audit(other)
            self.assertFalse(result['experiment_evidence_contract_passed'])


if __name__ == '__main__':
    unittest.main()
