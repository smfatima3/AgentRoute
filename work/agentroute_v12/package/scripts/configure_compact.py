"""One immutable, prospective compact protocol, prior to model predictions."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agentroute.core import write_json,file_hash

def main():
    dest=ROOT/'configs/compact.json'
    if dest.exists():raise RuntimeError('Do not overwrite a frozen configuration')
    c=json.loads((ROOT/'configs/evidence.json').read_text(encoding='utf-8'))
    c.update(study='AgentRoute-compact-v11',phase='compact',datasets=['mmlu','svamp','math','bbh'],
        methods=['cascade','dense','no_comm','safesieve','card','selforg','agentnet','single'],
        controls=['original','corroborating','route_fixed','dense_fixed','random_fixed','toggle_filter'],
        sample_seed=20261029,noninferiority_margin=.02)
    def arm(mode,stop=True,dissent=True):return dict(communication=mode,aggregation='vote',early_stop=stop,filter_dissent=dissent,group_answers=True,k=3)
    c['evidence_arms']={'original':arm('route'),'corroborating':arm('route',dissent=False),
        'cascade':arm('route'),'dense':arm('dense'),'no_comm':arm('none'),
        'route_fixed':arm('route',False),'dense_fixed':arm('dense',False),'random_fixed':arm('random',False),
        'toggle_filter':arm('route',dissent=False)}
    c['selforg_variant']='Pinned original SelfOrg implementation with DAG reform, weighted aggregation and its configured consensus rule; not the separately named Efficient SelfOrg variant'
    c['local_decision']['supported_metric']=['math','exact','numeric','mcq']
    c['confirmatory_status']='MMLU selected-panel primary; transfer panels and component controls exploratory; adaptation, not native-paper reproduction'
    write_json(dest,c)
    write_json(ROOT/'configs/compact_protocol.json',{
        'created_unix_s':time.time(),'config_sha256':file_hash(dest),'data_manifest_sha256':file_hash(ROOT/'compact_data/manifest.json'),
        'hard_deadline_utc':'2026-10-29T22:53:34Z','admission_deadline_utc':'2026-10-29T22:38:34Z',
        'target_finish_utc':'2026-10-29T22:23:34Z','gpu_seconds_cap':32400,'scored_attempt_cap':2700,
        'total_token_cap':25000000,'fit_token_cap':2000000,'replay_vcpu_seconds_cap':115200,
        'primary_seed':17,'primary_dataset':'mmlu','bootstrap_repetitions':10000,
        'statistical_unit':'question; repeated seeds cluster by question; equal selected-question weights within dataset',
        'bootstrap_strata':{'mmlu':'subject','math':'topic/difficulty','bbh':'task','svamp':'type'},
        'small_strata_warning':'Singleton strata have no within-stratum empirical uncertainty; conservative independent-question binary interval reported separately, with sampling assumptions',
        'noninferiority_margin':.02,
        'margin_justification':'Prospective screening tolerance of at most 2 additional wrong answers per 100 questions, chosen as a small accuracy loss for an efficiency study. This is not a user-approved deployment loss tolerance. Report strict zero-loss comparison too; never widen this margin. Small panels are expected to be inconclusive.',
        'candidate_rule':'Choose corroborating only if total correct across all 30 development questions is at least original and its total charged model tokens are no more than 1.25 times original; otherwise original. One decision, no item-specific tuning.',
        'development_negative_gate':'Candidate has no positive paired accuracy difference supported by exact two-sided McNemar p<0.05 versus no_comm AND candidate total tokens >= no_comm. Finish full main MMLU regardless; if triggered, other GPU panels remain deferred and run low-cost transport only.',
        'main_useful_tradeoff':'Against no_comm: either positive accuracy difference with Holm-adjusted exact McNemar p<0.05 (seven primary comparisons), or strictly lower total tokens with conservative paired accuracy lower bound > -0.02. Otherwise stop optional expansion. This screen does not permit claiming equivalence from nonsignificance.',
        'slo_rule':'After development only, ceil pooled dense inference p95 / 5 seconds times 5, minimum 5 seconds. Freeze before held-out predictions; identical threshold for every method.',
        'failure_rule':'No implicit retry. Preserve first attempt and all costs. Invalid outputs scored wrong. Infrastructure/scorer errors retained with quality zero in intention-to-evaluate denominator and separately flagged; stop campaign on infrastructure faults for audited repair, never silently skip.',
        'initial_response_rule':'Fresh inference for every cell; identical per-question/agent seeds. Prefix cache explicitly disabled and two identical fixed warmup calls before each cell. Compare actual response bytes; report nondeterminism. No cached response reuse or simulated latency.',
        'checkpoint_reuse_sha256':file_hash(ROOT/'checkpoint_reuse.json'),
        'random_control_limitation':'k=3 matches unmodified v10 degree ceiling for four agents; random control is dense after canonical sender sorting, so this matrix does not test useful random sparsity.',
        'selforg_variant':c['selforg_variant'],
        'round_mapping':'Controlled arms: 4 initial plus at most 4 revision calls. SafeSieve/CARD: upstream graph num_rounds=2 plus native final aggregation. SelfOrg: 4 initial plus at most 2 propagation rounds, native consensus and weighted selector. AgentNet: forward_path_max_length=3/max_execution_times=5/max_calls=80; native routing/evolution preserved, frozen trained state per question.',
        'cost_note':'New calls count toward campaign caps. Reused historical successful and failed fitting costs reported separately and amortized; not claimed free.'})
    print('Prospective compact protocol written')
if __name__=='__main__':main()
