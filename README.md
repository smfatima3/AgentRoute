# AgentRoute — Qwen3-1.7B evidence study

Reproducible multi-agent reasoning experiments with deterministic answer-aware communication, matched dense/no-communication controls, and SafeSieve, CARD, SelfOrg and AgentNet adaptations.

**Status (2 November 2026): model evaluations and their statistical analysis are complete; physical transport is incomplete (0/90 accepted); the latest verified coordinator exited with code 137. This is not a completed communication-bottleneck study.**

## Scope and current results

The retained evaluation uses Qwen/Qwen3-1.7B, 100 selected MMLU questions and 50 selected BBH questions. Counts below are question-level accuracy; the full retained matrix contains 1,540 method/question/seed evaluations including ablations and controls. MATH and SVAMP were excluded by the operator; their older evidence is preserved in historical data assets and is not combined with these results.

| Method | MMLU correct / 100 | MMLU mean tokens | BBH correct / 50 | BBH mean tokens |
|---|---:|---:|---:|---:|
| AgentRoute | 63 | 2,726.8 | 11 | 3,112.1 |
| Dense | 63 | 2,976.7 | 12 | 3,567.4 |
| No communication | 64 | 2,344.6 | 10 | 2,334.4 |
| Single agent | 58 | 354.2 | 9 | 346.4 |
| SafeSieve | 62 | 5,302.3 | 11 | 4,325.9 |
| CARD | 61 | 5,199.9 | 10 | 4,242.8 |
| SelfOrg | 60 | 7,933.8 | 8 | 6,000.1 |
| AgentNet | 76 | 5,215.4 | 9 | 4,991.5 |

### What the results support

- AgentRoute uses **8.4% fewer online tokens than dense on MMLU** and **12.8% fewer on BBH**. Paired bootstrap 95% intervals are 4.9–11.5% and 9.2–16.4%, respectively.
- AgentRoute uses **16.3% more tokens than no communication on MMLU** and **33.3% more on BBH**. A general token-efficiency advantage is therefore not established.
- MMLU accuracy is 63% for AgentRoute and 76% for AgentNet; the paired exact McNemar comparison remains significant after Holm correction (p = 0.01645). Lower token use does not establish equal accuracy.
- The prespecified two-percentage-point noninferiority screen is inconclusive. These small selected panels do not establish population-level accuracy preservation.
- A correct initial candidate disappeared in the final AgentRoute answer on 11/74 eligible MMLU questions and 1/12 eligible BBH questions. This is descriptive evidence, not proof that aggregation alone caused those losses.

See [the quality report](results/quality/QUALITY_REPORT.md), [complete statistics](results/quality/QUALITY_STATISTICS.json), [question-level results](results/quality/QUESTION_RESULTS.csv), and [offline rescoring audit](results/quality/INDEPENDENT_RESCORE.json). All 1,540 stored outcomes matched the frozen scorer on recomputation.

## Physical transport

The physical workload uses two CPU VM sandboxes in distinct provider-reported US-East and US-West regions. The kernel-shaped link was validated at 9.38 Mbit/s and 50.64 ms additional round-trip delay. The matrix contains 90 independent repetitions, including four/eight/sixteen-agent payload replay and low/high offered loads.

The current run is **in progress**. Link calibration is not completion of the 90 repetitions. Replay transports real recorded payloads, control messages and acknowledgements; it does not execute new LLM reasoning or establish live multi-agent accuracy. There is no universal bottleneck-free or zero-overhead claim. Final transport results will be added after receipt validation and resource shutdown.

## Repository layout

- `work/agentroute_v12/package/`: frozen executed model policy, adapters, pinned requirements, task panels and compatible baseline fitting checkpoints.
- `work/agentroute_v12/`: coordinator, leases/receipts, telemetry, cloud checkpoints, transport and analysis tools, and tests. Earlier operational versions are retained where needed to explain runtime lineage.
- `work/v9_local/.../modal_runner/`: small retained serving/image helper dependency; this is not a second current experiment.
- `docs/execution_guide/`: the adopted experiment specification and completion rules.
- `results/quality/`: completed quality analysis and evidence hashes.
- `results/provenance/`: frozen task/config/source manifests and scope amendments.
- `data/RELEASE_ASSETS.json`: manifests for complete raw-data/history archives distributed as GitHub release assets.

## Raw data and recovery

Large raw results are prepared for publication as version-labelled assets on the repository's [Releases page](https://github.com/smfatima3/AgentRoute/releases). Every local results directory is included in the transfer, including historical/failed evidence. Historical results are never relabelled as the current study. Download the desired assets, compare SHA-256 to `data/RELEASE_ASSETS.json`, and extract at the repository root. Each new archive also contains a per-file `TRANSFER_MANIFEST.json`.

The `agentroute_v12_completion.zip` snapshot includes 1,540 hash-verified raw model records under `outputs/agentroute_v12_completion/analysis_inputs/raw/`; filenames are content hashes to avoid Windows path-length problems. The original scientific task IDs and source paths remain in the receipt and download manifests. A later transport-data snapshot is pending; a recovery launch has not yet been verified.

## Reproduce the statistics

Use Python 3.13 and an isolated environment. Install the analysis dependencies, restore the v12 data asset, then run:

```bash
python -m pip install numpy==2.5.4 scipy==1.18.1 matplotlib==3.11.2
python work/agentroute_v12/analyze_closed_quality.py
```

The script checks every raw record hash and exact fixed-panel membership before scoring. The analysis uses question-level Wilson intervals, exact paired McNemar tests with seven-comparison Holm correction per dataset, conservative paired quality intervals, stratified paired token bootstraps, and question-clustered seed analysis. It refuses missing panels and scoring disagreements. No model calls occur during analysis.

## Running new experiments

Read the execution guide and `work/agentroute_v12/AGENTS.md` first. The controller scripts preserve the original research workspace layout and include live-campaign resource references; review paths, ownership, budget and model/data/source locks before creating a new campaign. Authenticate Modal using its normal credential store or process environment. Never place credentials in this repository. The current campaign does not require any GPU restart.

Checkpoints are saved in the cloud every five minutes, with result archives every twenty minutes and closed task records committed more frequently. Provider timeouts and monetary reservations bound resources independently of an interactive coding session.

## Baseline fidelity and limitations

Baseline adapters use pinned upstream source with a common-task boundary and verified learned state. These are documented adaptations, not established native-paper reproductions. Fitting, inference, warmup, retry and transport accounting are distinct. Old and new latency cohorts have different instrumentation and must not be pooled as a clean timing comparison. Tail estimates on 50/100 questions are not capacity measurements.

Third-party source retains its original notices and licenses in the vendored directories. Publication of these files does not relicense third-party work.

Public copies omit Git object metadata and redact an embedded third-party credential. Scientific outcomes are unchanged; see [the explicit redaction map](data/PUBLIC_REDACTIONS.json). Original private hashes are preserved there.
