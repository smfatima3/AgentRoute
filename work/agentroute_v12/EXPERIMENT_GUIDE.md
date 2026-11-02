# AgentRoute v12: concurrent execution and complete evidence

31 October 2026. This is a redesign of the execution guide, not a report of newly executed experiments.

## What the supplied evidence establishes

`FINAL_COMPLETION.json` reports 710 of 800 primary attempts, 57 of 90 transport repetitions, an incomplete experiment matrix and stopped owned resources. `ARCHIVE_VERIFICATION.json` describes a verified archive of a capped, incomplete study. Both name SHA-256 `dae1e2d1f445b299bf717fdb40cc5a9321c99d57931557a9fb7dc48a683505bb`, 244,476,558 bytes and 165 verified members. These are claims in the supplied reports; the ZIP itself was unavailable to this guide author and has not been independently verified here.

There are at least 90 primary slots and 33 transport repetitions not marked complete in those reports. The identities, validity and other blocks' completion are unknown. The executor must recover raw evidence before awarding reuse credit. Finishing an export, deleting a heartbeat or generating a report does not prove experiment completion or resource termination.

## Fixed scientific scope

Use **Qwen/Qwen3-1.7B**, four reasoning agents and a two-round maximum for the controlled comparisons. Retain MMLU 100, SVAMP 60, MATH 50 and BBH 50: **260 distinct held-out questions**. Do not silently expand to full benchmarks or the 8B/14B models. Those are separate future campaigns.

Evaluate all eight main systems: AgentRoute, matched dense/broadcast, matched no communication, SafeSieve, CARD, SelfOrg, AgentNet and single agent. Fix SelfOrg's variant explicitly and run the alternative original/efficient variant in an additional control. Verify exact upstream identities and revisions from source; AgentNet's previously verified source was Yang et al.'s AgentNet, not an assumed paper matching a similar name. Record available tools, training, topology, stopping and aggregation differences. These are common-task adaptations, not automatically native-paper reproductions.

Freeze one source-task fitting protocol: three learned-baseline jobs (SafeSieve, CARD, AgentNet), each using the prespecified disjoint source pool of 120 MMLU development questions, with the full fixed fitting recipe. Reuse compatible validated checkpoints. Count all fitting trajectories, iterations, tokens and resources; do not truncate training to the old two-million-token cap. Complete the recipe or report a blocked fit. SelfOrg and the frozen AgentRoute policy do not acquire an invented fitting requirement. Subsequent datasets test transfer of source-trained routing without target-test tuning; this limitation belongs beside their results.

Reuse v11 IDs and outcomes only if the policy, source, prompts, model, scorer, data and evaluation protocol remain compatible. Old test outcomes already inspected for policy selection cannot become fresh confirmatory evidence for a revised policy. Freeze fresh IDs when necessary, keeping old evidence separate.

## Required generation matrix

| Group key | Experiment | Size | Logical generation tasks |
|---|---|---|---:|
| development | 20 MATH + 10 BBH development questions; original policy, dissent-filter toggle, matched dense, matched no communication | 30 × 4 | 120 |
| main_mmlu | All eight methods, seed 17 | 100 × 8 | 800 |
| transfer_svamp | All eight methods, seed 17 | 60 × 8 | 480 |
| short_math | All eight methods, seed 17 | 50 × 8 | 400 |
| short_bbh | All eight methods, seed 17 | 50 × 8 | 400 |
| component_ablations | Fixed-round routing; fixed-round dense; random-k fixed-round routing; dissent toggle | 40 × 4 | 160 |
| seed_sensitivity | AgentRoute, dense, no communication; seeds 29 and 43 | 40 × 3 × 2 | 240 |
| live_anchors | AgentRoute/dense; 4/8 agents; ten development requests each | 10 × 2 × 2 | 40 |
| aggregation_controls | Fixed-round routing + paid final aggregator; fixed-round dense + same paid aggregator | 40 × 2 | 80 |
| self_consistency | Independent single-agent samples and local selection under a prespecified token budget | 40 × 1 | 40 |
| selforg_variant | Alternative published SelfOrg original/efficient variant, labeled explicitly | 40 × 1 | 40 |
| stage_ablations | Each additional distinct implemented cascade-stage removal | 40 × S | 40S |
| **Total** | **Before valid reuse; model calls per task vary** | | **2,800 + 40S** |

Use the same 40 ablation/seed/control IDs, 20 MMLU and 20 MATH, selected before inspecting their outcomes. Reuse the matched full-policy rows already in the main panels. The four fixed/local-versus-paid routing/dense conditions form a 2×2 routing/aggregation comparison. Do not rerun the local fixed-round conditions merely to populate another table.

Discover S from the actual source before freezing the plan. For every implemented cascade stage, specify what is bypassed and the default behavior after bypass. A stage removal identical to a named component arm is mapped to that arm once; count only additional distinct settings in S. S=0 is allowed only with an explicit source-linked explanation showing every actual stage already covered. Do not invent stage names or omit expensive candidate-scoring work. Stage tests may change topology or payload; measure those changes rather than claiming perfectly controlled token budgets.

For self-consistency, derive a fixed per-dataset total inference-token budget from development runs, not the test answer or a favorable realized outcome. Stop admitting new samples when the next sample would exceed a reserved bound, count all input/output tokens and report actual spend and unused budget. Match AgentRoute's budget approximately and visibly; do not claim an exact compute match if it was not achieved. Use the same admissible offline scoring rule, with no gold-assisted answer selection.

Development checks diagnose scorer equivalence, incorrect dissent amplification, cache/seed differences, no-communication isolation and stopping. Freeze the candidate once. Validation failures require repair before evaluation; unfavorable scientific results do not cancel remaining blocks. Existing compatible completed development work can satisfy this requirement, avoiding repeated tuning.

A task is an assigned question/method/seed/configuration, not a model call or a successful answer. Incorrect, invalid and protocol-timeout outcomes remain in denominators. Infrastructure faults remain separately recorded and trigger bounded recovery. Training jobs and CPU replay are counted separately from generation tasks.

## Concurrent sandbox architecture

| Execution role | Resource | Work and ownership |
|---|---|---|
| Coordinator | Operator process or one small CPU sandbox | Single task/lease owner, cost/resource registry, checkpoint journal, merged reports |
| GPU-A | One GPU | One model replica; leased fitting or quality shards |
| GPU-B | One matching GPU | Independent replica; leased fitting or quality shards concurrently |
| CPU-NET-A | CPU host | Transport workers/load source, one measurement condition at a time |
| CPU-NET-B | Different CPU host | Remote transport workers; measured network path |
| Analysis | Coordinator/local CPU or a temporary CPU worker | Statistics and plots from closed immutable shards |

Reuse the pilot's GPU type if available, or select one type for both new workers and document the change. Two L40S sandboxes or two A100-40GB sandboxes are options, not interchangeable timing evidence. Do not assume two GPU sandboxes share one physical GPU or that two sandboxes imply separate physical hosts. Validate residency for distributed claims.

Start both GPU workers asynchronously and wait for each to prove health, model identity and metering. Default to two concurrent questions per replica after development tuning. Randomize balanced method/question shards across both devices. Do not pin all AgentRoute work to GPU-A and all baselines to GPU-B. Preserve load, warmup, batching and cache conditions; record unavoidable differences.

The coordinator leases ready shards and enforces dependencies: validated fits precede dependent baseline evaluation; frozen policy precedes all test use; completed trace sources precede replay; closed paired records precede statistics. Fill free capacity with independent ready work. CPU replay can overlap quality runs only with independent resources and no relevant shared-link contention. Timed runs on the transport pair are serialized; parallelizing them on the same path would measure interference from the experiment runner.

Use one mutable coordinator database and unique immutable worker output prefixes. Modal's documented Volume semantics require commit/reload handling and do not provide distributed file locking. Never concurrently append to a shared results file or run multiple SQLite writers across a Volume. Pin and test the SDK version, lease fencing, checkpoint visibility and crash recovery. Asynchronous dispatch is an execution mechanism, not proof of speedup.

## Recovery, budget and completion behavior

Import v11 results into an immutable namespace. Build `REUSE_AUDIT.json` mapping each accepted task and measurement to raw records and fingerprints, and each rejected reuse to its reason. Recompute scores from raw outputs. Missing telemetry may require a separate live measurement even when an answer is reusable. Do not assign an old generation's latency to a new measured run.

Use bounded task shards, durable receipts and retry fencing. Archive every infrastructure attempt. A process lost after generation but before acknowledgment must not silently cause duplicate accepted results or unmetered duplicate work. Deduplicate logical outcomes, not the actual cost ledger. Test recovery from a deliberately interrupted development shard before the full queue.

The old arbitrary study-wide resource cutoffs are replaced by completion of this finite manifest. Keep user/provider/account monetary caps. Display estimated remaining compute and cost with fitting, startup, CPU/RAM, retries, storage and transfer assumptions; use observed throughput rather than promising that two workers halve the bill. Cap active GPU workers at two and shut down idle owned resources. Use bounded sandbox lifetimes and checkpoints so replacing an expired sandbox continues the study rather than ends it. A genuine account limit produces a resumable blocker, never a success flag.

## Fourteen required metric families

Record per-question measurements for all main methods and all datasets, plus detailed systems traces where applicable. Statistical families are derived after data collection. Families are not fourteen separate GPU sweeps.

| ID | Metric family | Required evidence |
|---:|---|---|
| 1 | Quality | Correct/total, accuracy, answer validity and uncertainty; preserve failures |
| 2 | Paired quality change | Per-question gains/losses, difference, CI, corrected tests, noninferiority outcome |
| 3 | Tokens | Prompt/completion/total by purpose; reasoning-token convention; per-task reconciliation |
| 4 | Model calls | Worker/router/aggregation/fitting/retry counts and unique response references |
| 5 | Deliveries | Per-recipient messages, attempts, successes, fanout and rounds |
| 6 | Bytes | Serialized peer/control payload and measured link bytes including overhead; peer vs inference traffic separated |
| 7 | Routing cost | CPU/wall time, peak memory and actual per-stage/candidate-scoring overhead |
| 8 | End-to-end latency | p50/p95/p99, sample counts, cold/warm conditions and uncertainty limitations |
| 9 | Queues/critical path | Admission, routing, transport and serving waits, causal spans, clock uncertainty |
| 10 | Throughput | Offered/completed/failed requests per second at declared concurrency/load |
| 11 | SLO goodput | Live correct-and-timely completions/s; separately labeled transport delivery goodput |
| 12 | Resources | CPU/RSS, GPU utilization/memory, NIC counters and sampling coverage |
| 13 | Cost | Fit/startup/run/retry/shutdown resource time, dated price estimates, bill if available, amortized totals |
| 14 | Reliability | Invalid/timeout/truncation/retry/drop/transport-error rates and all denominators |

Each family gets a coverage table by method, dataset and measurement mode. Estimates are permitted only for inherently estimated quantities such as rate-based costs or amortization, with inputs and formulas. Actual wire bytes, resource traces and live timings require measurement. Genuine zero peer messages is valid for no communication/single agent; zero peer latency is not a substitute for not-applicable. Any required unavailable field must be filled through valid measurement or remain an explicit blocker. A small sample's wide interval is a valid result, not missing data.

Batch throughput from accuracy runs is not maximum serving capacity. p99 computed from 50 answers is not a reliable tail estimate. Store the statistic and its limitations; use the transport repetitions for better tail characterization within that subsystem. Do not transfer replay tail results to live LLM execution.

## Full communication scaling matrix: 90 repetitions

1. **Baseline anchors:** six communicating methods, four agents, one frozen ordinary network/low-load setting, three independent repetitions: 18 repetitions.
2. **Scaling:** AgentRoute/dense × 4/8/16 agents × ordinary/constrained network × low/high offered load × three independent repetitions: 72 repetitions.
3. Keep all 90 IDs. Even if an anchor matches a scaling configuration, these are independently scheduled repetition IDs; do not satisfy two independent repetitions with one trace. This choice avoids ambiguous 57-of-90 recovery bookkeeping.
4. Use verified separate CPU hosts. Constrained target: a measured 10-Mbit/s link cap and 25-ms additional one-way delay. Freeze tolerance and validate achieved shaping. If the chosen service cannot provide real shaping/host isolation, use a suitable already-authorized execution environment or mark the physical experiment blocked. Application-delay emulation can be supplemental, never a substitute credited as physical-network completion.
5. CPU calibration ramps establish fixed common low/high arrival rates per N/network. Freeze rates before comparing methods. Low load should be clearly below dense capacity and high load should probe its saturation. A short bounded ramp must show the chosen bracket; if saturation lies outside the tested range, report that, not a fabricated threshold.
6. Replay actual recorded payloads and communication schedules, including central manager, global summaries, acknowledgements, retries and aggregation traffic. Hash source payloads. Publish the mapping used to scale four-agent traces to 8/16 agents. Such replay measures transport behavior under a generated workload, not accuracy of an unexecuted 16-agent team.
7. Per repeat: 10-second warmup, at least 60 seconds steady state, at least 1,000 scheduled arrivals, then drain. Extend duration at a low rate to reach the fixed arrival count; do not silently shorten it to fit a time cap. Set per-request deadlines and finite queue bounds in advance. Record scheduled/accepted/completed/dropped/timed-out/unfinished requests. Open-loop arrivals must not slow down when the system does, avoiding coordinated omission.
8. Randomize order, use distinct scheduling seeds and record variation across three repetitions. Report latency distributions, delivery integrity, throughput, goodput, queue growth, bandwidth and manager/router resource use. p99 still needs uncertainty; 3,000 arrivals does not make it exact.
9. Run four live anchor conditions from the generation matrix: two methods × N=4/8, ten development requests each, actual peer transport plus actual inference on the constrained setup. Reserve the network pair and one GPU endpoint per condition; do not share that endpoint with unrelated timed work. These checks connect subsystem replay to live behavior but do not establish live p99 precision.

The scientific claim is measured communication reduction/scaling within a stated load/hardware/payload envelope. Finite systems always have a capacity limit; do not claim universal absence of communication bottlenecks. Count all-pairs scoring and the coordinator, not just sparse peer edges. If a code-verified degree cap permits a bound R*N*min(k,N-1) on peer deliveries, state its assumptions and separately account for control traffic/retries. A message-count bound does not establish queue stability.

## Statistical analysis and interpretation

Use seed 17 as the primary paired quality comparison and name MMLU as the primary dataset. Report each other panel and each ablation separately as exploratory. Freeze panel weighting and task coverage before results. These deliberately small, stratified panels are not full-dataset scores; tiny BBH strata limit generalization and interval precision.

Compute per-method accuracy intervals, paired differences, gain/loss tables, exact McNemar tests and Holm adjustment for the seven main comparisons within each dataset. Use 10,000 paired question resamples for token ratios `1 - sum(AR)/sum(reference)` and appropriate accuracy-difference intervals. Retain paired IDs and fixed stratum weights. For independent-question binary settings, include a suitable paired score or conservative gain/harm interval with its assumptions; do not treat a degenerate [0,0] empirical bootstrap as proof of equality when no discordance was observed.

Report seeds 29/43 on the shared 40-question subset separately and use question-cluster resampling if combining seeds. Repeated generations are not new independent questions. For system time series, use run/block uncertainty; requests in a burst are correlated.

Prespecify an application-justified acceptable accuracy loss. A 2-percentage-point margin is a candidate, not an automatic fact. Noninferiority needs its one-sided lower bound above minus that margin. Failure to reject a difference does not establish preservation. This small campaign may be fully executed with an INCONCLUSIVE accuracy-preservation result. If tight certification is desired afterward, design a new adequately powered held-out study; do not keep sampling until significance appears.

Use independently rescored final answers, ledger reconciliation and invalid-output diagnosis. Show accuracy/cost Pareto plots, routing-only versus stopping/aggregation effects, transfer limitations and comparison to no communication/cost-matched self-consistency. A negative result can establish a useful limit; do not require AgentRoute to win before allowing completion.

## Required output bundle

- `RESULTS.md`: complete numerical tables by method/dataset, all uncertainty and failures; denominators explicit.
- `statistics.json`, `ablation_results.json`, `scaling_results.json` and rendered figures.
- `METHOD_FIDELITY.md`, `REUSE_AUDIT.json`, source/config/data/checkpoint lock and training logs.
- Raw attempts/responses, traces, metering, resource sampling, exact split IDs and immutable result receipts.
- `metrics_coverage.json`, full cost ledger, `RESOURCE_SHUTDOWN.json`, `STATUS.json`, `RESUME.md`.
- `campaign.lock.json`, `results_index.jsonl`, `artifact_index.json`, checker output, `ARCHIVE_VERIFICATION.json` and SHA-256 manifest.

Completion is the intersection of the frozen task universe, valid receipts, required measured evidence, derived reports and verified cleanup. Archive integrity is a separate property. The final report must state execution completeness, measurement completeness and scientific claim status independently. Mobility buffering, actor migration, privacy and model-size generalization are outside this finite plan and must not be claimed as established.

## Completion-checker integration contract

`campaign_spec.json` specifies fixed counts. The executor expands exact tasks after inspecting source and IDs. Its `campaign.lock.json` must contain `frozen: true`, `campaign_id`, `model`, `protocol_sha256`, `stage_inventory_reviewed: true`, `additional_stage_ids` and `tasks`. Each task has `task_id`, `group`, and a configuration `fingerprint` (SHA-256). Use a unique task ID derived from the full scientific key and configuration, including repetition identity. Required groups/counts are checked against the specification, with 40 tasks per additional stage and three fitting tasks.

Each `results_index.jsonl` row has `task_id`, `group`, `fingerprint`, `status` (`measured` or `verified_reuse`), `outcome`, and `evidence`: a nonempty array of `{role, path, sha256}`. Generation needs `raw`, `metrics`, `score`, `metering`; fits need `raw`, `metrics`, `checkpoint`, `metering`; transport needs `raw`, `metrics`. Evidence paths are relative to the campaign directory, nonempty and hash-valid. Reused rows additionally need `reuse_audit` evidence. Model-answer failures may be outcomes; initialization/provider errors must not masquerade as measured scores.

`artifact_index.json` has `artifacts`, a list of `{role, path, sha256}` entries. Required roles: `results_report`, `statistics`, `ablations`, `scaling`, `figures`, `method_fidelity`, `reuse_audit`, `training_ledger`, `cost_ledger`, `source_lock`, `split_manifest`, `independent_rescore`, `protocol_audit`, `shutdown_evidence`, `archive_verification`, `resume`. These artifacts must contain the actual corresponding results, not empty placeholders.

`metrics_coverage.json` has `families`, exactly IDs 1..14. Each record has `id`, `status: complete`, `evidence` and `unresolved_required_cells: []`. Its referenced coverage report enumerates all methods/datasets/modes with measured, estimated or structurally not-applicable fields and reasons. An overall label cannot replace that detailed review.

`resources.json` has `campaign_id`, `registry_complete: true`, and `resources` entries with `id`, `owned: true`, `state: terminated`, `observed_at`, `evidence`. Include every owned GPU and CPU sandbox, even failed/replaced workers. Never infer shutdown from a deleted file; cite provider state. The coordinator can run this final check locally after terminating cloud workers.

The checker validates IDs/counts, receipt status, fingerprints, evidence presence/hashes, coverage bookkeeping and shutdown records. It cannot detect fabricated science inside an otherwise well-formed file, prove baseline fidelity, or independently validate a cloud account. The executor must implement rescoring, protocol/scheduler validation and provider checks, then a separate audit must review their evidence. Package unit tests are mechanical tests, not GPU experiments.

## Official implementation references

- Codex instruction discovery: https://learn.chatgpt.com/docs/agent-configuration/agents-md. Use `AGENTS.md` at the executor repository root; review applicable overrides.
- Modal asynchronous sandbox lifecycle and explicit timeouts: https://modal.com/docs/guide/sandboxes.
- Modal Volume concurrency/visibility constraints: https://modal.com/docs/sdk/py/latest/Volume.
- Sandbox filesystem and Volume persistence: https://modal.com/docs/guide/sandbox-files.

API behavior must be checked against the installed SDK and actual Volume version during implementation. This guide does not claim that cloud execution, physical shaping or end-to-end recovery has been tested in this authoring session.
