# AgentRoute experiment execution contract

These instructions govern the executor repository after this package is installed. The user's goal is a fully executed, auditable finite experiment matrix. A verified ZIP or finished report is not a completed study. Follow higher-priority instructions and explicit user changes.

## 1. Read, recover and freeze

- Read `EXPERIMENT_GUIDE.md`, `campaign_spec.json`, `INPUT_EVIDENCE_STATUS.json` and the executed v11 source/configuration before running paid work. This package is a guide and completion checker, not an implementation of AgentRoute or a cloud launcher.
- Recover the actual v11 archive from the operator's local workspace. Verify its SHA-256 against the supplied reports. Audit raw attempts, split IDs, training artifacts, resource traces and source hashes before crediting any old result. Summary counts alone earn zero verified reuse credit.
- Preserve all earlier evidence. Never overwrite, fabricate, relabel development results as test results, or silently substitute older source code.
- Scope: Qwen/Qwen3-1.7B; MMLU 100, SVAMP 60, MATH 50 and BBH 50 held-out questions; the complete matrix in the guide. Do not silently add larger models or full-dataset sweeps.
- Generate and hash `campaign.lock.json` with exact expected task IDs, source/model/data/prompt/scorer/configuration hashes, seeds, split membership, full metric contracts, baseline fit recipes, stage inventory, load/SLO settings and resource assignment. Freeze it before new held-out execution. Never shrink its expected IDs to make completion pass.
- A plan change requires a new immutable version, a reason and a lineage report. If a changed policy was selected using old test answers, those answers are development evidence; select a fresh evaluation panel before further tuning.

## 2. Complete the finite scope

- Execute every required block, every method, every declared seed/repetition and every distinct component ablation. Negative, null and unfavorable findings remain required results. Do not cancel the remaining matrix because AgentRoute loses.
- This completion-first revision replaces the previous guide's suggested 10-GPU-hour, 25-million-token and 2,700-attempt cutoff. It does not override an actual user monetary limit, provider quota or account restriction. Estimate the whole remaining scope and record cost assumptions before execution; track all consumed resources.
- Use bounded parallelism and bounded retries, not unlimited retries or scale-out. If an actual spending limit, unavailable input or infrastructure problem prevents continuation, persist `BLOCKED` or `INCOMPLETE`, list exact missing task IDs and the required remedy. Do not call this completion.
- For reproducible code bugs, repair and test before retrying. For transient infrastructure failures, use at most three automatic infrastructure attempts per task, with backoff. Preserve each failed attempt and its costs. After exhaustion, leave a blocker; never remove that task.
- Incorrect answers, invalid generated answers and valid protocol timeouts can be measured outcomes. A missing model, failed initialization, absent baseline checkpoint or corrupt export is an infrastructure failure, not an evaluated answer. Do not rerun unfavorable stochastic answers to replace them with better ones.

## 3. Concurrent sandboxes are required

- Coordinator: one CPU process owns the lease/receipt database, global resource registry and merged reports. This may run on the operator's machine or one small CPU sandbox. Persist a journal and checkpoints independently of GPU lifetime.
- GPU-A and GPU-B: two independent, matching GPU sandboxes; one Qwen3-1.7B serving replica each, identical pinned image/hardware/settings. Start independent eligible work concurrently. Default query concurrency is 2 per replica, frozen after development tuning.
- CPU-NET-A and CPU-NET-B: a separate two-host transport pair, CPU only. Keep one timed transport condition active on the pair at a time. It may overlap GPU quality work only when host and link contention are excluded or measured and controlled. Distinct sandbox IDs alone do not prove distinct physical hosts.
- CPU analysis may run concurrently on closed, committed shards. Do not serialize independent quality jobs unnecessarily. Do not overlap timed conditions on a shared GPU, NIC or transport pair.
- Assign small balanced shards, initially ten questions. Rotate methods and question blocks across GPU-A/GPU-B so method and device are not confounded. Record actual host/sandbox/device identity, timestamps, cache state, queue depth and scheduling order.
- Never run one unmetered GPU server per baseline. Never place a training job alongside a timed inference job on the same device. Lease slots for fitting, release them, then schedule evaluation.

## 4. Ownership, persistence and recovery

- Claim work through one coordinator using atomic leases, a lease epoch/fencing token, expiry and heartbeat. Check the prior worker is dead or fenced before reassigning; an expired heartbeat alone is not proof it stopped.
- Each execution writes only to its unique `campaign/worker/task_id/attempt_id/` prefix. Immutable output shards, checksums and receipts are mandatory. No concurrent appends to one CSV/JSONL, shared report, checkpoint or SQLite file on a Modal Volume.
- The coordinator alone owns mutable summary files. Use a transactional store suitable for the deployment, or a local SQLite database behind the single coordinator. Do not treat a shared Volume as a distributed lock or multi-writer database.
- Close and durably commit each shard using the installed Modal SDK/Volume version's documented procedure. Reload on readers before consuming committed data. A process exit or a file existing in one sandbox is not durable export verification.
- A result receipt must include its task ID, configuration fingerprint, attempt ID, source path, artifact hashes, raw-response references, scoring outcome, metering and completion status. Deduplicate by scientific task ID; preserve duplicate executions in the cost/provenance ledger.
- Reuse only matching, raw-verified results. Generation reuse and telemetry reuse are separate: a compatible old answer without required timing/byte traces does not satisfy those measurements. Remeasure such cells separately; never backfill timings from another run.
- On restart, reconstruct queue state from the frozen manifest and verified receipts. Resume missing or invalid work, not the whole campaign. Keep a context-independent `RESUME.md` and `STATUS.json` after every committed shard.

## 5. Baseline fidelity and comparisons

- Run AgentRoute, SafeSieve, CARD, SelfOrg, AgentNet, matched dense/broadcast, matched no communication and single agent. Preserve native baseline logic; document common-task adaptations and pinned upstream revisions. Missing baseline source is a blocker, not permission for an invented replacement.
- Complete the frozen fitting recipe for learned baselines. Validate actual learning/updates and training/test separation. Checkpoint files existing is insufficient. Use the same source-task transfer protocol for all learned methods and disclose it.
- Match AgentRoute/dense/no-communication agent registry, prompts, tools, final selector, stopping condition and maximum rounds. No-communication reasoning agents receive no other agent's answer content. A coordinator may inspect outputs for stopping/final selection; meter that control traffic.
- Keep native whole-system baseline comparisons separate from controlled routing-only comparisons. Do not attribute aggregation, early stopping or training advantages solely to routing.
- Preserve single-agent and no-communication diagnostics even when they outperform AgentRoute. Run the prespecified cost-matched self-consistency control and all attribution ablations.
- Verify mathematical equivalence and parsing against hand-checked tests. Gold answers must never reach routing, stopping, selection or runtime vote grouping. Protect exact evaluation IDs from fitting/tuning.

## 6. Measurement and statistical integrity

- Collect all fourteen metric families in the guide. Every reported number must link to raw evidence. Use explicit measured/estimated/not-applicable labels. Missing required measurements keep the relevant block incomplete.
- Meter every worker, router, final aggregator, fitting, warmup and retry call. Include vote summaries, registry/control messages and acknowledgements in communication accounting. Separate peer-agent traffic from calls to the model server.
- Log causally linked spans and monotonic durations; account for cross-host clock uncertainty. Do not sum overlapping work as end-to-end latency. Include queue wait and failed/dropped requests in load results.
- Warm up consistently, balance cache conditions and randomize block order. Equal seeds do not guarantee identical model output. Never reuse responses after incoming messages or prompts change. Cached initial-response ablations are separately labeled and cannot supply live latency results.
- Run paired statistics, multiplicity correction, seed sensitivity, uncertainty and failure analysis after all fixed-panel data arrive. No significance-based optional stopping. Keep question identity as the sampling/cluster unit, not model calls or repeated seeds.
- An inconclusive noninferiority test is a complete statistical result, not proof of preservation. Do not change the acceptable loss after seeing results. No universal bottleneck-free claim from finite load tests.
- Physical distributed claims require verified distinct hosts and measured traffic shaping. Application sleeps or scaled trace replay must never be relabeled as physical networking or live multi-agent reasoning. Missing required infrastructure is a blocker for that experiment.

## 7. Completion gate and deliverables

- Populate the complete task manifest, append-only results index, artifact index, measurement coverage and resource registry described in the guide. Run `python tools/check_completion.py --run-dir <campaign_directory>`.
- `COMPLETE` requires every expected task ID to have one accepted verified receipt, all required analyses and all fourteen metric-family reports to exist with hash-valid evidence, no unresolved infrastructure failures, no missing scope, and confirmed shutdown of all campaign-owned cloud resources.
- A packaging flag, existing PDF, shell exit code, score above a threshold or `all_members_verified=true` cannot override a failed completion gate. Do not hand-edit the checker output or weaken its tests.
- Produce the baseline tables, paired statistics, ablation tables, transfer results, scaling/load plots, failure analysis, fitting/accounting ledger, method-fidelity notes, immutable raw records, source/config lock, resume instructions and archive manifest. Export unfavorable results too.
- Verify archive integrity and a clean restore. Release idle GPUs promptly after durable export; perform large downloads and report rendering on CPU. Confirm shutdown via provider state, not deletion of a heartbeat file. Preserve persistent evidence volumes.
- Report three separate outcomes: execution completeness, evidence/measurement completeness and scientific claim status. Report actual inference runs separately from mocked/unit tests, trace replay and derived calculations.
- Keep credentials in the operator's authenticated environment/secret store. Do not ask for tokens in chat or include them in logs, notebooks or archives. Do not change account-wide budgets or terminate unrelated resources.

## 8. Handoff and progress

Persist a progress record at least every five minutes during execution: accepted/expected IDs per block, active workers, pending dependencies, blocked IDs, spent cost and forecast remaining cost. Keep running independent valid work while resolving a localized blocker. On an unavoidable interruption, provide exact resume commands and outstanding work with `INCOMPLETE` status. A tool or runtime limit is not evidence that the scientific task is finished.
