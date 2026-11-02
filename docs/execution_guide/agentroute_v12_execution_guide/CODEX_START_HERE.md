# Prompt to pass to Codex

Install this package's `AGENTS.md` at the root of the actual AgentRoute executor repository, merging existing applicable project rules without discarding them. Read `AGENTS.md`, `EXPERIMENT_GUIDE.md` and `campaign_spec.json`. Start a fresh Codex session at that root if necessary so the instruction file is loaded.

Redesign and complete the AgentRoute experiment campaign using the actual v11 source and raw evidence. Use two matching GPU sandboxes concurrently for independent fitting/evaluation work, a single coordinator, and a separate two-host CPU transport pair. Implement the scheduler, measurement hooks, result receipts and recovery behavior required by AGENTS.md in the existing runner. This package provides the execution specification and a mechanical completion checker; it is not a cloud runner.

First locate `AgentRoute_Compact_v11_Evidence.zip` in my existing workspace and verify SHA-256 `dae1e2d1f445b299bf717fdb40cc5a9321c99d57931557a9fb7dc48a683505bb`. The supplied completion report says 710/800 primary attempts and 57/90 transport repetitions were completed, but audit individual records before deciding which can be reused. Do not infer completion of any other block. Do not substitute an old version of the code.

Keep Qwen3-1.7B and the specified small dataset panels. Complete all main baselines, all fourteen metric families, statistics, transfer panels, component/aggregation/self-consistency ablations and the full communication scaling matrix. Discover the actual cascade stages and include every distinct leave-one-stage-out arm before freezing the manifest. Execute the finite frozen scope even if AgentRoute loses. Remove the earlier guide's artificial stop-after-10-GPU-hours rule; keep actual user/account spending limits intact and display the whole-scope estimate and ongoing cost ledger.

Resume verified compatible work, execute all remaining required cells, and export all results and raw evidence. Never report success solely because packaging finished. Run the completion checker, audit the science and verify cloud resource shutdown before saying COMPLETE. If a real input, account or infrastructure blocker prevents completion, identify it precisely, retain an INCOMPLETE state and continue any independent authorized work. Never manufacture a missing result or claim an unsupported guarantee.

Use the existing authenticated Modal setup. Do not ask me to paste credentials into chat. Do not launch a broader model/dataset sweep than this guide defines.
