# AgentRoute v12 execution guide package

This package redesigns the experiment instructions. It does not contain new experimental results, the v11 source, or the unavailable 244-MB evidence archive.

1. Give `CODEX_START_HERE.md` and this folder to the Codex session that has the actual AgentRoute repository and Modal access.
2. Place/merge `AGENTS.md` at that repository root. Use the uppercase filename so normal Codex instruction discovery can find it.
3. Follow `EXPERIMENT_GUIDE.md`. Fill and freeze the exact manifest from real source and data; do not use placeholders for execution.
4. The included checker is a tested mechanical audit utility. Integration into the actual experiment runner is required. It does not execute cloud jobs or establish scientific validity by itself.

Run the package's offline checks:

```bash
python -m unittest discover -s tests -v
```

Audit a completed or interrupted real campaign after the executor writes the specified files:

```bash
python tools/check_completion.py --run-dir /absolute/path/to/campaign
```

Exit 0 means the mechanical evidence contract passed; exit 2 means incomplete/invalid evidence. The JSON output explains every detected gap. Raw data must still be independently rescored and statistical/methodological conclusions reviewed.

The guide keeps the four small panels and one model, increases attribution controls, and replaces arbitrary early cutoff with resumable completion of a finite manifest. Expected generation tasks: **2,800 + 40 per distinct additional cascade-stage ablation**. There are also **90 CPU transport repetitions** and **three baseline fitting jobs**. Compatible raw results/checkpoints can reduce new compute; counts are not runtime or cost promises.
