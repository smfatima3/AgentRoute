"""
concurrent_load_test.py
========================

Concurrent-load scalability test for AgentRoute (EMNLP 2026 Industry Track,
appendix). Sweeps the target queries-per-second (QPS) and measures how each
router's tail latency degrades under load.

What this script does
---------------------
Fires queries concurrently against the chosen router at a target QPS, using
asyncio to schedule arrivals with Poisson inter-arrival times (the realistic
production traffic model). Measures end-to-end latency distribution under each
QPS and reports the saturation point at which the router can no longer keep up
with the offered load.

Two backends supported
----------------------
1. Local Qwen 2.5-3B via transformers (reproducible without API credits;
   true concurrency is bounded by single-GPU serial generation, but the
   queueing pressure on each router's internal logic is preserved).
2. Claude API via the anthropic SDK (true concurrent requests; will hit the
   real rate limits at high QPS, which is the point).

Workload
--------
A small fixed pool of MedQA questions, cycled to feed the sustained QPS for
`--duration_s` seconds at each level. Different routers see the same query
mix at the same arrival timestamps for fair comparison.

Metrics per (router, target_qps):
    achieved_qps                  # actually-completed queries / duration
    n_completed                    # how many requests finished
    n_timeouts                     # requests that hit --per_query_timeout_s
    n_429 / n_529                  # API rate-limit / overload counts (Claude only)
    mean / p50 / p95 / p99 e2e_latency_s
    saturation_ratio               # achieved_qps / target_qps
                                     (drops below ~0.95 once the router saturates)
    mean_cost_usd, total_cost_usd  # Claude only
    gpu_peak_mb                    # local Qwen only

Statistical analysis is light here (the headline result is the saturation
curve, not a paired t-test): we report Spearman rank correlation between
target_qps and p95 latency per router, and a Friedman chi-square across all
routers at each target_qps. Full per-(router, qps, seed) raw rows are saved
to CSV for offline plotting.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

# We reuse the router classes from medqa_evaluation.py. They are synchronous,
# so for the local-Qwen backend we wrap them with asyncio.to_thread(); for the
# Claude backend we wrap the ClaudeBackbone in an async layer below.
from medqa_evaluation import (  # noqa: E402
    AgentRouteWrapper,
    DyTopoWrapper,
    EvoMASWrapper,
    OIMASWrapper,
    RCRRouterWrapper,
    RopMuraWrapper,
    grade_mcq,
    load_medqa,
    make_agents,
)

ROUTER_CLASSES = [
    AgentRouteWrapper, OIMASWrapper, RCRRouterWrapper,
    EvoMASWrapper, DyTopoWrapper, RopMuraWrapper,
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("concurrent_load")


# ============================================================================
# 1. BACKENDS
# ============================================================================

class LocalQwenBackend:
    """Local Qwen 2.5-3B with asyncio.to_thread() wrapping the sync generator.

    Note: a single GPU serializes generations, so concurrency here exposes
    *queueing* pressure on the router's internal logic (planner/manager loops),
    not true parallel inference. That is still informative: it shows whether
    the router's own control flow is the bottleneck at high QPS.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        import threading
        from medqa_evaluation import Qwen25Backbone
        self.inner = Qwen25Backbone(model_name=model_name)
        self.model_name = model_name
        # Counters surfaced for reporting (kept zero — local has no API errors)
        self.spent_usd = 0.0
        self.n_429 = 0
        self.n_529 = 0
        self.n_retries = 0
        self.price = {"in": 0.0, "out": 0.0}
        # Serialize all GPU calls. IMPORTANT: this must be a threading.Lock,
        # not an asyncio.Lock — AsyncRouterRunner moves router.solve() into
        # worker threads via asyncio.to_thread, so generate_sync() is invoked
        # from multiple OS threads concurrently. An asyncio.Lock would not
        # guard that path, and concurrent model.generate() calls on a single
        # GPU model can crash CUDA or interleave KV caches.
        self._gpu_lock = threading.Lock()

    async def generate(self, prompt: str, max_new_tokens: int = 256,
                       temperature: float = 0.0) -> Tuple[str, int, int, float]:
        return await asyncio.to_thread(
            self.generate_sync, prompt, max_new_tokens, temperature
        )

    def generate_sync(self, prompt: str, max_new_tokens: int = 256,
                      temperature: float = 0.0) -> Tuple[str, int, int, float]:
        """Sync entry point used by router classes that aren't asyncio-aware.
        Thread-safe: all GPU work is serialized through _gpu_lock."""
        with self._gpu_lock:
            return self.inner.generate(prompt, max_new_tokens, temperature)


class ClaudeBackend:
    """Claude API with truly concurrent requests via the async SDK."""
    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 budget_usd: float = 50.0,
                 max_retries: int = 4):
        # Import lazily so the script runs without anthropic installed
        # when only the local backend is used.
        from production_casestudy import (  # noqa: E402
            ClaudeBackbone, PRICE_PER_1M, BudgetExceeded,
        )
        self._budget_exceeded = BudgetExceeded
        self.inner = ClaudeBackbone(model=model, budget_usd=budget_usd,
                                    max_retries=max_retries)
        self.model_name = model
        self.price = PRICE_PER_1M[model]

    @property
    def spent_usd(self): return self.inner.spent_usd
    @property
    def n_429(self):     return self.inner.n_429
    @property
    def n_529(self):     return self.inner.n_529
    @property
    def n_retries(self): return self.inner.n_retries

    async def generate(self, prompt: str, max_new_tokens: int = 256,
                       temperature: float = 0.0) -> Tuple[str, int, int, float]:
        return await asyncio.to_thread(
            self.inner.generate, prompt, max_new_tokens, temperature
        )

    def generate_sync(self, prompt: str, max_new_tokens: int = 256,
                      temperature: float = 0.0) -> Tuple[str, int, int, float]:
        return self.inner.generate(prompt, max_new_tokens, temperature)


# ============================================================================
# 2. ASYNC ROUTER WRAPPER
# ============================================================================

class AsyncRouterRunner:
    """Wraps a synchronous router class so it can be awaited.

    The router classes from medqa_evaluation.py call llm.generate() (sync).
    We pass them an LLM stand-in whose .generate is the sync method of the
    backend; the *whole* solve() call is moved off the event loop via
    asyncio.to_thread, which preserves the router's internal control flow.
    """
    def __init__(self, router_cls, num_agents: int, backend):
        self.router_cls = router_cls
        self.backend = backend

        class _SyncLLMAdapter:
            """Exposes only the sync .generate so routers see a normal LLM."""
            def __init__(self, bk):
                self._bk = bk
                self.model_name = bk.model_name
            def generate(self, prompt, max_new_tokens=256, temperature=0.0):
                return self._bk.generate_sync(prompt, max_new_tokens, temperature)

        self._sync_llm = _SyncLLMAdapter(backend)
        self.agents = make_agents(num_agents)
        self.router = router_cls(agents=self.agents, llm=self._sync_llm)
        self.router.reset()

    async def solve(self, question: str, options):
        return await asyncio.to_thread(self.router.solve, question, options)


# ============================================================================
# 3. LOAD GENERATOR
# ============================================================================

@dataclass
class CompletedRequest:
    arrival_t: float       # seconds since run start when the request was scheduled
    start_t: float         # seconds since run start when the router began processing
    end_t: float           # seconds since run start when the request completed
    e2e_latency_s: float
    correct: bool
    n_llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timed_out: bool
    error: Optional[str]


async def _drive_one_request(arrival_t: float,
                             prob: Dict[str, Any],
                             runner: AsyncRouterRunner,
                             backend,
                             run_start_t: float,
                             per_query_timeout_s: float
                             ) -> CompletedRequest:
    # Wait until the scheduled arrival time
    delay = arrival_t - (time.time() - run_start_t)
    if delay > 0:
        await asyncio.sleep(delay)
    start_t = time.time() - run_start_t

    err: Optional[str] = None
    timed_out = False
    correct = False
    n_calls = 0
    p_tok = 0
    c_tok = 0
    cost = 0.0
    e2e = 0.0
    try:
        st = await asyncio.wait_for(
            runner.solve(prob["question"], prob["options"]),
            timeout=per_query_timeout_s,
        )
        correct = grade_mcq(st.final_text, prob["gold_letter"])
        n_calls = st.n_llm_calls
        p_tok = st.prompt_tokens
        c_tok = st.completion_tokens
        cost = ((p_tok / 1_000_000.0) * backend.price["in"]
              + (c_tok / 1_000_000.0) * backend.price["out"])
        e2e = st.e2e_latency_s
    except asyncio.TimeoutError:
        timed_out = True
        err = "timeout"
    except Exception as e:  # pragma: no cover (defensive)
        err = f"{type(e).__name__}: {e}"

    end_t = time.time() - run_start_t
    return CompletedRequest(
        arrival_t=arrival_t,
        start_t=start_t,
        end_t=end_t,
        e2e_latency_s=e2e if e2e > 0 else (end_t - start_t),
        correct=correct,
        n_llm_calls=n_calls,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        cost_usd=cost,
        timed_out=timed_out,
        error=err,
    )


async def run_load_one(router_cls, problems, backend, target_qps: float,
                       duration_s: float, num_agents: int,
                       per_query_timeout_s: float, seed: int
                       ) -> List[CompletedRequest]:
    """Fire requests at `target_qps` for `duration_s` seconds, return all completed rows."""
    random.seed(seed)
    np.random.seed(seed)
    runner = AsyncRouterRunner(router_cls, num_agents, backend)

    # Pre-compute Poisson arrival times for the whole window
    arrivals: List[float] = []
    t = 0.0
    while t < duration_s:
        # Exponential inter-arrival -> Poisson process
        gap = -math.log(1.0 - random.random()) / max(1e-6, target_qps)
        t += gap
        if t < duration_s:
            arrivals.append(t)

    # Cycle through the question pool to feed the load
    q_pool = problems
    run_start = time.time()
    coros = []
    for i, arr in enumerate(arrivals):
        prob = q_pool[i % len(q_pool)]
        coros.append(_drive_one_request(arr, prob, runner, backend,
                                        run_start, per_query_timeout_s))
    # Fire them all; gather waits for everything (including arrivals scheduled
    # after duration_s if they were already created before cutoff -- we cap
    # in the loop above so no late arrivals exist).
    results = await asyncio.gather(*coros, return_exceptions=False)
    return list(results)


# ============================================================================
# 4. PER-CELL AGGREGATION
# ============================================================================

@dataclass
class LoadCellRow:
    router: str
    target_qps: float
    seed: int
    duration_s: float
    n_offered: int        # requests scheduled in this cell
    n_completed: int
    n_timeouts: int
    n_errors: int
    achieved_qps: float
    saturation_ratio: float   # achieved / target
    mean_e2e_latency_s: float
    p50_e2e_latency_s: float
    p95_e2e_latency_s: float
    p99_e2e_latency_s: float
    mean_llm_calls: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_cost_usd: float
    total_cost_usd: float
    accuracy_completed: float
    n_429: int
    n_529: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_cell(router_name: str, target_qps: float, seed: int,
                   duration_s: float, completed: List[CompletedRequest],
                   backend_429_before: int, backend_529_before: int,
                   backend_429_after: int, backend_529_after: int
                   ) -> LoadCellRow:
    n_off = len(completed)
    succeeded = [r for r in completed if (not r.timed_out and r.error is None)]
    timed_out = sum(1 for r in completed if r.timed_out)
    errors = sum(1 for r in completed if (r.error is not None and not r.timed_out))
    n_done = len(succeeded)

    if succeeded:
        latencies = np.array([r.e2e_latency_s for r in succeeded])
        ncalls    = np.array([r.n_llm_calls for r in succeeded])
        ptoks     = np.array([r.prompt_tokens for r in succeeded])
        ctoks     = np.array([r.completion_tokens for r in succeeded])
        costs     = np.array([r.cost_usd for r in succeeded])
        accuracy  = float(np.mean([r.correct for r in succeeded]))
    else:
        latencies = np.array([0.0])
        ncalls = ptoks = ctoks = costs = np.array([0.0])
        accuracy = 0.0

    return LoadCellRow(
        router=router_name,
        target_qps=target_qps,
        seed=seed,
        duration_s=duration_s,
        n_offered=n_off,
        n_completed=n_done,
        n_timeouts=timed_out,
        n_errors=errors,
        achieved_qps=n_done / max(1e-6, duration_s),
        saturation_ratio=(n_done / max(1e-6, duration_s)) / max(1e-6, target_qps),
        mean_e2e_latency_s=float(np.mean(latencies)),
        p50_e2e_latency_s=float(np.percentile(latencies, 50)),
        p95_e2e_latency_s=float(np.percentile(latencies, 95)),
        p99_e2e_latency_s=float(np.percentile(latencies, 99)),
        mean_llm_calls=float(np.mean(ncalls)),
        mean_prompt_tokens=float(np.mean(ptoks)),
        mean_completion_tokens=float(np.mean(ctoks)),
        mean_cost_usd=float(np.mean(costs)),
        total_cost_usd=float(np.sum(costs)),
        accuracy_completed=accuracy,
        n_429=backend_429_after - backend_429_before,
        n_529=backend_529_after - backend_529_before,
    )


# ============================================================================
# 5. STATISTICAL ANALYSIS
# ============================================================================

def per_qps_friedman(df: pd.DataFrame, metric: str = "p95_e2e_latency_s"
                     ) -> pd.DataFrame:
    rows = []
    for qps, df_q in df.groupby("target_qps"):
        routers = df_q["router"].unique().tolist()
        cols = [df_q[df_q["router"] == r].sort_values("seed")[metric].to_numpy()
                for r in routers]
        if (len(cols) >= 3
                and all(len(c) == len(cols[0]) for c in cols)
                and len(cols[0]) >= 2):
            try:
                chi2, p = stats.friedmanchisquare(*cols)
            except ValueError:
                chi2, p = float("nan"), float("nan")
        else:
            chi2, p = float("nan"), float("nan")
        rows.append({"target_qps": float(qps), "metric": metric,
                     "n_routers": len(routers),
                     "chi2": float(chi2), "p_value": float(p)})
    return pd.DataFrame(rows)


def per_router_qps_correlation(df: pd.DataFrame,
                               metric: str = "p95_e2e_latency_s"
                               ) -> pd.DataFrame:
    """Spearman rank correlation between target QPS and metric per router."""
    rows = []
    for router, df_r in df.groupby("router"):
        if df_r["target_qps"].nunique() < 3:
            continue
        agg = df_r.groupby("target_qps")[metric].mean().reset_index()
        x = agg["target_qps"].to_numpy()
        y = agg[metric].to_numpy()
        try:
            res = stats.spearmanr(x, y)
            rho, p = float(res.statistic), float(res.pvalue)
        except ValueError:
            rho, p = float("nan"), float("nan")
        rows.append({"router": router, "metric": metric, "n_points": len(agg),
                     "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows).sort_values("spearman_rho")


# ============================================================================
# 6. MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["local", "claude"], default="local",
                   help="local = Qwen 2.5-3B on this GPU; "
                        "claude = real concurrent calls to Claude.")
    p.add_argument("--model", type=str,
                   default="Qwen/Qwen2.5-3B-Instruct",
                   help="Local: HF model id. Claude: API model string.")
    p.add_argument("--target_qps", type=str, default="0.1,0.5,1,2,5",
                   help="Comma-separated target QPS values to sweep.")
    p.add_argument("--duration_s", type=float, default=120.0,
                   help="How long to sustain each QPS level (seconds).")
    p.add_argument("--per_query_timeout_s", type=float, default=180.0,
                   help="Per-request timeout. Any request exceeding this is "
                        "counted as a timeout, not an error.")
    p.add_argument("--num_problems_pool", type=int, default=50,
                   help="Size of the MedQA question pool to cycle from.")
    p.add_argument("--num_agents", type=int, default=8)
    p.add_argument("--runs", type=int, default=2,
                   help="Independent seeds per (router, QPS) cell.")
    p.add_argument("--seeds", type=str, default="13,14")
    p.add_argument("--routers", type=str, default="",
                   help="Comma-separated subset of routers to run.")
    p.add_argument("--budget_usd", type=float, default=20.0,
                   help="Hard $ cap (Claude backend only).")
    p.add_argument("--out_dir", type=str, default=".")
    return p.parse_args()


async def amain():
    args = parse_args()
    target_qps_list = [float(x) for x in args.target_qps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")][:args.runs]
    if len(seeds) < args.runs:
        seeds += [seeds[-1] + i for i in range(1, args.runs - len(seeds) + 1)]

    chosen = ROUTER_CLASSES
    if args.routers:
        names = {s.strip() for s in args.routers.split(",")}
        chosen = [c for c in ROUTER_CLASSES if c.name in names]

    log.info(f"backend={args.backend}  model={args.model}")
    log.info(f"routers={[c.name for c in chosen]}")
    log.info(f"QPS sweep={target_qps_list}  duration_s={args.duration_s}")
    log.info(f"seeds={seeds}  pool_size={args.num_problems_pool}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build backend
    if args.backend == "local":
        backend = LocalQwenBackend(model_name=args.model)
    else:
        backend = ClaudeBackend(model=args.model, budget_usd=args.budget_usd)

    # Same question pool for every (router, qps, seed) cell at a given seed
    problems_by_seed = {s: load_medqa(args.num_problems_pool, seed=s)
                        for s in seeds}

    all_cells: List[LoadCellRow] = []
    all_raw: List[Dict[str, Any]] = []

    for seed in seeds:
        problems = problems_by_seed[seed]
        for qps in target_qps_list:
            for router_cls in chosen:
                log.info(f"=== router={router_cls.name}  qps={qps}  seed={seed} ===")
                n429_before = backend.n_429
                n529_before = backend.n_529
                completed = await run_load_one(
                    router_cls, problems, backend,
                    target_qps=qps,
                    duration_s=args.duration_s,
                    num_agents=args.num_agents,
                    per_query_timeout_s=args.per_query_timeout_s,
                    seed=seed,
                )
                cell = aggregate_cell(
                    router_name=router_cls.name,
                    target_qps=qps,
                    seed=seed,
                    duration_s=args.duration_s,
                    completed=completed,
                    backend_429_before=n429_before,
                    backend_529_before=n529_before,
                    backend_429_after=backend.n_429,
                    backend_529_after=backend.n_529,
                )
                log.info(
                    f"   offered={cell.n_offered} done={cell.n_completed} "
                    f"timeouts={cell.n_timeouts} sat={cell.saturation_ratio:.2f} "
                    f"p95={cell.p95_e2e_latency_s:.2f}s p99={cell.p99_e2e_latency_s:.2f}s"
                )
                all_cells.append(cell)
                # Save the raw per-request rows too
                for r in completed:
                    rd = asdict(r)
                    rd.update({"router": router_cls.name,
                               "target_qps": qps,
                               "seed": seed})
                    all_raw.append(rd)
                gc.collect()

    df = pd.DataFrame([c.to_dict() for c in all_cells])
    df.to_csv(out_dir / "concurrent_load_results.csv", index=False)
    with open(out_dir / "concurrent_load_results.json", "w") as f:
        json.dump([c.to_dict() for c in all_cells], f, indent=2)
    pd.DataFrame(all_raw).to_csv(out_dir / "concurrent_load_raw.csv", index=False)
    log.info(f"Raw and aggregated outputs written to {out_dir}/")

    # Aggregate: mean +/- std per (router, qps) across seeds
    agg = (df.groupby(["router", "target_qps"])
             .agg(achieved_qps_mean=("achieved_qps", "mean"),
                  saturation_ratio_mean=("saturation_ratio", "mean"),
                  p50_mean=("p50_e2e_latency_s", "mean"),
                  p95_mean=("p95_e2e_latency_s", "mean"),
                  p99_mean=("p99_e2e_latency_s", "mean"),
                  timeouts_sum=("n_timeouts", "sum"),
                  rate_limit_429_sum=("n_429", "sum"),
                  rate_limit_529_sum=("n_529", "sum"),
                  accuracy_mean=("accuracy_completed", "mean"))
             .reset_index())
    agg.to_csv(out_dir / "concurrent_load_aggregate.csv", index=False)
    log.info("\n" + "=" * 100)
    log.info(" Aggregate per (router, qps)")
    log.info("=" * 100)
    print(agg.round(4).to_string(index=False))

    # Friedman across routers at each QPS (on p95 latency)
    fr = per_qps_friedman(df, metric="p95_e2e_latency_s")
    fr.to_csv(out_dir / "concurrent_load_friedman.csv", index=False)
    log.info("\n" + "=" * 100)
    log.info(" Friedman chi-square at each QPS (p95 latency)")
    log.info("=" * 100)
    print(fr.round(6).to_string(index=False))

    # Per-router Spearman rank correlation (target_qps vs p95)
    sp = per_router_qps_correlation(df, metric="p95_e2e_latency_s")
    sp.to_csv(out_dir / "concurrent_load_spearman.csv", index=False)
    log.info("\n" + "=" * 100)
    log.info(" Spearman rank correlation (target_qps vs p95 latency) per router")
    log.info("=" * 100)
    print(sp.round(4).to_string(index=False))

    if args.backend == "claude":
        log.info("\n" + "=" * 100)
        log.info(" Claude API spend summary")
        log.info("=" * 100)
        log.info(f"  total spend     : ${backend.spent_usd:.4f}")
        log.info(f"  total retries   : {backend.n_retries}")
        log.info(f"  429 hits        : {backend.n_429}")
        log.info(f"  529 hits        : {backend.n_529}")


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()


# =====================================================================
# HOW TO RUN
# =====================================================================
"""
# Install deps (one-time)
pip install torch transformers datasets accelerate scipy pandas numpy tqdm
# (Add anthropic>=0.49.0 only if you want the Claude backend.)
pip install "anthropic>=0.49.0"

# ---------------- LOCAL Qwen 2.5-3B BACKEND ----------------
# Smoke test (1 seed, 3 QPS levels, 60s each, AgentRoute + OI-MAS only)
python concurrent_load_test.py \\
    --backend local --runs 1 --duration_s 60 \\
    --target_qps 0.5,1,2 --routers AgentRoute,OI-MAS \\
    --out_dir smoke_load_local

# Full local sweep: ~1 hour on a single 16 GB GPU
python concurrent_load_test.py \\
    --backend local --runs 2 --seeds 13,14 \\
    --target_qps 0.1,0.5,1,2,5 --duration_s 120 \\
    --num_problems_pool 50 --num_agents 8 \\
    --out_dir load_local_out

# ---------------- CLAUDE API BACKEND ----------------
# WARNING: Claude tier-1 rate limits are ~50 RPM. Setting target_qps above
# ~0.8 will deliberately push past the cap; that is the point (to measure
# how each router handles 429s) but expect API spend to climb fast.
export ANTHROPIC_API_KEY=sk-ant-...

# Smoke test ($1 cap)
python concurrent_load_test.py \\
    --backend claude --model claude-haiku-4-5-20251001 \\
    --runs 1 --duration_s 30 --target_qps 0.5,1,2 \\
    --budget_usd 1.00 --routers AgentRoute,OI-MAS \\
    --out_dir smoke_load_claude

# Full Claude sweep ($20 cap, expect ~$10-15 actual spend)
python concurrent_load_test.py \\
    --backend claude --model claude-haiku-4-5-20251001 \\
    --runs 2 --seeds 13,14 \\
    --target_qps 0.2,0.5,1,2,4 --duration_s 90 \\
    --budget_usd 20.00 --out_dir load_claude_out

# Outputs (in --out_dir):
#   concurrent_load_results.csv     # one row per (router, qps, seed) cell
#   concurrent_load_results.json
#   concurrent_load_raw.csv         # one row per individual request
#   concurrent_load_aggregate.csv   # mean +/- std per (router, qps)
#   concurrent_load_friedman.csv    # Friedman chi-square at each QPS
#   concurrent_load_spearman.csv    # rank correlation (qps vs p95) per router
"""
