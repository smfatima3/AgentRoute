"""
scalability_sweep.py
=====================

Agent-pool scalability study for the AgentRoute paper (EMNLP 2026 Industry Track).

What this script does
---------------------
Holds the query workload fixed at `--num_problems` MedQA questions and sweeps
the number of registered specialist agents N over `--agent_pool_sizes`. For
each (router, N) pair it runs `--runs` independent seeds and records every
metric the paper reports.

Goal: characterise how each router's behaviour scales with N.
  - AgentRoute's routing-decision cost is O(|domains| x |keywords|), which is
    independent of N. End-to-end latency should be roughly flat in N.
  - Learned-controller baselines (MasRouter, OI-MAS) and topology-rebuilding
    methods (DyTopo, RopMura, RCR-Router, EvoMAS) embed or score over the
    candidate set, so we expect their per-query overhead to grow with N.

Six routers compared:
    AgentRoute, OI-MAS, RCR-Router, EvoMAS, DyTopo, RopMura

Reuses the architecturally-faithful router implementations from
medqa_evaluation.py (already validated in prior runs).

Metrics recorded per (router, N, seed):
    accuracy, mean / p50 / p95 end-to-end latency,
    routing-decision latency (mean / p95), throughput (q/s),
    mean prompt + completion tokens, mean LLM calls,
    mean per-query cost (USD), total cost, GPU peak memory (MB).

Statistical analysis:
    - per-N paired t-test of AgentRoute vs each other router on
      (accuracy, mean_e2e_latency_s, mean_cost_usd, mean_routing_decision_ms)
    - Cohen's d (paired), 95% CI on paired difference
    - Bonferroni-corrected significance flag
    - per-N Friedman chi-square across all six routers
    - per-router OLS slope of mean_e2e_latency vs N (the "scaling exponent")

Outputs:
    scalability_results.csv          # one row per (router, N, seed)
    scalability_aggregate.csv        # mean +/- std per (router, N)
    scalability_message_volume.csv   # §5 figure source: messages/q vs N
                                     # rows = N values, columns = each router
                                     # (mean) and Broadcast (= N, reference);
                                     # <router>_sem columns hold error-bar SEM
    scalability_pairwise.csv         # paired-test table for every N
    scalability_friedman.csv         # Friedman chi-square per N and metric
    scalability_scaling_slopes.csv   # OLS slope of latency vs N per router
    scalability_results.json         # raw blob mirroring the CSV
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

# Reuse the validated router implementations from the MedQA evaluation script.
# medqa_evaluation.py must be in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from medqa_evaluation import (  # noqa: E402
    Qwen25Backbone,
    ROUTER_CLASSES,
    cost_usd,
    grade_mcq,
    gpu_mem_mb,
    load_medqa,
    make_agents,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("scalability_sweep")

DEFAULT_POOL_SIZES = [5, 10, 25, 50, 100, 200]
DEFAULT_SEEDS = [13, 14, 15]


# ============================================================================
# 1. PER-RUN MEASUREMENT
# ============================================================================

@dataclass
class ScaleRow:
    """One row of the result table: a single (router, N, seed) measurement."""
    router: str
    pool_size_N: int
    seed: int
    n_problems: int
    n_correct: int
    accuracy: float
    mean_e2e_latency_s: float
    p50_e2e_latency_s: float
    p95_e2e_latency_s: float
    mean_routing_decision_ms: float
    p95_routing_decision_ms: float
    throughput_qps: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_llm_calls: float
    mean_cost_usd: float
    total_cost_usd: float
    gpu_peak_mb: float
    # --- inter-agent communication metrics (Section 5 of the paper) ---
    # mean_messages_per_query: number of agent-to-agent LLM messages exchanged
    #   per query (one message = one agent receiving a prompt and replying).
    #   For the routers we evaluate this equals mean_llm_calls, but the field
    #   is named separately because the *framing* matters for the paper: the
    #   bottleneck is inter-agent communication, of which each LLM call is the
    #   physical instantiation.
    # total_messages: cumulative messages across the whole workload at this N.
    # broadcast_reference_msgs_per_query: the analytical worst case (one query
    #   sent to all N registered agents). Constant per N, included on every
    #   row so the plot script can draw the dashed reference line without
    #   joining a separate table.
    # broadcast_reference_total_msgs: same, total across the workload.
    # message_reduction_vs_broadcast: 1 - (mean_messages / N). Bounded [0,1].
    #   AgentRoute should be ~1 - 1/N (essentially 1.0 for large N); multi-call
    #   routers will drop as their per-query call count grows.
    mean_messages_per_query: float
    total_messages: int
    broadcast_reference_msgs_per_query: int
    broadcast_reference_total_msgs: int
    message_reduction_vs_broadcast: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def measure_one(router_cls, problems, llm, pool_size_N: int,
                seed: int) -> ScaleRow:
    """Run one (router, N, seed) measurement and return a ScaleRow."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    agents = make_agents(pool_size_N)
    router = router_cls(agents=agents, llm=llm)
    router.reset()

    e2e: List[float] = []
    routing_ms: List[float] = []
    p_tokens: List[int] = []
    c_tokens: List[int] = []
    n_calls: List[int] = []
    cost: List[float] = []
    n_correct = 0
    wall_start = time.time()

    for prob in problems:
        # Time the routing decision separately when possible. For all our
        # router classes the routing decision is the constant-time prefix of
        # router.solve() before the LLM is called; we measure it via the
        # router.solve() total time minus the LLM-only seconds (CallStats
        # already tracks llm_latency_s for every router).
        st = router.solve(prob["question"], prob["options"])
        ok = grade_mcq(st.final_text, prob["gold_letter"])
        n_correct += int(ok)
        e2e.append(st.e2e_latency_s)
        # Routing-decision time = end-to-end - LLM time. For AgentRoute on a
        # cache miss this is ~0.5 ms; for DyTopo / RopMura it includes the
        # planner/manager logic between LLM calls.
        routing_ms.append(max(0.0, (st.e2e_latency_s - st.llm_latency_s) * 1000.0))
        p_tokens.append(st.prompt_tokens)
        c_tokens.append(st.completion_tokens)
        n_calls.append(st.n_llm_calls)
        cost.append(cost_usd(st.prompt_tokens, st.completion_tokens))
        # EvoMAS uses online execution feedback to update its fitness pool.
        if hasattr(router, "update_fitness"):
            router.update_fitness(ok)

    total_wall = time.time() - wall_start
    _, peak_mb = gpu_mem_mb()

    # --- inter-agent message volume (Section 5 of the paper) ---
    mean_msgs = float(np.mean(n_calls))
    total_msgs = int(np.sum(n_calls))
    broadcast_per_q = int(pool_size_N)
    broadcast_total = broadcast_per_q * len(problems)
    msg_reduction = 1.0 - (mean_msgs / max(1, broadcast_per_q))

    return ScaleRow(
        router=router_cls.name,
        pool_size_N=pool_size_N,
        seed=seed,
        n_problems=len(problems),
        n_correct=n_correct,
        accuracy=n_correct / max(1, len(problems)),
        mean_e2e_latency_s=float(np.mean(e2e)),
        p50_e2e_latency_s=float(np.percentile(e2e, 50)),
        p95_e2e_latency_s=float(np.percentile(e2e, 95)),
        mean_routing_decision_ms=float(np.mean(routing_ms)),
        p95_routing_decision_ms=float(np.percentile(routing_ms, 95)),
        throughput_qps=len(problems) / max(1e-6, total_wall),
        mean_prompt_tokens=float(np.mean(p_tokens)),
        mean_completion_tokens=float(np.mean(c_tokens)),
        mean_llm_calls=float(np.mean(n_calls)),
        mean_cost_usd=float(np.mean(cost)),
        total_cost_usd=float(np.sum(cost)),
        gpu_peak_mb=peak_mb,
        mean_messages_per_query=mean_msgs,
        total_messages=total_msgs,
        broadcast_reference_msgs_per_query=broadcast_per_q,
        broadcast_reference_total_msgs=broadcast_total,
        message_reduction_vs_broadcast=msg_reduction,
    )


# ============================================================================
# 2. STATISTICAL ANALYSIS
# ============================================================================

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    if sd == 0.0:
        return float("inf") if diff.mean() != 0 else 0.0
    return float(diff.mean() / sd)


def ci95_paired(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = a - b
    n = len(diff)
    if n < 2:
        return (float("nan"), float("nan"))
    m = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(n))
    h = se * stats.t.ppf(0.975, n - 1)
    return (m - h, m + h)


def per_N_pairwise(df: pd.DataFrame, reference: str,
                   metrics: List[str]) -> pd.DataFrame:
    """Paired t-test (reference vs each other router) at every value of N."""
    rows = []
    routers = df["router"].unique().tolist()
    if reference not in routers:
        return pd.DataFrame()
    n_tests = max(1, (len(routers) - 1) * len(metrics))
    bonferroni_alpha = 0.05 / n_tests
    for N, df_N in df.groupby("pool_size_N"):
        for r in routers:
            if r == reference:
                continue
            ref = df_N[df_N["router"] == reference].sort_values("seed")
            other = df_N[df_N["router"] == r].sort_values("seed")
            if len(ref) != len(other) or len(ref) < 2:
                continue
            for metric in metrics:
                a = ref[metric].to_numpy()
                b = other[metric].to_numpy()
                if np.all(a == b):
                    t_stat, p = 0.0, 1.0
                else:
                    t_stat, p = stats.ttest_rel(a, b)
                d = cohens_d_paired(a, b)
                lo, hi = ci95_paired(a, b)
                rows.append({
                    "pool_size_N": int(N),
                    "comparison": f"{reference} vs {r}",
                    "metric": metric,
                    "n_seeds": len(a),
                    f"{reference}_mean": float(a.mean()),
                    f"{r}_mean": float(b.mean()),
                    "delta_mean": float(a.mean() - b.mean()),
                    "t_stat": float(t_stat),
                    "p_value": float(p),
                    "cohens_d": d,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "bonferroni_alpha": bonferroni_alpha,
                    "significant_bonferroni": bool(p < bonferroni_alpha),
                })
    return pd.DataFrame(rows)


def per_N_friedman(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for N, df_N in df.groupby("pool_size_N"):
        routers = df_N["router"].unique().tolist()
        for metric in metrics:
            cols = []
            for r in routers:
                vals = df_N[df_N["router"] == r].sort_values("seed")[metric].to_numpy()
                cols.append(vals)
            if (len(cols) >= 3
                    and all(len(c) == len(cols[0]) for c in cols)
                    and len(cols[0]) >= 2):
                try:
                    chi2, p = stats.friedmanchisquare(*cols)
                except ValueError:
                    chi2, p = float("nan"), float("nan")
            else:
                chi2, p = float("nan"), float("nan")
            rows.append({
                "pool_size_N": int(N),
                "metric": metric,
                "n_routers": len(routers),
                "chi2": float(chi2),
                "p_value": float(p),
            })
    return pd.DataFrame(rows)


def per_router_scaling_slope(df: pd.DataFrame,
                             metric: str = "mean_e2e_latency_s"
                             ) -> pd.DataFrame:
    """Fit a line metric vs N for each router. Slope is the scaling exponent
    for that router-metric pair, intercept the constant-load cost."""
    rows = []
    for router, df_r in df.groupby("router"):
        # Average across seeds at each N first, then fit.
        agg = df_r.groupby("pool_size_N")[metric].mean().reset_index()
        if len(agg) < 2:
            continue
        x = agg["pool_size_N"].to_numpy(dtype=float)
        y = agg[metric].to_numpy(dtype=float)
        # Linear regression
        try:
            res = stats.linregress(x, y)
        except ValueError:
            continue
        rows.append({
            "router": router,
            "metric": metric,
            "n_points": len(agg),
            "slope": float(res.slope),
            "intercept": float(res.intercept),
            "r_squared": float(res.rvalue ** 2),
            "p_value": float(res.pvalue),
            "stderr_slope": float(res.stderr),
        })
    return pd.DataFrame(rows).sort_values("slope")


# ============================================================================
# 3. AGGREGATION
# ============================================================================

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std for each (router, N) cell across seeds."""
    return (
        df.groupby(["router", "pool_size_N"])
          .agg(accuracy_mean=("accuracy", "mean"),
               accuracy_std=("accuracy", "std"),
               e2e_mean=("mean_e2e_latency_s", "mean"),
               e2e_std=("mean_e2e_latency_s", "std"),
               p95_e2e_mean=("p95_e2e_latency_s", "mean"),
               routing_ms_mean=("mean_routing_decision_ms", "mean"),
               routing_ms_std=("mean_routing_decision_ms", "std"),
               qps_mean=("throughput_qps", "mean"),
               cost_per_q_mean=("mean_cost_usd", "mean"),
               cost_per_q_std=("mean_cost_usd", "std"),
               llm_calls_mean=("mean_llm_calls", "mean"),
               messages_per_q_mean=("mean_messages_per_query", "mean"),
               messages_per_q_std=("mean_messages_per_query", "std"),
               total_messages_mean=("total_messages", "mean"),
               msg_reduction_vs_broadcast_mean=(
                   "message_reduction_vs_broadcast", "mean"),
               gpu_peak_mb_max=("gpu_peak_mb", "max"))
          .reset_index()
    )


def build_message_volume_table(df: pd.DataFrame) -> pd.DataFrame:
    """Plot-ready table for the §5 figure on inter-agent message volume.

    Rows are pool sizes N; columns are routers (mean messages/query averaged
    over seeds), plus a synthetic 'Broadcast' column with the analytical
    worst-case reference value (= N). Standard error of the mean across seeds
    is included for each router as `<router>_sem`.

    Use this to draw the figure:
        - x-axis: pool_size_N (log scale)
        - y-axis: messages per query (log scale)
        - one solid line per router (essentially flat for all of them)
        - one dashed line for 'Broadcast' (linear in N, the worst case)
        - error bars from <router>_sem
    """
    Ns = sorted(df["pool_size_N"].unique().tolist())
    routers = sorted(df["router"].unique().tolist())

    rows: List[Dict[str, Any]] = []
    for N in Ns:
        row: Dict[str, Any] = {"pool_size_N": int(N)}
        df_N = df[df["pool_size_N"] == N]
        for r in routers:
            vals = df_N[df_N["router"] == r]["mean_messages_per_query"].to_numpy()
            if len(vals) > 0:
                row[r] = float(np.mean(vals))
                row[f"{r}_sem"] = (float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                                   if len(vals) > 1 else 0.0)
            else:
                row[r] = float("nan")
                row[f"{r}_sem"] = float("nan")
        # Broadcast reference: one query, fanned out to every registered agent
        row["Broadcast"] = int(N)
        row["Broadcast_sem"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# 4. MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_problems", type=int, default=100,
                   help="Number of MedQA problems to fix the workload at.")
    p.add_argument("--agent_pool_sizes", type=str,
                   default=",".join(str(n) for n in DEFAULT_POOL_SIZES),
                   help="Comma-separated N values to sweep.")
    p.add_argument("--runs", type=int, default=3,
                   help="Independent seeds per (router, N) cell.")
    p.add_argument("--seeds", type=str,
                   default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--routers", type=str, default="",
                   help="Comma-separated subset of routers to run; default = all six.")
    p.add_argument("--out_dir", type=str, default=".",
                   help="Directory to write the CSV/JSON outputs.")
    return p.parse_args()


def main():
    args = parse_args()
    Ns = [int(x) for x in args.agent_pool_sizes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")][:args.runs]
    if len(seeds) < args.runs:
        seeds += [seeds[-1] + i for i in range(1, args.runs - len(seeds) + 1)]

    chosen = ROUTER_CLASSES
    if args.routers:
        names = {s.strip() for s in args.routers.split(",")}
        chosen = [c for c in ROUTER_CLASSES if c.name in names]
    log.info(f"Routers: {[c.name for c in chosen]}")
    log.info(f"N sweep: {Ns}")
    log.info(f"Seeds:   {seeds}")
    log.info(f"Problems per cell: {args.num_problems}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = Qwen25Backbone(model_name=args.model)

    all_rows: List[ScaleRow] = []
    for seed in seeds:
        problems = load_medqa(num_problems=args.num_problems, seed=seed)
        for N in Ns:
            for router_cls in chosen:
                log.info(f"=== router={router_cls.name}  N={N}  seed={seed} ===")
                t0 = time.time()
                row = measure_one(router_cls, problems, llm, N, seed)
                dur = time.time() - t0
                _, peak = gpu_mem_mb()
                log.info(
                    f"   acc={row.accuracy:.4f}  e2e={row.mean_e2e_latency_s:.2f}s  "
                    f"route={row.mean_routing_decision_ms:.1f}ms  "
                    f"cost=${row.total_cost_usd:.4f}  GPU={peak:.0f}MB  "
                    f"wall={dur:.1f}s"
                )
                all_rows.append(row)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save raw rows
    df = pd.DataFrame([r.to_dict() for r in all_rows])
    df.to_csv(out_dir / "scalability_results.csv", index=False)
    with open(out_dir / "scalability_results.json", "w") as f:
        json.dump([r.to_dict() for r in all_rows], f, indent=2)
    log.info(f"Raw rows written to {out_dir}/scalability_results.{{csv,json}}")

    # Aggregate
    agg = aggregate(df)
    agg.to_csv(out_dir / "scalability_aggregate.csv", index=False)
    log.info("\n" + "=" * 100)
    log.info(" Aggregate: mean +/- std across seeds, per (router, N)")
    log.info("=" * 100)
    print(agg.round(4).to_string(index=False))

    # --- Section 5 figure: messages per query vs N, with Broadcast reference ---
    msg_tbl = build_message_volume_table(df)
    msg_tbl.to_csv(out_dir / "scalability_message_volume.csv", index=False)
    log.info("\n" + "=" * 100)
    log.info(" Messages per query vs N  (Section 5 figure source)")
    log.info("=" * 100)
    print(msg_tbl.round(3).to_string(index=False))
    # Sanity sentinel printed inline so the reviewer-relevant numbers are
    # visible at a glance from the log alone:
    largest_N = int(max(msg_tbl["pool_size_N"]))
    log.info(
        f"  Broadcast worst case at N={largest_N}: {largest_N} messages/query. "
        f"AgentRoute at N={largest_N}: "
        f"{float(msg_tbl.loc[msg_tbl['pool_size_N']==largest_N, 'AgentRoute'].iloc[0]):.2f} "
        f"messages/query."
    )

    # Pairwise tests
    metrics = ["accuracy", "mean_e2e_latency_s", "mean_cost_usd",
               "mean_routing_decision_ms", "mean_messages_per_query"]
    pw = per_N_pairwise(df, reference="AgentRoute", metrics=metrics)
    pw.to_csv(out_dir / "scalability_pairwise.csv", index=False)
    log.info("\n" + "=" * 100)
    log.info(" Pairwise (AgentRoute vs each baseline) per N, Bonferroni-corrected")
    log.info("=" * 100)
    if not pw.empty:
        print(pw[["pool_size_N", "comparison", "metric", "delta_mean",
                  "p_value", "cohens_d", "significant_bonferroni"]]
              .round(4).to_string(index=False))

    # Friedman tests
    fr = per_N_friedman(df, metrics=metrics)
    fr.to_csv(out_dir / "scalability_friedman.csv", index=False)
    log.info("\n" + "=" * 100)
    log.info(" Friedman chi-square per N across all six routers")
    log.info("=" * 100)
    print(fr.round(6).to_string(index=False))

    # Per-router scaling slope (the headline scalability number)
    slopes_e2e = per_router_scaling_slope(df, metric="mean_e2e_latency_s")
    slopes_cost = per_router_scaling_slope(df, metric="mean_cost_usd")
    slopes_route = per_router_scaling_slope(df, metric="mean_routing_decision_ms")
    slopes_msgs = per_router_scaling_slope(df, metric="mean_messages_per_query")
    slopes_e2e["fitted_metric"] = "mean_e2e_latency_s"
    slopes_cost["fitted_metric"] = "mean_cost_usd"
    slopes_route["fitted_metric"] = "mean_routing_decision_ms"
    slopes_msgs["fitted_metric"] = "mean_messages_per_query"
    slopes = pd.concat([slopes_e2e, slopes_cost, slopes_route, slopes_msgs],
                       ignore_index=True)
    slopes.to_csv(out_dir / "scalability_scaling_slopes.csv", index=False)
    log.info("\n" + "=" * 100)
    log.info(" Per-router scaling slopes (metric vs N), sorted by slope")
    log.info("=" * 100)
    print(slopes.round(6).to_string(index=False))

    log.info(f"\nAll outputs written under {out_dir.resolve()}/")


if __name__ == "__main__":
    main()


# =====================================================================
# HOW TO RUN
# =====================================================================
"""
# Smoke test (1 seed, 50 problems, only 3 N values, only AgentRoute + OI-MAS)
python scalability_sweep.py \\
    --num_problems 50 --agent_pool_sizes 5,25,100 --runs 1 \\
    --routers AgentRoute,OI-MAS --out_dir smoke_scalability

# Full sweep (3 seeds, 100 problems, six N values, all six routers)
# Expected wall-clock: ~3 hours on a single 16 GB GPU.
python scalability_sweep.py \\
    --num_problems 100 \\
    --agent_pool_sizes 5,10,25,50,100,200 \\
    --runs 3 --seeds 13,14,15 \\
    --out_dir scalability_out

# Output files (in --out_dir):
#   scalability_results.csv         # one row per (router, N, seed)
#   scalability_results.json        # same data as JSON
#   scalability_aggregate.csv       # mean +/- std per (router, N)
#   scalability_message_volume.csv  # §5 figure source (messages/q vs N + Broadcast)
#   scalability_pairwise.csv        # AgentRoute vs each baseline per N
#   scalability_friedman.csv        # Friedman chi-square per N
#   scalability_scaling_slopes.csv  # linear-fit slope per router for each metric
"""
