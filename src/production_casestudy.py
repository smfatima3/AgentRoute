"""
production_casestudy.py
========================

Production case study for AgentRoute (EMNLP 2026 Industry Track).

What this script does
---------------------
Runs the six routers (AgentRoute + 5 baselines) against the real Claude API
under realistic production constraints: per-minute rate limits, hard dollar
budget cap, exponential-backoff retry with jitter, and 3-seed statistical
analysis. This replaces the n=10 single-seed run in the tech report.

Workload
--------
We use a sample of MedQA-USMLE questions as a "realistic enterprise coding
assistant" stand-in: well-defined questions with verifiable answers, a clear
domain decomposition (clinical knowledge), and a result that the answer-
extraction regex can grade unambiguously.

Why Claude Haiku 4.5
--------------------
Haiku 4.5 (model id: claude-haiku-4-5-20251001) is positioned for
"high-concurrency, low-latency, cost-sensitive tasks" (Anthropic, 2026) which
matches the AgentRoute production target. Pricing is ~$1/$5 per 1M tokens
input/output, so the full study costs roughly $15-25 in API spend.

Six routers compared:
    AgentRoute, OI-MAS, RCR-Router, EvoMAS, DyTopo, RopMura

Metrics:
    accuracy, mean / p50 / p95 end-to-end latency (wall clock, includes
    network round-trip), throughput (queries/sec), mean prompt + completion
    tokens, mean LLM calls/q, mean per-query cost (USD), total spend,
    retry count, rate-limit-hit count.

Statistical analysis: identical to medqa_evaluation.py / scalability_sweep.py
    - paired t-test vs AgentRoute on accuracy / e2e / cost
    - Cohen's d (paired), 95% CI, Bonferroni
    - Friedman chi-square across all six routers
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Anthropic SDK (>= 0.49.0 verified May 2026)
try:
    import anthropic
    from anthropic import APIStatusError, RateLimitError
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "anthropic>=0.49.0 is required. Install with `pip install anthropic>=0.49.0`."
    ) from e

# Reuse the MCQ format, letter extraction, grader, agent factory, and the
# six router implementations from the MedQA evaluation script. The routers
# treat their `llm` argument as a duck-typed object exposing .generate(...);
# we replace that with a Claude-backed wrapper below.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from medqa_evaluation import (  # noqa: E402
    Agent,
    AgentRouteRouter,
    AgentRouteWrapper,
    DyTopoWrapper,
    EvoMASWrapper,
    OIMASWrapper,
    RCRRouterWrapper,
    RopMuraWrapper,
    extract_letter,
    format_mcq,
    grade_mcq,
    load_medqa,
    make_agents,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("casestudy")

ROUTER_CLASSES_CASESTUDY = [
    AgentRouteWrapper, OIMASWrapper, RCRRouterWrapper,
    EvoMASWrapper, DyTopoWrapper, RopMuraWrapper,
]


# ============================================================================
# 1. CLAUDE-BACKED LLM (duck-typed to replace Qwen25Backbone in routers)
# ============================================================================

# Verified May 2026 Anthropic pricing for Claude Haiku 4.5 ($/1M tokens).
PRICE_PER_1M = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6":         {"in": 3.00, "out": 15.00},
    "claude-opus-4-7":           {"in": 15.00, "out": 75.00},
}


class BudgetExceeded(RuntimeError):
    """Raised when the hard dollar budget cap is reached."""


@dataclass
class CallStats:
    """Mirrors medqa_evaluation.CallStats so router code is unchanged."""
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_s: float = 0.0
    e2e_latency_s: float = 0.0
    final_text: str = ""

    def add(self, prompt_t: int, comp_t: int, lat: float, text: str):
        self.n_llm_calls += 1
        self.prompt_tokens += prompt_t
        self.completion_tokens += comp_t
        self.llm_latency_s += lat
        self.final_text = text


class ClaudeBackbone:
    """Claude-API-backed inference engine.

    Exposes the same .generate(prompt, max_new_tokens, temperature) signature
    as Qwen25Backbone so the router classes don't change. Adds:
      - hard dollar budget cap (raises BudgetExceeded)
      - per-minute request & token rate-limit tracking
      - exponential backoff + jitter on 429s
      - retry-count and rate-limit-hit counters surfaced for reporting
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 budget_usd: float = 25.0,
                 max_retries: int = 6,
                 base_backoff_s: float = 2.0,
                 max_backoff_s: float = 60.0,
                 rpm_cap: int = 50,
                 itpm_cap: int = 30_000):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running this "
                "script:  export ANTHROPIC_API_KEY=sk-ant-..."
            )
        if model not in PRICE_PER_1M:
            raise ValueError(
                f"Unknown model {model}; add pricing to PRICE_PER_1M first."
            )
        self.client = anthropic.Anthropic()
        self.model_name = model
        self.price = PRICE_PER_1M[model]

        # Budget and counters
        self.budget_usd = float(budget_usd)
        self.spent_usd = 0.0
        self.n_calls = 0
        self.n_retries = 0
        self.n_429 = 0
        self.n_529 = 0

        # Retry/backoff policy
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s

        # Client-side rate-limit tracking (sliding 60-s windows)
        self.rpm_cap = rpm_cap
        self.itpm_cap = itpm_cap
        self._request_times: List[float] = []
        self._token_times: List[Tuple[float, int]] = []  # (time, input_tokens)
        # Thread-safety: production_casestudy.py runs single-threaded, but
        # concurrent_load_test.py invokes generate() from multiple worker
        # threads (asyncio.to_thread). Guard the sliding windows and the
        # spend/counter mutations so float += and list rebuilds cannot race.
        self._state_lock = threading.Lock()

    # ---- pricing -----------------------------------------------------------
    def _cost(self, in_tok: int, out_tok: int) -> float:
        return ((in_tok  / 1_000_000.0) * self.price["in"]
              + (out_tok / 1_000_000.0) * self.price["out"])

    # ---- client-side rate limit gating ------------------------------------
    def _rate_limit_wait(self, est_input_tokens: int):
        """Sleep until BOTH RPM and ITPM budgets allow the next request.
        Sleeps happen OUTSIDE the lock so other threads are not blocked."""
        while True:
            with self._state_lock:
                now = time.time()
                # Drop entries older than 60 s
                self._request_times = [t for t in self._request_times
                                       if now - t < 60.0]
                self._token_times = [(t, k) for (t, k) in self._token_times
                                     if now - t < 60.0]
                wait = 0.0
                if len(self._request_times) >= self.rpm_cap:
                    wait = max(wait, 60.0 - (now - self._request_times[0]) + 0.1)
                used_tokens = sum(k for _, k in self._token_times)
                if (self._token_times
                        and used_tokens + est_input_tokens > self.itpm_cap):
                    wait = max(wait, 60.0 - (now - self._token_times[0][0]) + 0.1)
            if wait <= 0:
                return
            log.info(f"  [rate-limit] cap reached, sleeping {wait:.1f}s")
            time.sleep(wait)

    def _record_call(self, in_tokens: int):
        now = time.time()
        with self._state_lock:
            self._request_times.append(now)
            self._token_times.append((now, in_tokens))

    # ---- main entry point --------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> Tuple[str, int, int, float]:
        """Returns (text, prompt_tokens, completion_tokens, latency_s)."""
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"Budget cap ${self.budget_usd:.2f} reached "
                f"(${self.spent_usd:.4f} spent)."
            )

        # Rough input-token estimate (conservative: 1 token ~ 3.5 chars)
        est_in = max(1, int(len(prompt) / 3.5))
        self._rate_limit_wait(est_input_tokens=est_in)

        messages = [{"role": "user", "content": prompt}]
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=messages,
                )
                latency = time.time() - t0
                # Sum text blocks (Haiku 4.5 may return >1 if it interleaves
                # thinking blocks, though by default it returns plain text)
                parts = []
                for block in resp.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                text = "".join(parts) if parts else ""
                in_tok  = int(resp.usage.input_tokens)
                out_tok = int(resp.usage.output_tokens)
                with self._state_lock:
                    self.n_calls += 1
                    self.spent_usd += self._cost(in_tok, out_tok)
                self._record_call(in_tok)
                return text, in_tok, out_tok, latency

            except RateLimitError as e:
                self.n_429 += 1
                self.n_retries += 1
                # Anthropic returns Retry-After in seconds when known
                retry_after = None
                hdrs = getattr(e, "response", None)
                if hdrs is not None:
                    try:
                        retry_after = float(e.response.headers.get(
                            "retry-after", "0") or 0)
                    except (ValueError, AttributeError):
                        retry_after = None
                # Fall back to exponential backoff with jitter
                if retry_after is None or retry_after <= 0:
                    base = min(self.max_backoff_s,
                               self.base_backoff_s * (2 ** attempt))
                    retry_after = base + random.uniform(0, base / 2)
                retry_after = min(retry_after, self.max_backoff_s)
                log.warning(
                    f"  [429] attempt {attempt + 1}/{self.max_retries + 1}, "
                    f"sleeping {retry_after:.1f}s"
                )
                time.sleep(retry_after)

            except APIStatusError as e:
                # 529 overloaded_error gets the same treatment as 429
                status = getattr(e, "status_code", None)
                if status == 529:
                    self.n_529 += 1
                    self.n_retries += 1
                    base = min(self.max_backoff_s,
                               self.base_backoff_s * (2 ** attempt))
                    delay = base + random.uniform(0, base / 2)
                    log.warning(
                        f"  [529 overloaded] attempt {attempt + 1}, "
                        f"sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(
            f"Max retries ({self.max_retries}) exhausted for prompt; aborting."
        )


# ============================================================================
# 2. PER-RUN MEASUREMENT
# ============================================================================

@dataclass
class CaseStudyRow:
    router: str
    seed: int
    n_problems: int
    n_correct: int
    accuracy: float
    mean_e2e_latency_s: float
    p50_e2e_latency_s: float
    p95_e2e_latency_s: float
    throughput_qps: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_llm_calls: float
    mean_cost_usd: float
    total_cost_usd: float
    n_retries: int
    n_429: int
    n_529: int
    completed: bool       # False if BudgetExceeded interrupted the run
    notes: str            # free-form (e.g., "budget exhausted at q=83/100")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def measure_one(router_cls, problems, claude: ClaudeBackbone,
                num_agents: int, seed: int) -> CaseStudyRow:
    random.seed(seed)
    np.random.seed(seed)

    agents = make_agents(num_agents)
    router = router_cls(agents=agents, llm=claude)
    router.reset()

    # Snapshot Claude-side counters before this run so per-run totals are clean
    n_retries_before = claude.n_retries
    n_429_before     = claude.n_429
    n_529_before     = claude.n_529

    e2e: List[float] = []
    p_toks: List[int] = []
    c_toks: List[int] = []
    calls: List[int] = []
    costs: List[float] = []
    n_correct = 0
    wall_start = time.time()
    completed = True
    notes = ""

    pbar = tqdm(problems, desc=f"{router_cls.name}/seed={seed}", leave=False)
    for i, prob in enumerate(pbar):
        try:
            st = router.solve(prob["question"], prob["options"])
        except BudgetExceeded as e:
            completed = False
            notes = f"budget exhausted at q={i}/{len(problems)}: {e}"
            log.warning(f"  Budget exhausted: {e}")
            break
        ok = grade_mcq(st.final_text, prob["gold_letter"])
        n_correct += int(ok)
        e2e.append(st.e2e_latency_s)
        p_toks.append(st.prompt_tokens)
        c_toks.append(st.completion_tokens)
        calls.append(st.n_llm_calls)
        # cost = backbone-computed actual cost (uses real input/output tokens)
        per_q_cost = ((st.prompt_tokens / 1_000_000.0) * claude.price["in"]
                    + (st.completion_tokens / 1_000_000.0) * claude.price["out"])
        costs.append(per_q_cost)
        if hasattr(router, "update_fitness"):
            router.update_fitness(ok)
        pbar.set_postfix(acc=f"{n_correct/(len(e2e)):.3f}",
                         spent=f"${claude.spent_usd:.2f}")
    wall = time.time() - wall_start
    n_done = max(1, len(e2e))

    return CaseStudyRow(
        router=router_cls.name,
        seed=seed,
        n_problems=len(e2e),
        n_correct=n_correct,
        accuracy=n_correct / n_done,
        mean_e2e_latency_s=float(np.mean(e2e)) if e2e else 0.0,
        p50_e2e_latency_s=float(np.percentile(e2e, 50)) if e2e else 0.0,
        p95_e2e_latency_s=float(np.percentile(e2e, 95)) if e2e else 0.0,
        throughput_qps=n_done / max(1e-6, wall),
        mean_prompt_tokens=float(np.mean(p_toks)) if p_toks else 0.0,
        mean_completion_tokens=float(np.mean(c_toks)) if c_toks else 0.0,
        mean_llm_calls=float(np.mean(calls)) if calls else 0.0,
        mean_cost_usd=float(np.mean(costs)) if costs else 0.0,
        total_cost_usd=float(np.sum(costs)),
        n_retries=claude.n_retries - n_retries_before,
        n_429=claude.n_429 - n_429_before,
        n_529=claude.n_529 - n_529_before,
        completed=completed,
        notes=notes,
    )


# ============================================================================
# 3. STATISTICAL ANALYSIS  (identical pattern to the other scripts)
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


def run_statistical_analysis(rows: List[CaseStudyRow],
                             reference: str = "AgentRoute") -> Dict[str, Any]:
    df = pd.DataFrame([r.to_dict() for r in rows])
    routers = df["router"].unique().tolist()
    if reference not in routers:
        return {}

    metrics = ["accuracy", "mean_e2e_latency_s", "mean_cost_usd"]
    n_metrics = len(metrics)
    n_compare = max(1, len(routers) - 1)
    bonferroni_alpha = 0.05 / (n_metrics * n_compare)

    pairwise = []
    for r in routers:
        if r == reference:
            continue
        ref = df[df["router"] == reference].sort_values("seed")
        oth = df[df["router"] == r].sort_values("seed")
        if len(ref) != len(oth) or len(ref) < 2:
            continue
        for metric in metrics:
            a = ref[metric].to_numpy()
            b = oth[metric].to_numpy()
            if np.all(a == b):
                t_stat, p = 0.0, 1.0
            else:
                t_stat, p = stats.ttest_rel(a, b)
            d = cohens_d_paired(a, b)
            lo, hi = ci95_paired(a, b)
            pairwise.append({
                "comparison": f"{reference} vs {r}",
                "metric": metric,
                "n_runs": len(a),
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

    friedman = []
    for metric in metrics:
        cols = []
        for r in routers:
            vals = df[df["router"] == r].sort_values("seed")[metric].to_numpy()
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
        friedman.append({"metric": metric, "n_routers": len(cols),
                         "chi2": float(chi2), "p_value": float(p)})

    return {
        "pairwise": pairwise,
        "friedman": friedman,
        "bonferroni_alpha": bonferroni_alpha,
        "reference": reference,
    }


# ============================================================================
# 4. MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_problems", type=int, default=100,
                   help="MedQA questions per seed.")
    p.add_argument("--runs", type=int, default=3,
                   help="Independent seeds.")
    p.add_argument("--seeds", type=str, default="13,14,15")
    p.add_argument("--num_agents", type=int, default=8)
    p.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                   help="Claude model id. Haiku 4.5 by default (cheapest).")
    p.add_argument("--budget_usd", type=float, default=25.0,
                   help="Hard dollar cap across the entire run.")
    p.add_argument("--rpm_cap", type=int, default=50,
                   help="Client-side requests-per-minute cap.")
    p.add_argument("--itpm_cap", type=int, default=30_000,
                   help="Client-side input-tokens-per-minute cap.")
    p.add_argument("--routers", type=str, default="",
                   help="Comma-separated subset of routers to run.")
    p.add_argument("--out_dir", type=str, default=".")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")][:args.runs]
    if len(seeds) < args.runs:
        seeds += [seeds[-1] + i for i in range(1, args.runs - len(seeds) + 1)]

    chosen = ROUTER_CLASSES_CASESTUDY
    if args.routers:
        names = {s.strip() for s in args.routers.split(",")}
        chosen = [c for c in ROUTER_CLASSES_CASESTUDY if c.name in names]

    log.info(f"Case study  model={args.model}  budget=${args.budget_usd:.2f}")
    log.info(f"Routers: {[c.name for c in chosen]}")
    log.info(f"Seeds:   {seeds}")
    log.info(f"Problems per seed: {args.num_problems}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    claude = ClaudeBackbone(
        model=args.model,
        budget_usd=args.budget_usd,
        rpm_cap=args.rpm_cap,
        itpm_cap=args.itpm_cap,
    )

    all_rows: List[CaseStudyRow] = []
    for seed in seeds:
        problems = load_medqa(num_problems=args.num_problems, seed=seed)
        for router_cls in chosen:
            log.info(f"=== {router_cls.name}  seed={seed} ===")
            row = measure_one(router_cls, problems, claude,
                              args.num_agents, seed)
            log.info(
                f"   acc={row.accuracy:.4f}  e2e={row.mean_e2e_latency_s:.2f}s "
                f" total=${row.total_cost_usd:.4f}  retries={row.n_retries} "
                f" 429={row.n_429}  529={row.n_529}  done={row.completed}"
            )
            all_rows.append(row)
            gc.collect()
            if not row.completed:
                log.warning(
                    f"Run interrupted by budget; remaining "
                    f"(router, seed) cells will be skipped."
                )
                # Save partial results before exiting
                _flush(all_rows, claude, out_dir, args)
                return

    _flush(all_rows, claude, out_dir, args)


def _flush(rows, claude, out_dir: Path, args):
    """Write raw results, aggregate, and statistical tables to disk."""
    df = pd.DataFrame([r.to_dict() for r in rows])
    df.to_csv(out_dir / "casestudy_results.csv", index=False)
    with open(out_dir / "casestudy_results.json", "w") as f:
        json.dump([r.to_dict() for r in rows], f, indent=2)
    log.info(f"Raw rows written to {out_dir}/casestudy_results.{{csv,json}}")

    # Aggregate
    agg = df.groupby("router").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        e2e_mean=("mean_e2e_latency_s", "mean"),
        e2e_std=("mean_e2e_latency_s", "std"),
        p95_mean=("p95_e2e_latency_s", "mean"),
        qps_mean=("throughput_qps", "mean"),
        cost_per_q_mean=("mean_cost_usd", "mean"),
        cost_per_q_std=("mean_cost_usd", "std"),
        total_cost_mean=("total_cost_usd", "mean"),
        llm_calls_mean=("mean_llm_calls", "mean"),
        retries_sum=("n_retries", "sum"),
        rate_limit_429_sum=("n_429", "sum"),
    )
    agg.to_csv(out_dir / "casestudy_aggregate.csv")
    log.info("\n" + "=" * 100)
    log.info(" Aggregate: mean +/- std across seeds")
    log.info("=" * 100)
    print(agg.round(4).to_string())

    stats_out = run_statistical_analysis(rows, reference="AgentRoute")
    if stats_out:
        pw = pd.DataFrame(stats_out["pairwise"])
        pw.to_csv(out_dir / "casestudy_pairwise.csv", index=False)
        log.info("\n" + "=" * 100)
        log.info(f" Pairwise: AgentRoute vs each baseline  "
                 f"Bonferroni alpha={stats_out['bonferroni_alpha']:.6f}")
        log.info("=" * 100)
        if not pw.empty:
            print(pw[["comparison", "metric", "delta_mean", "p_value",
                      "cohens_d", "significant_bonferroni"]]
                  .round(4).to_string(index=False))

        fr = pd.DataFrame(stats_out["friedman"])
        fr.to_csv(out_dir / "casestudy_friedman.csv", index=False)
        log.info("\n" + "=" * 100)
        log.info(" Friedman chi-square across all six routers")
        log.info("=" * 100)
        print(fr.round(6).to_string(index=False))

        with open(out_dir / "casestudy_stats.json", "w") as f:
            json.dump(stats_out, f, indent=2)

    # Final budget report
    log.info("\n" + "=" * 100)
    log.info(" Claude API spend summary")
    log.info("=" * 100)
    log.info(f"  model           : {claude.model_name}")
    log.info(f"  total spend     : ${claude.spent_usd:.4f} "
             f"(cap ${claude.budget_usd:.2f})")
    log.info(f"  total API calls : {claude.n_calls}")
    log.info(f"  total retries   : {claude.n_retries}")
    log.info(f"  429 hits        : {claude.n_429}")
    log.info(f"  529 hits        : {claude.n_529}")


if __name__ == "__main__":
    main()


# =====================================================================
# HOW TO RUN
# =====================================================================
"""
# 1. Install deps (one-time)
pip install "anthropic>=0.49.0" datasets pandas scipy tqdm

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Smoke test (1 seed, 10 questions, AgentRoute + OI-MAS only, $1 cap)
python production_casestudy.py \\
    --num_problems 10 --runs 1 --seeds 13 \\
    --routers AgentRoute,OI-MAS \\
    --budget_usd 1.00 --out_dir smoke_casestudy

# 4. Full run (3 seeds x 100 questions x 6 routers on Haiku 4.5)
# Expected spend: ~$15-25.  Expected wall-clock: ~45-90 min (rate-limited).
python production_casestudy.py \\
    --num_problems 100 --runs 3 --seeds 13,14,15 \\
    --model claude-haiku-4-5-20251001 \\
    --budget_usd 25.00 --rpm_cap 50 --itpm_cap 30000 \\
    --out_dir casestudy_out

# Output files (in --out_dir):
#   casestudy_results.csv       # one row per (router, seed)
#   casestudy_results.json
#   casestudy_aggregate.csv     # mean +/- std per router
#   casestudy_pairwise.csv      # paired t-tests vs AgentRoute
#   casestudy_friedman.csv      # Friedman chi-square per metric
#   casestudy_stats.json        # full statistical-analysis blob
"""
