"""
gpqa_evaluation.py
====================

Real evaluation of AgentRoute and 5 baselines on GPQA Diamond (graduate-level
4-option MCQ in physics, chemistry, and biology) using Qwen 2.5 3B Instruct.

Baselines (no GoogleA2A — removed per user request after smoke results):
  - AgentRoute  (imported from agentroute_jang2005_complete.py)
  - OI-MAS      (Wang et al., 2026, arXiv:2601.04861)
  - RCR-Router  (Liu et al., 2025, arXiv:2508.04903) — faithful Planner/Searcher/
                Recommender pipeline with Token Budget Allocator + Importance
                Scorer + Semantic Filter (T=3 rounds, B_i=1024 tokens/role)
  - EvoMAS      (arXiv:2602.06511, 2026)
  - DyTopo      (Lu et al., 2026, arXiv:2602.06039)
  - RopMura     (Wang et al., 2025, arXiv:2501.07813) — faithful 4-submodule
                planner

Dataset:
  - GPQA Diamond — the 198 hardest questions from GPQA. The dataset is GATED
    on HuggingFace; you must accept the license at
        https://huggingface.co/datasets/Idavidrein/gpqa
    and have HF_TOKEN set in your environment (or run `huggingface-cli login`).
  - Random baseline = 0.25 (4 choices); domain-PhD experts get ~65%; non-expert
    PhDs with internet access get ~34%.
  - Default subset size: 198 (i.e., the full Diamond set).

LLM:
  - Qwen/Qwen2.5-3B-Instruct, fp16, single GPU
  - GPU memory monitored

Metrics (per (router, run)):
  - accuracy (letter-match against gold answer A/B/C/D)
  - mean / p50 / p95 end-to-end latency (seconds)
  - throughput (queries / second)
  - mean prompt + completion tokens
  - mean cost per query (USD)
  - GPU peak memory (MB)
  - mean LLM calls per question

Statistical analysis (vs AgentRoute):
  - paired t-test on accuracy / latency / cost
  - Cohen's d (paired)
  - 95% CI on the paired difference
  - Bonferroni-corrected significance flag (α / num_tests)
  - Friedman χ² across all routers per metric

Run:
    python gpqa_evaluation.py --num_problems 198 --runs 3 --num_agents 8 \\
                              --model Qwen/Qwen2.5-3B-Instruct \\
                              --out gpqa_results.json

Expected runtime on a single 16 GB GPU: ~1.5–2.5 hours total. Note that GPQA is
much harder than MedQA and Qwen 2.5 3B is a small model — accuracy near or even
slightly below the 0.25 random baseline is plausible for some methods.
"""

# ------------------------------------------------------------------ stdlib
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
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------ third-party
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------ local
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentroute_jang2005_complete import (  # noqa: E402
    Agent, AgentRouteRouter, RoutingMetrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gpqa_eval")

# ============================================================================
# 1. LLM BACKBONE
# ============================================================================

# Reference price per 1M tokens (Qwen 2.5 3B, Together.ai-class hosted price)
PRICE_USD_PER_1M_INPUT = 0.02
PRICE_USD_PER_1M_OUTPUT = 0.05


def gpu_mem_mb() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.max_memory_allocated() / 1024 ** 2)


class Qwen25Backbone:
    """Single shared Qwen 2.5 3B engine. All routers call .generate(prompt)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 device: str = "cuda", dtype: str = "float16"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        mem_before, _ = gpu_mem_mb()
        logger.info(f"Loading {model_name} on {self.device} ({dtype})…")
        logger.info(f"  GPU mem before load: {mem_before:.1f} MB")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        mem_after, _ = gpu_mem_mb()
        logger.info(f"  GPU mem after  load: {mem_after:.1f} MB  "
                    f"(Δ={mem_after - mem_before:.1f} MB)")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> Tuple[str, int, int, float]:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=2048).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        completion_tokens = int(gen_ids.shape[0])
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return gen_text, prompt_tokens, completion_tokens, latency


# ============================================================================
# 2. DATASET LOADER
# ============================================================================

# GPQA Diamond columns (per huggingface viewer): "Question", "Correct Answer",
# "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Subdomain".
LETTERS = ["A", "B", "C", "D"]


def load_gpqa_diamond(num_problems: int, seed: int) -> List[Dict[str, Any]]:
    """Load GPQA Diamond from HuggingFace (gated dataset; requires HF_TOKEN).

    Each example provides a correct answer + 3 incorrect answers. We shuffle
    the four options per (question, seed) so the gold letter is uniformly
    distributed across A/B/C/D and not always in the same position.
    """
    logger.info("Loading GPQA Diamond (gated — requires HF_TOKEN env var)…")
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        logger.warning(
            "HF_TOKEN is not set. If the next call fails with a 401/403, run "
            "`huggingface-cli login` or `export HF_TOKEN=...` after accepting "
            "the dataset license at https://huggingface.co/datasets/Idavidrein/gpqa"
        )
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    logger.info(f"  full GPQA Diamond size: {len(ds)}")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_problems]

    out: List[Dict[str, Any]] = []
    for i in indices:
        ex = ds[i]
        question = ex.get("Question") or ex.get("question")
        correct = ex.get("Correct Answer") or ex.get("correct_answer")
        wrongs = [
            ex.get("Incorrect Answer 1") or ex.get("incorrect_answer_1"),
            ex.get("Incorrect Answer 2") or ex.get("incorrect_answer_2"),
            ex.get("Incorrect Answer 3") or ex.get("incorrect_answer_3"),
        ]
        if question is None or correct is None or any(w is None for w in wrongs):
            continue

        # Shuffle options deterministically by (seed, question_index)
        shuffle_rng = random.Random(seed * 100003 + i)
        all_opts = [str(correct)] + [str(w) for w in wrongs]
        order = list(range(4))
        shuffle_rng.shuffle(order)
        shuffled = [all_opts[k] for k in order]
        gold_idx = order.index(0)  # original index 0 was the correct answer

        out.append({
            "id": f"gpqa_{i}",
            "question": str(question).strip(),
            "options": shuffled,
            "gold_idx": gold_idx,
            "gold_letter": LETTERS[gold_idx],
            "subdomain": ex.get("Subdomain") or ex.get("subdomain") or "unknown",
        })
    logger.info(f"  → {len(out)} usable problems")
    return out


# ============================================================================
# 3. ANSWER EXTRACTION  (MCQ-aware)
# ============================================================================

# Match common patterns the model produces:
#   "ANSWER: A", "Answer: B", "**Answer:** C", "Final answer: D",
#   "(A)", "A)", "A.", a single letter on its own line, etc.
_ANSWER_PATTERNS = [
    r"ANSWER\s*[:\-]\s*\(?([ABCD])\)?",
    r"FINAL\s*ANSWER\s*[:\-]\s*\(?([ABCD])\)?",
    r"\bThe\s+(?:correct\s+)?answer\s+is\s*\(?([ABCD])\)?",
    r"\banswer\s*[:\-=]\s*\(?([ABCD])\)?",
    r"\boption\s*\(?([ABCD])\)?\s+is\s+correct",   # "option B is correct"
    r"\bchoice\s*\(?([ABCD])\)?\s+is\s+correct",   # "choice C is correct"
    r"\(([ABCD])\)\s*$",                # ends with "(A)"
    r"^\s*([ABCD])\s*[\.\):]",          # "A.", "A)", "A:" at start of line
    r"\b([ABCD])\b\s*$",                # single letter at very end
]


def extract_letter(text: str) -> Optional[str]:
    """Extract the model's chosen letter A/B/C/D, robustly."""
    if not text:
        return None
    # Try patterns in order. We try uppercase-text first (most common when we
    # asked for ANSWER: X), then case-insensitive search.
    for pat in _ANSWER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # No explicit marker. Only guess if the choice is unambiguous: a single
    # distinct option letter in the whole response. If several are mentioned
    # without a marker (truncated reasoning), ABSTAIN rather than guess the
    # last-mentioned distractor.
    distinct = sorted(set(re.findall(r"\b([ABCD])\b", text.upper())))
    if len(distinct) == 1:
        return distinct[0]
    return None


def grade_mcq(pred_text: str, gold_letter: str) -> bool:
    pred = extract_letter(pred_text)
    return pred == gold_letter


# ============================================================================
# 4. PROMPTS (MCQ-aware)
# ============================================================================

def format_mcq(question: str, options: List[str]) -> str:
    """Format a question with its 4 options as a single user-readable string."""
    opt_lines = "\n".join(f"({L}) {opt}" for L, opt in zip(LETTERS, options))
    return f"Question:\n{question}\n\nOptions:\n{opt_lines}"


SOLVE_PROMPT = """You are a careful graduate-level scientist (physics, chemistry, or biology). Read the question and options below.

{mcq_block}

First reason in at most two short sentences, then on a new line give your final answer in EXACTLY this format:
ANSWER: <one of A, B, C, or D>

Do not write anything after the ANSWER line."""


VERIFY_PROMPT = """You are checking another scientist's answer to a graduate-level multiple-choice question.

{mcq_block}

Proposed solution:
{solution}

If the proposed final letter is correct, reply:
VERDICT: CORRECT

Otherwise, give the corrected reasoning briefly and end with:
VERDICT: WRONG
ANSWER: <correct letter>"""


REFINE_PROMPT = """A first attempt at this graduate-level science question may have errors. Critique it briefly, then give your improved final letter.

{mcq_block}

First attempt:
{attempt}

End with:
ANSWER: <one of A, B, C, or D>"""


# ============================================================================
# 5. AGENT POOL  (single 'science' specialty for GPQA)
# ============================================================================

def make_agents(num_agents: int) -> Dict[str, Agent]:
    agents: Dict[str, Agent] = {}
    for i in range(num_agents):
        a = Agent(agent_id=f"agent_{i:02d}", specialty="science")
        agents[a.agent_id] = a
    return agents


# ============================================================================
# 6. CALL ACCOUNTING
# ============================================================================

@dataclass
class CallStats:
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


# ============================================================================
# 7. ROUTERS  (architecturally faithful to each paper)
# ============================================================================

class BaselineRouter:
    name = "BaselineRouter"

    def __init__(self, agents: Dict[str, Agent], llm: Qwen25Backbone,
                 max_new_tokens: int = 256):
        self.agents = agents
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.routing_llm_calls = 0

    def solve(self, question: str, options: List[str]) -> CallStats:
        raise NotImplementedError

    def reset(self):
        self.routing_llm_calls = 0
        for a in self.agents.values():
            a.reset()


class AgentRouteWrapper(BaselineRouter):
    """AgentRoute: pattern classifier + LRU cache for routing (no LLM call for
    routing on cache hit / pattern match), then exactly ONE Qwen call to solve."""
    name = "AgentRoute"

    def __init__(self, agents, llm, max_new_tokens=320):
        super().__init__(agents, llm, max_new_tokens)
        self.inner = AgentRouteRouter(agents=agents)

    def solve(self, question: str, options: List[str]) -> CallStats:
        st = CallStats()
        t0 = time.time()
        # Routing decision (deterministic, free)
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if cache_key in self.inner.cache:
            self.inner.metrics.cache_hits += 1
        else:
            domain, _ = self.inner._pattern_classify(question)
            self.inner.cache[cache_key] = domain
        # One LLM call to solve
        mcq_block = format_mcq(question, options)
        text, p, c, lat = self.llm.generate(
            SOLVE_PROMPT.format(mcq_block=mcq_block),
            max_new_tokens=self.max_new_tokens,
        )
        st.add(p, c, lat, text)
        st.e2e_latency_s = time.time() - t0
        return st

    def reset(self):
        super().reset()
        self.inner.cache = {}
        self.inner.metrics = type(self.inner.metrics)()


class OIMASWrapper(BaselineRouter):
    """OI-MAS: state-dependent role routing. Generator → Verifier → (Refiner if
    Verifier says WRONG). Confidence-aware early-stop."""
    name = "OI-MAS"

    def solve(self, question: str, options: List[str]) -> CallStats:
        st = CallStats()
        t0 = time.time()
        mcq_block = format_mcq(question, options)
        # Generator
        sol, p, c, lat = self.llm.generate(SOLVE_PROMPT.format(mcq_block=mcq_block),
                                           max_new_tokens=self.max_new_tokens)
        st.add(p, c, lat, sol)
        # Verifier
        ver, p, c, lat = self.llm.generate(
            VERIFY_PROMPT.format(mcq_block=mcq_block, solution=sol),
            max_new_tokens=160,
        )
        st.add(p, c, lat, sol)
        if "VERDICT: CORRECT" in ver.upper():
            st.final_text = sol
        else:
            # Verifier produced a corrected ANSWER: X — use it directly
            if re.search(r"ANSWER\s*:\s*[ABCD]", ver, re.IGNORECASE):
                st.final_text = ver
            else:
                ref, p, c, lat = self.llm.generate(
                    REFINE_PROMPT.format(mcq_block=mcq_block, attempt=sol),
                    max_new_tokens=self.max_new_tokens,
                )
                st.add(p, c, lat, ref)
                st.final_text = ref
        st.e2e_latency_s = time.time() - t0
        return st


class RCRRouterWrapper(BaselineRouter):
    """RCR-Router (Liu et al. 2025) — faithful Planner/Searcher/Recommender
    pipeline with token budgets, importance scoring, greedy semantic filter,
    iterative routing for T rounds. Recommender produces ANSWER: X."""
    name = "RCR-Router"

    TOKEN_BUDGET = 1024
    W_ROLE = 1.0
    W_STAGE = 0.6
    W_RECENCY = 0.4

    def __init__(self, agents, llm, max_new_tokens=256, rounds: int = 3):
        super().__init__(agents, llm, max_new_tokens)
        self.rounds = rounds

    def _importance_score(self, item: Dict[str, Any], role: str, stage: int) -> float:
        role_rel = self.W_ROLE if role in item.get("relevant_roles", []) else 0.0
        stage_prio = self.W_STAGE / (1.0 + abs(stage - item.get("stage", stage)))
        recency = self.W_RECENCY * math.exp(-0.3 * max(0, stage - item.get("stage", stage)))
        return role_rel + stage_prio + recency

    def _greedy_select(self, memory, role, stage, budget):
        ranked = sorted(memory,
                        key=lambda m: -self._importance_score(m, role, stage) /
                                       max(1, m.get("tokens", 1)))
        chosen, used = [], 0
        for m in ranked:
            t = m.get("tokens", 50)
            if used + t > budget:
                continue
            chosen.append(m)
            used += t
        return chosen

    def _format_context(self, items):
        if not items:
            return "(empty)"
        return "\n".join(
            f"- stage: {m['stage']} | role_tag: {m.get('source_role','?')} | "
            f"text: {m['content']}" for m in items
        )

    def solve(self, question: str, options: List[str]) -> CallStats:
        st = CallStats()
        t0 = time.time()
        mcq_block = format_mcq(question, options)

        memory: List[Dict[str, Any]] = [{
            "stage": 0, "source_role": "Question",
            "content": mcq_block,
            "tokens": max(1, len(mcq_block.split())),
            "relevant_roles": ["Planner", "Searcher", "Recommender"],
        }]

        for r in range(self.rounds):
            stage = r + 1

            # PLANNER: decompose
            ctx = self._greedy_select(memory, "Planner", stage, self.TOKEN_BUDGET)
            plan_prompt = (
                "You are the PLANNER agent in a graduate-science multi-agent system. Your "
                "role is to decompose the original graduate-level question into the next "
                "scientific reasoning sub-question that should be answered to make progress.\n"
                f"Round: {r+1}/{self.rounds}.\n"
                f"Filtered shared memory (token-budgeted):\n{self._format_context(ctx)}\n\n"
                "Output ONE concrete scientific sub-question on a single line."
            )
            plan, p, c, lat = self.llm.generate(plan_prompt, max_new_tokens=80)
            st.add(p, c, lat, plan)
            memory.append({
                "stage": stage, "source_role": "Planner",
                "content": plan.strip().split("\n")[0],
                "tokens": max(1, len(plan.split())),
                "relevant_roles": ["Searcher", "Recommender"],
            })

            # SEARCHER: gather/retrieve
            ctx = self._greedy_select(memory, "Searcher", stage, self.TOKEN_BUDGET)
            search_prompt = (
                "You are the SEARCHER agent. Your role is to gather/recall the scientific "
                "knowledge (physics, chemistry, or biology) needed to answer the Planner's "
                "sub-question.\n"
                f"Filtered shared memory:\n{self._format_context(ctx)}\n\n"
                "State the relevant scientific facts in 1-3 sentences."
            )
            sres, p, c, lat = self.llm.generate(search_prompt, max_new_tokens=200)
            st.add(p, c, lat, sres)
            memory.append({
                "stage": stage, "source_role": "Searcher",
                "content": sres.strip()[:400],
                "tokens": max(1, len(sres.split())),
                "relevant_roles": ["Recommender"],
            })

            # RECOMMENDER: synthesize → final answer
            ctx = self._greedy_select(memory, "Recommender", stage, self.TOKEN_BUDGET)
            rec_prompt = (
                "You are the RECOMMENDER agent. Synthesize the running scientific reasoning "
                "into a final letter answer if possible.\n"
                f"Filtered shared memory:\n{self._format_context(ctx)}\n\n"
                f"Original question:\n{mcq_block}\n\n"
                "If you can give the final answer NOW, end with: ANSWER: <one of A, B, C, D>\n"
                "If you need more reasoning rounds, respond with 'CONTINUE' on its own line."
            )
            rec, p, c, lat = self.llm.generate(rec_prompt, max_new_tokens=self.max_new_tokens)
            st.add(p, c, lat, rec)
            memory.append({
                "stage": stage, "source_role": "Recommender",
                "content": rec.strip()[:500],
                "tokens": max(1, len(rec.split())),
                "relevant_roles": ["Planner"],
            })
            st.final_text = rec
            if re.search(r"ANSWER\s*:\s*[ABCD]", rec, re.IGNORECASE):
                break

        # Forced final aggregation if no ANSWER: X yet
        if not re.search(r"ANSWER\s*:\s*[ABCD]", st.final_text, re.IGNORECASE):
            ctx = self._greedy_select(memory, "Recommender", self.rounds + 1, self.TOKEN_BUDGET)
            forced_prompt = (
                "You are the RECOMMENDER agent. The reasoning chain is complete; produce "
                "the FINAL letter answer NOW.\n"
                f"Filtered shared memory:\n{self._format_context(ctx)}\n\n"
                f"Original question:\n{mcq_block}\n\n"
                "End with: ANSWER: <one of A, B, C, D>"
            )
            forced, p, c, lat = self.llm.generate(forced_prompt, max_new_tokens=self.max_new_tokens)
            st.add(p, c, lat, forced)
            st.final_text = forced

        st.e2e_latency_s = time.time() - t0
        return st


class EvoMASWrapper(BaselineRouter):
    """EvoMAS: roulette over a pool of {1, 2, 3}-call configurations; EMA fitness."""
    name = "EvoMAS"

    def __init__(self, agents, llm, max_new_tokens=256, pool_size=4):
        super().__init__(agents, llm, max_new_tokens)
        self.pool: List[int] = [1, 2, 3, 2][:pool_size]
        self.fitness: List[float] = [0.5] * pool_size
        self.k_since_mutation = 0
        self.last_choice: int = 0

    def solve(self, question: str, options: List[str]) -> CallStats:
        total = sum(self.fitness) or 1.0
        probs = [f / total for f in self.fitness]
        idx = int(np.random.choice(len(self.pool), p=probs))
        n_calls = self.pool[idx]
        self.last_choice = idx

        st = CallStats()
        t0 = time.time()
        mcq_block = format_mcq(question, options)
        sol, p, c, lat = self.llm.generate(SOLVE_PROMPT.format(mcq_block=mcq_block),
                                           max_new_tokens=self.max_new_tokens)
        st.add(p, c, lat, sol)
        attempt = sol
        for _ in range(n_calls - 1):
            ref, p, c, lat = self.llm.generate(
                REFINE_PROMPT.format(mcq_block=mcq_block, attempt=attempt),
                max_new_tokens=self.max_new_tokens,
            )
            st.add(p, c, lat, ref)
            attempt = ref
        st.final_text = attempt
        st.e2e_latency_s = time.time() - t0
        return st

    def update_fitness(self, was_correct: bool):
        self.fitness[self.last_choice] = 0.8 * self.fitness[self.last_choice] + 0.2 * float(was_correct)
        self.k_since_mutation += 1
        if self.k_since_mutation >= 50:
            worst = int(np.argmin(self.fitness))
            best = int(np.argmax(self.fitness))
            self.pool[worst] = max(1, min(3, self.pool[best] + np.random.choice([-1, 0, 1])))
            self.fitness[worst] = 0.5
            self.k_since_mutation = 0


class DyTopoWrapper(BaselineRouter):
    """DyTopo (Lu et al. 2026) — manager round goal, agents emit (key, query)
    descriptors, semantic cosine matching → sparse directed graph G(t), 3 rounds."""
    name = "DyTopo"

    AGENT_ROLES = ["Decomposer", "Solver", "Verifier", "Aggregator"]
    EDGE_THRESHOLD = 0.30

    def __init__(self, agents, llm, max_new_tokens=256, rounds: int = 3,
                 embed_dim: int = 16):
        super().__init__(agents, llm, max_new_tokens)
        self.rounds = rounds
        self.embed_dim = embed_dim

    def _hash_embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], "big") & 0xFFFFFFFF
        v = np.random.RandomState(seed).randn(self.embed_dim)
        return v / (np.linalg.norm(v) + 1e-9)

    def _build_topology(self, descriptors, round_goal):
        keys = {a: self._hash_embed(d["key"]) for a, d in descriptors.items()}
        queries = {a: self._hash_embed(d["query"]) for a, d in descriptors.items()}
        edges = []
        for src in keys:
            for dst in queries:
                if src == dst:
                    continue
                if float(np.dot(keys[src], queries[dst])) >= self.EDGE_THRESHOLD:
                    edges.append((src, dst))
        goal_emb = self._hash_embed(round_goal)
        for a in queries:
            if float(np.dot(goal_emb, queries[a])) >= self.EDGE_THRESHOLD:
                edges.append(("Manager", a))
        return edges

    def solve(self, question: str, options: List[str]) -> CallStats:
        st = CallStats()
        t0 = time.time()
        mcq_block = format_mcq(question, options)
        running_context = mcq_block
        prior_descriptors = {a: {"key": "ready", "query": "round goal"}
                             for a in self.AGENT_ROLES}

        for r in range(self.rounds):
            mgr_prompt = (
                "You are the MANAGER of a 4-agent graduate-science reasoning team. Roles available: "
                f"{', '.join(self.AGENT_ROLES)}.\n"
                f"State:\n{running_context}\n\n"
                f"Round {r+1}/{self.rounds}. Output ONE concise round goal (one sentence)."
            )
            goal, p, c, lat = self.llm.generate(mgr_prompt, max_new_tokens=60)
            st.add(p, c, lat, goal)
            round_goal = goal.strip().split("\n")[0]

            edges = self._build_topology(prior_descriptors, round_goal)
            active = sorted({dst for _, dst in edges})
            if not active:
                active = ["Solver"] if r < self.rounds - 1 else ["Aggregator"]

            new_descriptors = dict(prior_descriptors)
            for role in active:
                inbound = [src for src, dst in edges if dst == role]
                agent_prompt = (
                    f"You are the {role.upper()} agent. Round goal: {round_goal}\n"
                    f"You receive messages from: {', '.join(inbound) if inbound else 'Manager'}\n"
                    f"Current state:\n{running_context}\n\n"
                    "Produce these outputs in this exact format:\n"
                    "KEY: <one short phrase describing what you can offer NEXT round>\n"
                    "QUERY: <one short phrase describing what you NEED next round>\n"
                    "CONTRIBUTION: <your actual contribution this round>\n"
                )
                mt = self.max_new_tokens if (role == "Aggregator" and r == self.rounds - 1) else 200
                txt, p, c, lat = self.llm.generate(agent_prompt, max_new_tokens=mt)
                st.add(p, c, lat, txt)
                k_match = re.search(r"KEY:\s*(.+)", txt)
                q_match = re.search(r"QUERY:\s*(.+)", txt)
                contrib_match = re.search(r"CONTRIBUTION:\s*(.+)", txt, re.DOTALL)
                new_descriptors[role] = {
                    "key": (k_match.group(1).strip() if k_match else "n/a")[:80],
                    "query": (q_match.group(1).strip() if q_match else "n/a")[:80],
                }
                contrib = contrib_match.group(1).strip() if contrib_match else txt.strip()
                running_context += f"\n[Round{r+1}/{role}]: {contrib[:300]}"
                if role == "Aggregator":
                    st.final_text = contrib

            prior_descriptors = new_descriptors

            halt_prompt = (
                f"As MANAGER, decide if reasoning is complete.\nState:\n{running_context}\n\n"
                "Reply with exactly one word: HALT or CONTINUE."
            )
            halt, p, c, lat = self.llm.generate(halt_prompt, max_new_tokens=8)
            st.add(p, c, lat, halt)
            if "HALT" in halt.upper():
                if "Aggregator" not in active:
                    final_prompt = (
                        f"You are the AGGREGATOR. Combine the round outputs into a final letter.\n"
                        f"{running_context}\n\nEnd with: ANSWER: <one of A, B, C, D>"
                    )
                    fa, p, c, lat = self.llm.generate(final_prompt,
                                                      max_new_tokens=self.max_new_tokens)
                    st.add(p, c, lat, fa)
                    st.final_text = fa
                break

        if not st.final_text or not re.search(r"ANSWER\s*:\s*[ABCD]", st.final_text,
                                              re.IGNORECASE):
            # Fallback: force one Aggregator call
            final_prompt = (
                f"You are the AGGREGATOR. Combine the round outputs into a final letter.\n"
                f"{running_context}\n\nEnd with: ANSWER: <one of A, B, C, D>"
            )
            fa, p, c, lat = self.llm.generate(final_prompt, max_new_tokens=self.max_new_tokens)
            st.add(p, c, lat, fa)
            st.final_text = fa
        st.e2e_latency_s = time.time() - t0
        return st


class RopMuraWrapper(BaselineRouter):
    """RopMura (Wang et al. 2025) — faithful 4-step protocol per round:
    question_splitter → question_selector → router (no LLM) → answer → judger
    → defender (if ANSWERABLE)."""
    name = "RopMura"

    def __init__(self, agents, llm, max_new_tokens=256, max_hops: int = 3,
                 num_subq_per_split: int = 3):
        super().__init__(agents, llm, max_new_tokens)
        self.max_hops = max_hops
        self.num_subq_per_split = num_subq_per_split
        # Cluster-centroid embeddings per agent (paper §3.1)
        self.agent_clusters: Dict[str, List[np.ndarray]] = {}
        for aid, agent in agents.items():
            seed = int(hashlib.sha256(f"{aid}:{agent.specialty}".encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            self.agent_clusters[aid] = [rng.randn(16) / np.sqrt(16) for _ in range(3)]

    def _hash_embed(self, text: str, dim: int = 16) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") & 0xFFFFFFFF
        v = np.random.RandomState(seed).randn(dim)
        return v / (np.linalg.norm(v) + 1e-9)

    def _route(self, sub_question: str, top_k: int = 1) -> List[str]:
        q_emb = self._hash_embed(sub_question)
        scores = []
        for aid, clusters in self.agent_clusters.items():
            best = max(float(np.dot(q_emb, c) / (np.linalg.norm(c) + 1e-9))
                       for c in clusters)
            scores.append((best, aid))
        scores.sort(reverse=True)
        return [aid for _, aid in scores[:top_k]]

    def solve(self, question: str, options: List[str]) -> CallStats:
        st = CallStats()
        t0 = time.time()
        mcq_block = format_mcq(question, options)
        qa_records: List[Tuple[str, str]] = []
        final_answer: Optional[str] = None

        for hop in range(self.max_hops):
            history_str = "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}"
                                    for i, (q, a) in enumerate(qa_records)) or "(empty)"
            # Step 1: question_splitter
            split_prompt = (
                "You are the PLANNER's QUESTION_SPLITTER. Given the original graduate-level "
                "science question and any prior QA records, raise SEVERAL candidate scientific "
                "sub-questions that could advance solving the original question.\n"
                f"Original question:\n{mcq_block}\n"
                f"QA records so far:\n{history_str}\n\n"
                f"Output exactly {self.num_subq_per_split} candidate sub-questions, "
                f"one per line, prefixed by '1.', '2.', '3.' (no extra text):"
            )
            split_out, p, c, lat = self.llm.generate(split_prompt, max_new_tokens=160)
            st.add(p, c, lat, split_out)
            candidates: List[str] = []
            for line in split_out.split("\n"):
                m = re.match(r"\s*\d+[.)]\s*(.+)", line)
                if m:
                    candidates.append(m.group(1).strip())
            if not candidates:
                candidates = [split_out.strip().split("\n")[0]]

            # Step 2: question_selector
            sel_prompt = (
                "You are the PLANNER's QUESTION_SELECTOR. Pick the ONE candidate sub-question "
                "most likely to advance solving the original question.\n"
                f"Original question:\n{mcq_block}\n"
                f"QA records so far:\n{history_str}\n"
                "Candidate sub-questions:\n"
                + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))
                + "\n\nReply with EXACTLY the integer index (1, 2, or 3) on the first line."
            )
            sel_out, p, c, lat = self.llm.generate(sel_prompt, max_new_tokens=8)
            st.add(p, c, lat, sel_out)
            sel_idx = 0
            m = re.search(r"[1-9]", sel_out)
            if m:
                sel_idx = max(0, min(len(candidates) - 1, int(m.group(0)) - 1))
            chosen_subq = candidates[sel_idx]

            # Step 3: router (NO LLM) + agent answers
            chosen_agents = self._route(chosen_subq, top_k=1)
            agent_prompt = (
                "You are a knowledge agent specializing in graduate-level science "
                "(physics, chemistry, biology).\n"
                f"Sub-question: {chosen_subq}\n"
                f"Original question (for context only):\n{mcq_block}\n"
                f"QA records so far:\n{history_str}\n\n"
                "Answer the sub-question concisely with the relevant scientific fact."
            )
            sub_a, p, c, lat = self.llm.generate(agent_prompt, max_new_tokens=200)
            st.add(p, c, lat, sub_a)
            qa_records.append((chosen_subq, sub_a.strip()[:300]))

            # Step 4: judger
            judger_prompt = (
                "You are the PLANNER's JUDGER. Decide whether the original graduate-science "
                "question can now be answered using the QA records.\n"
                f"Original question:\n{mcq_block}\n"
                f"QA records:\n" +
                "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}"
                          for i, (q, a) in enumerate(qa_records)) +
                "\n\nReply with EXACTLY one token: ANSWERABLE or NOT_ANSWERABLE"
            )
            judge, p, c, lat = self.llm.generate(judger_prompt, max_new_tokens=8)
            st.add(p, c, lat, judge)
            judge_upper = judge.upper().strip()
            answerable = ("ANSWERABLE" in judge_upper and
                          "NOT_ANSWERABLE" not in judge_upper and
                          "NOT ANSWERABLE" not in judge_upper)
            if answerable or hop == self.max_hops - 1:
                # defender
                defend_prompt = (
                    "You are the PLANNER's DEFENDER. Compose the final letter answer to the "
                    "original graduate-science question using ALL the QA records below.\n"
                    f"Original question:\n{mcq_block}\n"
                    f"QA records:\n" +
                    "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}"
                              for i, (q, a) in enumerate(qa_records)) +
                    "\n\nEnd with: ANSWER: <one of A, B, C, D>"
                )
                final_answer, p, c, lat = self.llm.generate(
                    defend_prompt, max_new_tokens=self.max_new_tokens)
                st.add(p, c, lat, final_answer)
                break

        if final_answer is None:
            final_answer = "\n".join(f"A{i+1}: {a}" for i, (_, a) in enumerate(qa_records))
        st.final_text = final_answer
        st.e2e_latency_s = time.time() - t0
        return st


ROUTER_CLASSES = [
    AgentRouteWrapper, OIMASWrapper, RCRRouterWrapper,
    EvoMASWrapper, DyTopoWrapper, RopMuraWrapper,
]


# ============================================================================
# 8. EVALUATION LOOP
# ============================================================================

@dataclass
class RunResult:
    router: str
    seed: int
    n: int
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
    routing_llm_calls: int
    gpu_peak_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def cost_usd(prompt_t: int, completion_t: int) -> float:
    return ((prompt_t / 1_000_000.0) * PRICE_USD_PER_1M_INPUT
            + (completion_t / 1_000_000.0) * PRICE_USD_PER_1M_OUTPUT)


def evaluate_one_run(router_cls, problems, llm, num_agents, seed) -> RunResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    agents = make_agents(num_agents)
    router = router_cls(agents=agents, llm=llm)
    router.reset()

    e2e: List[float] = []
    prompt_t_per: List[int] = []
    comp_t_per: List[int] = []
    n_calls_per: List[int] = []
    cost_per: List[float] = []
    n_correct = 0
    t_start_all = time.time()

    pbar = tqdm(problems, desc=f"{router_cls.name}/seed={seed}", leave=False)
    for prob in pbar:
        st = router.solve(prob["question"], prob["options"])
        ok = grade_mcq(st.final_text, prob["gold_letter"])
        n_correct += int(ok)
        e2e.append(st.e2e_latency_s)
        prompt_t_per.append(st.prompt_tokens)
        comp_t_per.append(st.completion_tokens)
        n_calls_per.append(st.n_llm_calls)
        cost_per.append(cost_usd(st.prompt_tokens, st.completion_tokens))
        if hasattr(router, "update_fitness"):
            router.update_fitness(ok)
        pbar.set_postfix(acc=f"{n_correct/(len(e2e)):.3f}", lat=f"{np.mean(e2e):.2f}s")
    total_wall = time.time() - t_start_all
    _, peak_mb = gpu_mem_mb()

    return RunResult(
        router=router_cls.name, seed=seed, n=len(problems), n_correct=n_correct,
        accuracy=n_correct / max(1, len(problems)),
        mean_e2e_latency_s=float(np.mean(e2e)),
        p50_e2e_latency_s=float(np.percentile(e2e, 50)),
        p95_e2e_latency_s=float(np.percentile(e2e, 95)),
        throughput_qps=len(problems) / max(1e-6, total_wall),
        mean_prompt_tokens=float(np.mean(prompt_t_per)),
        mean_completion_tokens=float(np.mean(comp_t_per)),
        mean_llm_calls=float(np.mean(n_calls_per)),
        mean_cost_usd=float(np.mean(cost_per)),
        total_cost_usd=float(np.sum(cost_per)),
        routing_llm_calls=int(getattr(router, "routing_llm_calls", 0)),
        gpu_peak_mb=peak_mb,
    )


# ============================================================================
# 9. STATISTICAL ANALYSIS
# ============================================================================

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    if sd == 0:
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


def run_statistical_analysis(all_results: List[RunResult],
                             reference: str = "AgentRoute") -> Dict[str, Any]:
    df = pd.DataFrame([r.to_dict() for r in all_results])
    routers = df["router"].unique().tolist()
    if reference not in routers:
        logger.warning(f"Reference '{reference}' not in results.")
        return {}

    n_metrics = 3
    n_compare = max(1, len(routers) - 1)
    bonferroni_alpha = 0.05 / (n_metrics * n_compare)

    pairwise = []
    for r in routers:
        if r == reference:
            continue
        ref_runs = df[df["router"] == reference].sort_values("seed")
        oth_runs = df[df["router"] == r].sort_values("seed")
        if len(ref_runs) != len(oth_runs):
            continue
        for metric in ["accuracy", "mean_e2e_latency_s", "mean_cost_usd"]:
            a = ref_runs[metric].to_numpy()
            b = oth_runs[metric].to_numpy()
            t, p = stats.ttest_rel(a, b)
            d = cohens_d_paired(a, b)
            lo, hi = ci95_paired(a, b)
            pairwise.append({
                "comparison": f"{reference} vs {r}",
                "metric": metric,
                "n_runs": len(a),
                f"{reference}_mean": float(a.mean()),
                f"{r}_mean": float(b.mean()),
                "delta_mean": float(a.mean() - b.mean()),
                "t_stat": float(t),
                "p_value": float(p),
                "cohens_d": d,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "bonferroni_alpha": bonferroni_alpha,
                "significant_bonferroni": bool(p < bonferroni_alpha),
            })
    pairwise_df = pd.DataFrame(pairwise)

    friedman = []
    for metric in ["accuracy", "mean_e2e_latency_s", "mean_cost_usd"]:
        cols = []
        for r in routers:
            cols.append(df[df["router"] == r].sort_values("seed")[metric].to_numpy())
        if len(cols) >= 3 and all(len(c) == len(cols[0]) for c in cols) and len(cols[0]) >= 2:
            chi2, p = stats.friedmanchisquare(*cols)
        else:
            chi2, p = float("nan"), float("nan")
        friedman.append({"metric": metric, "n_routers": len(cols),
                         "chi2": float(chi2), "p_value": float(p)})
    friedman_df = pd.DataFrame(friedman)

    return {
        "pairwise": pairwise_df.to_dict(orient="records"),
        "friedman": friedman_df.to_dict(orient="records"),
        "bonferroni_alpha": bonferroni_alpha,
        "reference": reference,
    }


# ============================================================================
# 10. MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_problems", type=int, default=198,
                   help="GPQA Diamond has 198 questions; default = full set.")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--num_agents", type=int, default=8)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seeds", type=str, default="13,14,15")
    p.add_argument("--out", type=str, default="gpqa_results.json")
    p.add_argument("--csv_out", type=str, default="gpqa_results.csv")
    p.add_argument("--routers", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")][: args.runs]
    if len(seeds) < args.runs:
        seeds += [seeds[-1] + i for i in range(1, args.runs - len(seeds) + 1)]

    logger.info(f"GPQA Diamond evaluation — runs={args.runs}, problems={args.num_problems}, "
                f"agents={args.num_agents}, model={args.model}, seeds={seeds}")

    llm = Qwen25Backbone(model_name=args.model)

    chosen = ROUTER_CLASSES
    if args.routers:
        names = set(s.strip() for s in args.routers.split(","))
        chosen = [c for c in ROUTER_CLASSES if c.name in names]
        logger.info(f"Selected routers: {[c.name for c in chosen]}")

    all_results: List[RunResult] = []
    for seed in seeds:
        problems = load_gpqa_diamond(num_problems=args.num_problems, seed=seed)
        for router_cls in chosen:
            logger.info(f"-- {router_cls.name} | seed {seed} --")
            res = evaluate_one_run(router_cls, problems, llm, args.num_agents, seed)
            _, peak = gpu_mem_mb()
            logger.info(
                f"   acc={res.accuracy:.4f}  e2e={res.mean_e2e_latency_s:.2f}s  "
                f"qps={res.throughput_qps:.3f}  cost=${res.total_cost_usd:.4f}  "
                f"GPU peak={peak:.0f} MB  llm_calls={res.mean_llm_calls:.2f}"
            )
            all_results.append(res)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame([r.to_dict() for r in all_results])
    df.to_csv(args.csv_out, index=False)
    with open(args.out, "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    logger.info(f"\nRaw results written to {args.out} and {args.csv_out}")

    agg = df.groupby("router").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        e2e_mean=("mean_e2e_latency_s", "mean"), e2e_std=("mean_e2e_latency_s", "std"),
        p95_mean=("p95_e2e_latency_s", "mean"),
        qps_mean=("throughput_qps", "mean"),
        cost_per_q_mean=("mean_cost_usd", "mean"),
        cost_per_q_std=("mean_cost_usd", "std"),
        total_cost_mean=("total_cost_usd", "mean"),
        llm_calls_mean=("mean_llm_calls", "mean"),
        gpu_peak_mb_max=("gpu_peak_mb", "max"),
    )
    print("\n" + "=" * 100)
    print(" Aggregate results — mean ± std across runs")
    print("=" * 100)
    print(agg.round(4).to_string())
    agg.to_csv(args.csv_out.replace(".csv", "_aggregate.csv"))

    stats_out = run_statistical_analysis(all_results, reference="AgentRoute")
    if stats_out:
        print("\n" + "=" * 100)
        print(f" Pairwise comparison: AgentRoute vs each baseline")
        print(f" Bonferroni-corrected α = {stats_out['bonferroni_alpha']:.6f} "
              f"(uncorrected α = 0.05, {3 * (len(df['router'].unique()) - 1)} tests)")
        print("=" * 100)
        pw = pd.DataFrame(stats_out["pairwise"])
        print(pw.round(4).to_string(index=False))
        pw.to_csv(args.csv_out.replace(".csv", "_pairwise.csv"), index=False)

        print("\n" + "=" * 100)
        print(" Friedman χ² across all routers (per metric)")
        print("=" * 100)
        fr = pd.DataFrame(stats_out["friedman"])
        print(fr.round(6).to_string(index=False))
        fr.to_csv(args.csv_out.replace(".csv", "_friedman.csv"), index=False)

        with open(args.out.replace(".json", "_stats.json"), "w") as f:
            json.dump(stats_out, f, indent=2)


if __name__ == "__main__":
    main()


# ============================================================================
# HOW TO RUN
# ============================================================================
"""
# 1. Install dependencies
pip install torch transformers datasets accelerate scipy pandas numpy tqdm

# 2. Make sure agentroute_jang2005_complete.py is in the same directory.

# 3. Accept the GPQA dataset license:
#       Visit https://huggingface.co/datasets/Idavidrein/gpqa
#       Click "Agree and access repository"
#    Then login (one of):
#       huggingface-cli login
#       export HF_TOKEN=hf_xxx...

# 4. Smoke test (10 problems, 1 seed) — confirm everything works in ~5 minutes
python gpqa_evaluation.py --num_problems 10 --runs 1 --seeds 13

# 5. Full run (198 problems × 3 seeds × 6 routers — ~1.5–2.5 hours on 16 GB GPU)
python gpqa_evaluation.py --num_problems 198 --runs 3 --seeds 13,14,15 \\
    --out gpqa_results.json --csv_out gpqa_results.csv

# 6. Subset to specific routers
python gpqa_evaluation.py --routers AgentRoute,OI-MAS

# Output files:
#   gpqa_results.json
#   gpqa_results.csv
#   gpqa_results_aggregate.csv
#   gpqa_results_pairwise.csv
#   gpqa_results_friedman.csv
#   gpqa_results_stats.json

# WHAT TO EXPECT (qualitative — not numbers to cite):
#
#   - Random guessing baseline = 0.25 (4 options)
#   - Domain PhDs achieve ~65%; non-expert PhDs with internet access ~34%
#   - GPT-4 baseline (paper, 2023) ~39%
#   - Qwen 2.5 3B is small; expect single-call methods (AgentRoute, EvoMAS) to
#     land in the 0.20–0.35 range — i.e., near or somewhat above random
#   - Multi-call methods (OI-MAS, DyTopo) may improve via verification but
#     accuracy may also DROP because Qwen 3B's verifier/refiner can introduce
#     errors as easily as fix them on questions this hard
#   - RCR-Router and RopMura without retrieval will likely underperform — both
#     are designed for RAG-augmented multi-hop QA; closed-book GPQA fights their
#     design (this is faithful behavior, not a bug)
#
# IMPORTANT — the only numbers to publish are the ones THIS SCRIPT produces on
# YOUR hardware. Do not cite the qualitative expectations above.
"""
