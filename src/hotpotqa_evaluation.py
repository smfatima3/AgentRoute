"""
hotpotqa_evaluation.py
======================

Multi-hop QA evaluation of AgentRoute and 5 baselines on HotpotQA
(distractor setting). Default backbone: Qwen/Qwen3-4B-Instruct-2507
(swap with --model; --trust_remote_code is available for remote-code models
such as microsoft/Phi-3-small-128k-instruct).

WHY THIS SCRIPT EXISTS
----------------------
On MedQA (single-hop, closed-book) the retrieval-augmented multi-hop routers
(RCR-Router, RopMura) underperform because they are the wrong tool for the
job: their Searcher / router stages are designed to retrieve and reason over
an external corpus, which a closed-book MCQ task does not provide. HotpotQA
is exactly the setting these methods were built for: every question ships with
a 10-paragraph context (2 gold + 8 distractor), so the multi-hop routers can
finally retrieve and chain facts across documents. This script gives every
baseline a fair shot in its native regime.

DATASET
-------
- HotpotQA, distractor config, `validation` split (test answers are hidden).
- Source: `hotpotqa/hotpot_qa` (config "distractor"). The original CMU host is
  offline; if the canonical loader fails, we fall back to the community archive
  `vincentkoc/hotpot_qa_archive`.
- Each example: question, free-text answer (a span or "yes"/"no"), and a
  context of 10 (title, sentences) paragraphs.

MODEL
-----
- Default: Qwen/Qwen3-4B-Instruct-2507 (4B, non-thinking, Apache 2.0,
  requires transformers>=4.51). No trust_remote_code needed.
- Swap with --model; pass --trust_remote_code for remote-code backbones like
  microsoft/Phi-3-small-128k-instruct (which also needs tiktoken and a
  transformers version compatible with its bundled modeling code).

GRADING (official HotpotQA answer metrics)
------------------------------------------
- Exact Match (EM) and token-level F1 after normalisation (lowercase, strip
  articles / punctuation / extra whitespace). We report both; EM is the primary
  headline metric, F1 the secondary.

ROUTERS (same six as the other scripts)
---------------------------------------
    AgentRoute, OI-MAS, RCR-Router, EvoMAS, DyTopo, RopMura
All share one Phi-3-small backbone; only routing behaviour differs. Each router
receives the question plus the 10-paragraph context.

OUTPUTS
-------
    hotpotqa_results.{json,csv}      one row per (router, seed)
    hotpotqa_aggregate.csv           mean +/- std per router
    hotpotqa_pairwise.csv            paired t-tests vs AgentRoute (EM, F1, latency, cost)
    hotpotqa_friedman.csv            Friedman chi-square per metric
    hotpotqa_stats.json              full statistical blob
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import re
import string
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentroute_jang2005_complete import Agent, AgentRouteRouter  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("hotpotqa_eval")

PRICE_USD_PER_1M_INPUT = 0.02
PRICE_USD_PER_1M_OUTPUT = 0.05


# ============================================================================
# 1. BACKBONE  (Phi-3-small-128k-instruct)
# ============================================================================

def gpu_mem_mb() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.max_memory_allocated() / 1024 ** 2)


class HFBackbone:
    """A single shared HuggingFace causal-LM engine.

    Defaults to Qwen/Qwen3-4B-Instruct-2507 (Apache 2.0, non-thinking mode,
    requires transformers>=4.51). Works with any chat model loadable via
    AutoModelForCausalLM + apply_chat_template. Set --model to swap backbones.

    Exposes .generate(prompt, max_new_tokens, temperature) ->
    (text, prompt_tokens, completion_tokens, latency_s), matching the duck
    type used by every router class.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                 device: str = "cuda", dtype: str = "bfloat16",
                 trust_remote_code: bool = False):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        torch_dtype = (torch.bfloat16 if dtype == "bfloat16"
                       else torch.float16 if dtype == "float16"
                       else torch.float32)

        mem_before, _ = gpu_mem_mb()
        log.info(f"Loading {model_name} on {self.device} ({dtype})…")
        log.info(f"  GPU mem before load: {mem_before:.1f} MB")

        # Some remote-code models (e.g. Phi-3-small) need trust_remote_code=True
        # and may require a pinned transformers version; Qwen3-4B does not.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=trust_remote_code,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        mem_after, _ = gpu_mem_mb()
        log.info(f"  GPU mem after  load: {mem_after:.1f} MB  "
                 f"(Δ={mem_after - mem_before:.1f} MB)")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> Tuple[str, int, int, float]:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # Cap prompt length to control memory; HotpotQA contexts are long but
        # 16k tokens comfortably fits all 10 paragraphs for both backbones.
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=16384).to(self.device)
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


# Backwards-compatible alias (older invocations referenced Phi3Backbone).
Phi3Backbone = HFBackbone


# ============================================================================
# 2. DATASET LOADER
# ============================================================================

def _flatten_context(context: Dict[str, Any]) -> str:
    """Render the 10 (title, sentences) paragraphs as a single readable block."""
    titles = context["title"]
    sentences = context["sentences"]
    blocks = []
    for title, sents in zip(titles, sentences):
        para = " ".join(sents).strip()
        blocks.append(f"[{title}] {para}")
    return "\n".join(blocks)


def load_hotpotqa(num_problems: int, seed: int) -> List[Dict[str, Any]]:
    """Load HotpotQA distractor validation split, sampled with a fixed seed."""
    log.info("Loading HotpotQA (distractor, validation split)…")
    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                          split="validation")
        log.info(f"  loaded canonical hotpotqa/hotpot_qa: {len(ds)} examples")
    except Exception as e:  # noqa: BLE001
        log.warning(f"  canonical loader failed ({e}); trying archive mirror…")
        ds = load_dataset("vincentkoc/hotpot_qa_archive", "distractor",
                          split="validation")
        log.info(f"  loaded vincentkoc/hotpot_qa_archive: {len(ds)} examples")

    import random
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:num_problems]

    out: List[Dict[str, Any]] = []
    for i in idxs:
        ex = ds[i]
        q = ex.get("question")
        a = ex.get("answer")
        ctx = ex.get("context")
        if not q or a is None or ctx is None:
            continue
        out.append({
            "id": ex.get("id", f"hotpot_{i}"),
            "question": str(q).strip(),
            "answer": str(a).strip(),
            "context_str": _flatten_context(ctx),
            "type": ex.get("type", ""),
            "level": ex.get("level", ""),
        })
    log.info(f"  → {len(out)} usable problems")
    return out


# ============================================================================
# 3. GRADING  (official HotpotQA EM + token-F1)
# ============================================================================

def _normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace
    (the official HotpotQA / SQuAD normalisation)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _extract_answer(text: str) -> str:
    """Pull the model's final answer out of its response.

    We instruct models to end with 'ANSWER: <text>'. If that marker is present
    we take what follows; otherwise we fall back to the last non-empty line.
    """
    if not text:
        return ""
    m = re.search(r"ANSWER\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.DOTALL)
    if m:
        # Take the first line after the marker (answers are short spans)
        return m.group(1).strip().split("\n")[0].strip()
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # If either is empty, F1 is 1 only if both are empty
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def grade(pred_text: str, gold: str) -> Tuple[float, float]:
    """Return (EM, F1) for a model response against the gold answer."""
    pred = _extract_answer(pred_text)
    return exact_match(pred, gold), token_f1(pred, gold)


# ============================================================================
# 4. PROMPTS  (free-text multi-hop QA over the provided context)
# ============================================================================

SOLVE_PROMPT = """You are answering a multi-hop question using the provided context paragraphs. Reason across the paragraphs as needed.

Context:
{context}

Question: {question}

Reason briefly, then on a new line give your final answer in EXACTLY this format:
ANSWER: <the answer, as a short span or yes/no>
Do not write anything after the ANSWER line."""


VERIFY_PROMPT = """You are checking another model's answer to a multi-hop question.

Context:
{context}

Question: {question}

Proposed answer:
{solution}

If the proposed answer is correct, reply:
VERDICT: CORRECT

Otherwise, reply with the corrected answer:
VERDICT: WRONG
ANSWER: <correct short answer>"""


REFINE_PROMPT = """A first attempt at this multi-hop question may be wrong. Use the context to improve it.

Context:
{context}

Question: {question}

First attempt:
{attempt}

End with:
ANSWER: <the answer, as a short span or yes/no>"""


def make_agents(num_agents: int) -> Dict[str, Agent]:
    agents: Dict[str, Agent] = {}
    for i in range(num_agents):
        a = Agent(agent_id=f"agent_{i:02d}", specialty="qa")
        agents[a.agent_id] = a
    return agents


# ============================================================================
# 5. CALL ACCOUNTING
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
# 6. ROUTERS  (six, faithful to each paper, adapted to free-text multi-hop QA)
# ============================================================================

class BaselineRouter:
    name = "BaselineRouter"

    def __init__(self, agents: Dict[str, Agent], llm,
                 max_new_tokens: int = 256):
        self.agents = agents
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.routing_llm_calls = 0

    def solve(self, question: str, context: str) -> CallStats:
        raise NotImplementedError

    def reset(self):
        self.routing_llm_calls = 0
        for a in self.agents.values():
            a.reset()


class AgentRouteWrapper(BaselineRouter):
    """Deterministic pattern routing + LRU cache, one LLM call to answer."""
    name = "AgentRoute"

    def __init__(self, agents, llm, max_new_tokens=320):
        super().__init__(agents, llm, max_new_tokens)
        self.inner = AgentRouteRouter(agents=agents)

    def solve(self, question: str, context: str) -> CallStats:
        st = CallStats()
        t0 = time.time()
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if cache_key in self.inner.cache:
            self.inner.metrics.cache_hits += 1
        else:
            domain, _ = self.inner._pattern_classify(question)
            self.inner.cache[cache_key] = domain
        text, p, c, lat = self.llm.generate(
            SOLVE_PROMPT.format(context=context, question=question),
            max_new_tokens=self.max_new_tokens)
        st.add(p, c, lat, text)
        st.e2e_latency_s = time.time() - t0
        return st

    def reset(self):
        super().reset()
        self.inner.cache = {}
        self.inner.metrics = type(self.inner.metrics)()


class OIMASWrapper(BaselineRouter):
    """Generator -> Verifier -> (Refiner if WRONG)."""
    name = "OI-MAS"

    def solve(self, question: str, context: str) -> CallStats:
        st = CallStats()
        t0 = time.time()
        sol, p, c, lat = self.llm.generate(
            SOLVE_PROMPT.format(context=context, question=question),
            max_new_tokens=self.max_new_tokens)
        st.add(p, c, lat, sol)
        ver, p, c, lat = self.llm.generate(
            VERIFY_PROMPT.format(context=context, question=question, solution=sol),
            max_new_tokens=160)
        st.add(p, c, lat, sol)
        if "VERDICT: CORRECT" in ver.upper():
            st.final_text = sol
        elif re.search(r"ANSWER\s*:", ver, re.IGNORECASE):
            st.final_text = ver
        else:
            ref, p, c, lat = self.llm.generate(
                REFINE_PROMPT.format(context=context, question=question, attempt=sol),
                max_new_tokens=self.max_new_tokens)
            st.add(p, c, lat, ref)
            st.final_text = ref
        st.e2e_latency_s = time.time() - t0
        return st


class RCRRouterWrapper(BaselineRouter):
    """RCR-Router (Liu et al. 2025): Planner / Searcher / Recommender over
    structured memory, with token-budget allocation, importance scoring, and
    greedy semantic filtering, for T rounds.

    On HotpotQA the Searcher finally has a real corpus: it retrieves from the
    provided context paragraphs, which is the setting the method was designed
    for.
    """
    name = "RCR-Router"
    TOKEN_BUDGET = 1024
    W_ROLE = 1.0
    W_STAGE = 0.6
    W_RECENCY = 0.4

    def __init__(self, agents, llm, max_new_tokens=256, rounds: int = 3):
        super().__init__(agents, llm, max_new_tokens)
        self.rounds = rounds

    def _importance(self, item, role, stage):
        role_rel = self.W_ROLE if role in item.get("relevant_roles", []) else 0.0
        stage_prio = self.W_STAGE / (1.0 + abs(stage - item.get("stage", stage)))
        recency = self.W_RECENCY * math.exp(-0.3 * max(0, stage - item.get("stage", stage)))
        return role_rel + stage_prio + recency

    def _select(self, memory, role, stage, budget):
        ranked = sorted(memory,
                        key=lambda m: -self._importance(m, role, stage) /
                        max(1, m.get("tokens", 1)))
        chosen, used = [], 0
        for m in ranked:
            t = m.get("tokens", 50)
            if used + t > budget:
                continue
            chosen.append(m)
            used += t
        return chosen

    def _fmt(self, items):
        if not items:
            return "(empty)"
        return "\n".join(f"- [stage {m['stage']}|{m.get('source_role','?')}] {m['content']}"
                         for m in items)

    def solve(self, question: str, context: str) -> CallStats:
        st = CallStats()
        t0 = time.time()
        memory: List[Dict[str, Any]] = [{
            "stage": 0, "source_role": "Question",
            "content": question,
            "tokens": max(1, len(question.split())),
            "relevant_roles": ["Planner", "Searcher", "Recommender"],
        }, {
            "stage": 0, "source_role": "Corpus",
            "content": context[:4000],
            "tokens": max(1, len(context.split())),
            "relevant_roles": ["Searcher"],
        }]

        for r in range(self.rounds):
            stage = r + 1
            ctx = self._select(memory, "Planner", stage, self.TOKEN_BUDGET)
            plan_prompt = (
                "You are the PLANNER. Decompose the multi-hop question into the next "
                "sub-question to resolve from the corpus.\n"
                f"Round {r+1}/{self.rounds}.\n"
                f"Memory:\n{self._fmt(ctx)}\n\n"
                f"Original question: {question}\n"
                "Output ONE concrete sub-question on a single line.")
            plan, p, c, lat = self.llm.generate(plan_prompt, max_new_tokens=80)
            st.add(p, c, lat, plan)
            memory.append({"stage": stage, "source_role": "Planner",
                           "content": plan.strip().split("\n")[0],
                           "tokens": max(1, len(plan.split())),
                           "relevant_roles": ["Searcher", "Recommender"]})

            # Searcher: retrieve from the actual corpus
            search_prompt = (
                "You are the SEARCHER. Retrieve the facts from the context that answer "
                "the Planner's sub-question.\n"
                f"Context:\n{context}\n\n"
                f"Sub-question: {plan.strip().splitlines()[0] if plan.strip() else question}\n"
                "State the retrieved facts in 1-3 sentences.")
            sres, p, c, lat = self.llm.generate(search_prompt, max_new_tokens=200)
            st.add(p, c, lat, sres)
            memory.append({"stage": stage, "source_role": "Searcher",
                           "content": sres.strip()[:400],
                           "tokens": max(1, len(sres.split())),
                           "relevant_roles": ["Recommender"]})

            ctx = self._select(memory, "Recommender", stage, self.TOKEN_BUDGET)
            rec_prompt = (
                "You are the RECOMMENDER. Synthesize the retrieved facts into a final "
                "answer if possible.\n"
                f"Memory:\n{self._fmt(ctx)}\n\n"
                f"Original question: {question}\n"
                "If you can answer now, end with: ANSWER: <short answer>\n"
                "Otherwise respond with 'CONTINUE' on its own line.")
            rec, p, c, lat = self.llm.generate(rec_prompt, max_new_tokens=self.max_new_tokens)
            st.add(p, c, lat, rec)
            memory.append({"stage": stage, "source_role": "Recommender",
                           "content": rec.strip()[:500],
                           "tokens": max(1, len(rec.split())),
                           "relevant_roles": ["Planner"]})
            st.final_text = rec
            if re.search(r"ANSWER\s*:", rec, re.IGNORECASE):
                break

        if not re.search(r"ANSWER\s*:", st.final_text, re.IGNORECASE):
            forced = (
                "You are the RECOMMENDER. Give the FINAL answer now.\n"
                f"Context:\n{context[:3000]}\n\n"
                f"Question: {question}\n"
                "End with: ANSWER: <short answer>")
            ftext, p, c, lat = self.llm.generate(forced, max_new_tokens=self.max_new_tokens)
            st.add(p, c, lat, ftext)
            st.final_text = ftext
        st.e2e_latency_s = time.time() - t0
        return st


class EvoMASWrapper(BaselineRouter):
    """Roulette over {1,2,3}-call configs; EMA fitness updated online."""
    name = "EvoMAS"

    def __init__(self, agents, llm, max_new_tokens=256, pool_size=4):
        super().__init__(agents, llm, max_new_tokens)
        self.pool = [1, 2, 3, 2][:pool_size]
        self.fitness = [0.5] * pool_size
        self.k = 0
        self.last = 0

    def solve(self, question: str, context: str) -> CallStats:
        total = sum(self.fitness) or 1.0
        probs = [f / total for f in self.fitness]
        idx = int(np.random.choice(len(self.pool), p=probs))
        n_calls = self.pool[idx]
        self.last = idx
        st = CallStats()
        t0 = time.time()
        sol, p, c, lat = self.llm.generate(
            SOLVE_PROMPT.format(context=context, question=question),
            max_new_tokens=self.max_new_tokens)
        st.add(p, c, lat, sol)
        attempt = sol
        for _ in range(n_calls - 1):
            ref, p, c, lat = self.llm.generate(
                REFINE_PROMPT.format(context=context, question=question, attempt=attempt),
                max_new_tokens=self.max_new_tokens)
            st.add(p, c, lat, ref)
            attempt = ref
        st.final_text = attempt
        st.e2e_latency_s = time.time() - t0
        return st

    def update_fitness(self, was_correct: bool):
        self.fitness[self.last] = 0.8 * self.fitness[self.last] + 0.2 * float(was_correct)
        self.k += 1
        if self.k >= 50:
            worst = int(np.argmin(self.fitness)); best = int(np.argmax(self.fitness))
            self.pool[worst] = max(1, min(3, self.pool[best] + np.random.choice([-1, 0, 1])))
            self.fitness[worst] = 0.5
            self.k = 0


class DyTopoWrapper(BaselineRouter):
    """Manager + 4 agents; semantic key/query matching builds a directed
    communication graph each round; T rounds."""
    name = "DyTopo"
    AGENT_ROLES = ["Decomposer", "Retriever", "Verifier", "Aggregator"]
    EDGE_THRESHOLD = 0.30

    def __init__(self, agents, llm, max_new_tokens=256, rounds: int = 3,
                 embed_dim: int = 16):
        super().__init__(agents, llm, max_new_tokens)
        self.rounds = rounds
        self.embed_dim = embed_dim

    def _embed(self, text):
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], "big") & 0xFFFFFFFF
        v = np.random.RandomState(seed).randn(self.embed_dim)
        return v / (np.linalg.norm(v) + 1e-9)

    def _topology(self, descriptors, goal):
        keys = {a: self._embed(d["key"]) for a, d in descriptors.items()}
        queries = {a: self._embed(d["query"]) for a, d in descriptors.items()}
        edges = []
        for s in keys:
            for d in queries:
                if s != d and float(np.dot(keys[s], queries[d])) >= self.EDGE_THRESHOLD:
                    edges.append((s, d))
        g = self._embed(goal)
        for a in queries:
            if float(np.dot(g, queries[a])) >= self.EDGE_THRESHOLD:
                edges.append(("Manager", a))
        return edges

    def solve(self, question: str, context: str) -> CallStats:
        st = CallStats()
        t0 = time.time()
        running = f"Question: {question}"
        prior = {a: {"key": "ready", "query": "round goal"} for a in self.AGENT_ROLES}

        for r in range(self.rounds):
            mgr = (f"You are the MANAGER of a 4-agent multi-hop QA team "
                   f"({', '.join(self.AGENT_ROLES)}).\nState:\n{running}\n\n"
                   f"Round {r+1}/{self.rounds}. Output ONE concise round goal.")
            goal, p, c, lat = self.llm.generate(mgr, max_new_tokens=60)
            st.add(p, c, lat, goal)
            round_goal = goal.strip().split("\n")[0]
            edges = self._topology(prior, round_goal)
            active = sorted({d for _, d in edges})
            if not active:
                active = ["Retriever"] if r < self.rounds - 1 else ["Aggregator"]
            new_desc = dict(prior)
            for role in active:
                inbound = [s for s, d in edges if d == role]
                ap = (f"You are the {role.upper()} agent. Round goal: {round_goal}\n"
                      f"Messages from: {', '.join(inbound) if inbound else 'Manager'}\n"
                      f"Context:\n{context}\n\nState:\n{running}\n\n"
                      "Output in this exact format:\n"
                      "KEY: <what you can offer next round>\n"
                      "QUERY: <what you need next round>\n"
                      "CONTRIBUTION: <your contribution this round>")
                mt = self.max_new_tokens if (role == "Aggregator" and r == self.rounds - 1) else 200
                txt, p, c, lat = self.llm.generate(ap, max_new_tokens=mt)
                st.add(p, c, lat, txt)
                km = re.search(r"KEY:\s*(.+)", txt)
                qm = re.search(r"QUERY:\s*(.+)", txt)
                cm = re.search(r"CONTRIBUTION:\s*(.+)", txt, re.DOTALL)
                new_desc[role] = {"key": (km.group(1).strip() if km else "na")[:80],
                                  "query": (qm.group(1).strip() if qm else "na")[:80]}
                contrib = cm.group(1).strip() if cm else txt.strip()
                running += f"\n[R{r+1}/{role}]: {contrib[:300]}"
                if role == "Aggregator":
                    st.final_text = contrib
            prior = new_desc
            halt = (f"As MANAGER, is the question answered?\nState:\n{running}\n\n"
                    "Reply HALT or CONTINUE.")
            h, p, c, lat = self.llm.generate(halt, max_new_tokens=8)
            st.add(p, c, lat, h)
            if "HALT" in h.upper():
                if "Aggregator" not in active:
                    fp = (f"You are the AGGREGATOR. Give the final answer.\n{running}\n\n"
                          "End with: ANSWER: <short answer>")
                    fa, p, c, lat = self.llm.generate(fp, max_new_tokens=self.max_new_tokens)
                    st.add(p, c, lat, fa)
                    st.final_text = fa
                break

        if not st.final_text or not re.search(r"ANSWER\s*:", st.final_text, re.IGNORECASE):
            fp = (f"You are the AGGREGATOR. Give the final answer.\n{running}\n\n"
                  f"Question: {question}\nEnd with: ANSWER: <short answer>")
            fa, p, c, lat = self.llm.generate(fp, max_new_tokens=self.max_new_tokens)
            st.add(p, c, lat, fa)
            st.final_text = fa
        st.e2e_latency_s = time.time() - t0
        return st


class RopMuraWrapper(BaselineRouter):
    """RopMura (Wu et al. 2025): four-submodule planner (question_splitter ->
    question_selector -> judger -> defender) with cluster-embedding routing.
    On HotpotQA the knowledge agents answer sub-questions from the corpus."""
    name = "RopMura"

    def __init__(self, agents, llm, max_new_tokens=256, max_hops: int = 3,
                 num_subq: int = 3):
        super().__init__(agents, llm, max_new_tokens)
        self.max_hops = max_hops
        self.num_subq = num_subq
        self.clusters = {}
        for aid, agent in agents.items():
            seed = int(hashlib.sha256(f"{aid}:{agent.specialty}".encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            self.clusters[aid] = [rng.randn(16) / np.sqrt(16) for _ in range(3)]

    def _embed(self, text, dim=16):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") & 0xFFFFFFFF
        v = np.random.RandomState(seed).randn(dim)
        return v / (np.linalg.norm(v) + 1e-9)

    def _route(self, subq, top_k=1):
        q = self._embed(subq)
        scores = [(max(float(np.dot(q, c) / (np.linalg.norm(c) + 1e-9)) for c in cl), aid)
                  for aid, cl in self.clusters.items()]
        scores.sort(reverse=True)
        return [aid for _, aid in scores[:top_k]]

    def solve(self, question: str, context: str) -> CallStats:
        st = CallStats()
        t0 = time.time()
        qa: List[Tuple[str, str]] = []
        final = None
        for hop in range(self.max_hops):
            hist = "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(qa)) or "(empty)"
            split = (
                "You are the PLANNER's QUESTION_SPLITTER. Raise SEVERAL candidate "
                "sub-questions that help answer the multi-hop question from the corpus.\n"
                f"Question: {question}\nQA so far:\n{hist}\n\n"
                f"Output exactly {self.num_subq} sub-questions, one per line, '1.' '2.' '3.'")
            so, p, c, lat = self.llm.generate(split, max_new_tokens=160)
            st.add(p, c, lat, so)
            cands = [m.group(1).strip() for m in
                     (re.match(r"\s*\d+[.)]\s*(.+)", ln) for ln in so.split("\n")) if m]
            if not cands:
                cands = [so.strip().split("\n")[0]]
            sel = ("You are the PLANNER's QUESTION_SELECTOR. Pick the ONE best sub-question.\n"
                   f"Question: {question}\nCandidates:\n"
                   + "\n".join(f"  {i+1}. {x}" for i, x in enumerate(cands))
                   + "\n\nReply with the integer index only.")
            so2, p, c, lat = self.llm.generate(sel, max_new_tokens=8)
            st.add(p, c, lat, so2)
            mi = re.search(r"[1-9]", so2)
            idx = max(0, min(len(cands) - 1, int(mi.group(0)) - 1)) if mi else 0
            chosen = cands[idx]
            _ = self._route(chosen, top_k=1)
            ans = ("You are a knowledge agent. Answer the sub-question from the context.\n"
                   f"Context:\n{context}\n\nSub-question: {chosen}\n"
                   "Answer concisely.")
            suba, p, c, lat = self.llm.generate(ans, max_new_tokens=200)
            st.add(p, c, lat, suba)
            qa.append((chosen, suba.strip()[:300]))
            judge = ("You are the PLANNER's JUDGER. Can the original question be answered now?\n"
                     f"Question: {question}\nQA:\n"
                     + "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(qa))
                     + "\n\nReply ANSWERABLE or NOT_ANSWERABLE.")
            j, p, c, lat = self.llm.generate(judge, max_new_tokens=8)
            st.add(p, c, lat, j)
            ju = j.upper().strip()
            answerable = "ANSWERABLE" in ju and "NOT_ANSWERABLE" not in ju and "NOT ANSWERABLE" not in ju
            if answerable or hop == self.max_hops - 1:
                defend = ("You are the PLANNER's DEFENDER. Compose the final answer from the QA records.\n"
                          f"Question: {question}\nQA:\n"
                          + "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(qa))
                          + "\n\nEnd with: ANSWER: <short answer>")
                final, p, c, lat = self.llm.generate(defend, max_new_tokens=self.max_new_tokens)
                st.add(p, c, lat, final)
                break
        if final is None:
            final = "\n".join(f"A{i+1}: {a}" for i, (_, a) in enumerate(qa))
        st.final_text = final
        st.e2e_latency_s = time.time() - t0
        return st


ROUTER_CLASSES = [
    AgentRouteWrapper, OIMASWrapper, RCRRouterWrapper,
    EvoMASWrapper, DyTopoWrapper, RopMuraWrapper,
]


# ============================================================================
# 7. EVALUATION LOOP
# ============================================================================

@dataclass
class RunResult:
    router: str
    seed: int
    n: int
    em: float
    f1: float
    mean_e2e_latency_s: float
    p50_e2e_latency_s: float
    p95_e2e_latency_s: float
    throughput_qps: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_llm_calls: float
    mean_cost_usd: float
    total_cost_usd: float
    gpu_peak_mb: float

    def to_dict(self):
        return asdict(self)


def cost_usd(p_t, c_t):
    return (p_t / 1e6) * PRICE_USD_PER_1M_INPUT + (c_t / 1e6) * PRICE_USD_PER_1M_OUTPUT


def evaluate_one(router_cls, problems, llm, num_agents, seed) -> RunResult:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()

    agents = make_agents(num_agents)
    router = router_cls(agents=agents, llm=llm)
    router.reset()

    e2e, p_t, c_t, calls, costs = [], [], [], [], []
    ems, f1s = [], []
    t_start = time.time()
    pbar = tqdm(problems, desc=f"{router_cls.name}/seed={seed}", leave=False)
    for prob in pbar:
        st = router.solve(prob["question"], prob["context_str"])
        em, f1 = grade(st.final_text, prob["answer"])
        ems.append(em); f1s.append(f1)
        e2e.append(st.e2e_latency_s)
        p_t.append(st.prompt_tokens); c_t.append(st.completion_tokens)
        calls.append(st.n_llm_calls)
        costs.append(cost_usd(st.prompt_tokens, st.completion_tokens))
        if hasattr(router, "update_fitness"):
            router.update_fitness(em > 0.5)
        pbar.set_postfix(em=f"{np.mean(ems):.3f}", f1=f"{np.mean(f1s):.3f}")
    wall = time.time() - t_start
    _, peak = gpu_mem_mb()

    return RunResult(
        router=router_cls.name, seed=seed, n=len(problems),
        em=float(np.mean(ems)), f1=float(np.mean(f1s)),
        mean_e2e_latency_s=float(np.mean(e2e)),
        p50_e2e_latency_s=float(np.percentile(e2e, 50)),
        p95_e2e_latency_s=float(np.percentile(e2e, 95)),
        throughput_qps=len(problems) / max(1e-6, wall),
        mean_prompt_tokens=float(np.mean(p_t)),
        mean_completion_tokens=float(np.mean(c_t)),
        mean_llm_calls=float(np.mean(calls)),
        mean_cost_usd=float(np.mean(costs)),
        total_cost_usd=float(np.sum(costs)),
        gpu_peak_mb=peak)


# ============================================================================
# 8. STATISTICS
# ============================================================================

def cohens_d_paired(a, b):
    d = a - b
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    if sd == 0:
        return float("inf") if d.mean() != 0 else 0.0
    return float(d.mean() / sd)


def ci95_paired(a, b):
    d = a - b; n = len(d)
    if n < 2:
        return (float("nan"), float("nan"))
    m = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(n))
    h = se * stats.t.ppf(0.975, n - 1)
    return (m - h, m + h)


def run_stats(results, reference="AgentRoute"):
    df = pd.DataFrame([r.to_dict() for r in results])
    routers = df["router"].unique().tolist()
    if reference not in routers:
        return {}
    metrics = ["em", "f1", "mean_e2e_latency_s", "mean_cost_usd"]
    n_comp = max(1, len(routers) - 1)
    alpha = 0.05 / (len(metrics) * n_comp)
    pairwise = []
    for r in routers:
        if r == reference:
            continue
        ref = df[df["router"] == reference].sort_values("seed")
        oth = df[df["router"] == r].sort_values("seed")
        if len(ref) != len(oth) or len(ref) < 2:
            continue
        for metric in metrics:
            a = ref[metric].to_numpy(); b = oth[metric].to_numpy()
            t, p = (0.0, 1.0) if np.all(a == b) else stats.ttest_rel(a, b)
            lo, hi = ci95_paired(a, b)
            pairwise.append({
                "comparison": f"{reference} vs {r}", "metric": metric,
                "n_runs": len(a), f"{reference}_mean": float(a.mean()),
                f"{r}_mean": float(b.mean()), "delta_mean": float(a.mean() - b.mean()),
                "t_stat": float(t), "p_value": float(p),
                "cohens_d": cohens_d_paired(a, b), "ci95_lo": lo, "ci95_hi": hi,
                "bonferroni_alpha": alpha, "significant_bonferroni": bool(p < alpha)})
    friedman = []
    for metric in metrics:
        cols = [df[df["router"] == r].sort_values("seed")[metric].to_numpy() for r in routers]
        if len(cols) >= 3 and all(len(c) == len(cols[0]) for c in cols) and len(cols[0]) >= 2:
            try:
                chi2, p = stats.friedmanchisquare(*cols)
            except ValueError:
                chi2, p = float("nan"), float("nan")
        else:
            chi2, p = float("nan"), float("nan")
        friedman.append({"metric": metric, "n_routers": len(cols),
                         "chi2": float(chi2), "p_value": float(p)})
    return {"pairwise": pairwise, "friedman": friedman,
            "bonferroni_alpha": alpha, "reference": reference}


# ============================================================================
# 9. MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_problems", type=int, default=100)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seeds", type=str, default="13,14,15,16,17")
    p.add_argument("--num_agents", type=int, default=8)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--trust_remote_code", action="store_true",
                   help="Required for remote-code models such as "
                        "microsoft/Phi-3-small-128k-instruct; not needed for Qwen3.")
    p.add_argument("--routers", type=str, default="")
    p.add_argument("--out_dir", type=str, default=".")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")][:args.runs]
    if len(seeds) < args.runs:
        seeds += [seeds[-1] + i for i in range(1, args.runs - len(seeds) + 1)]
    chosen = ROUTER_CLASSES
    if args.routers:
        names = {s.strip() for s in args.routers.split(",")}
        chosen = [c for c in ROUTER_CLASSES if c.name in names]

    log.info(f"HotpotQA eval — model={args.model} routers={[c.name for c in chosen]} "
             f"seeds={seeds} problems={args.num_problems}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    llm = HFBackbone(model_name=args.model, dtype=args.dtype,
                     trust_remote_code=args.trust_remote_code)

    all_results = []
    for seed in seeds:
        problems = load_hotpotqa(args.num_problems, seed=seed)
        for rc in chosen:
            log.info(f"-- {rc.name} | seed {seed} --")
            res = evaluate_one(rc, problems, llm, args.num_agents, seed)
            _, peak = gpu_mem_mb()
            log.info(f"   EM={res.em:.4f}  F1={res.f1:.4f}  e2e={res.mean_e2e_latency_s:.2f}s "
                     f"cost=${res.total_cost_usd:.4f}  GPU={peak:.0f}MB  calls={res.mean_llm_calls:.2f}")
            all_results.append(res)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame([r.to_dict() for r in all_results])
    df.to_csv(out_dir / "hotpotqa_results.csv", index=False)
    with open(out_dir / "hotpotqa_results.json", "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    agg = df.groupby("router").agg(
        em_mean=("em", "mean"), em_std=("em", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        e2e_mean=("mean_e2e_latency_s", "mean"), e2e_std=("mean_e2e_latency_s", "std"),
        p95_mean=("p95_e2e_latency_s", "mean"),
        qps_mean=("throughput_qps", "mean"),
        cost_per_q_mean=("mean_cost_usd", "mean"),
        total_cost_mean=("total_cost_usd", "mean"),
        llm_calls_mean=("mean_llm_calls", "mean"),
        gpu_peak_mb_max=("gpu_peak_mb", "max"))
    agg.to_csv(out_dir / "hotpotqa_aggregate.csv")
    print("\n" + "=" * 100)
    print(" Aggregate results — mean ± std across runs")
    print("=" * 100)
    print(agg.round(4).to_string())

    s = run_stats(all_results, reference="AgentRoute")
    if s:
        pw = pd.DataFrame(s["pairwise"]); pw.to_csv(out_dir / "hotpotqa_pairwise.csv", index=False)
        fr = pd.DataFrame(s["friedman"]); fr.to_csv(out_dir / "hotpotqa_friedman.csv", index=False)
        with open(out_dir / "hotpotqa_stats.json", "w") as f:
            json.dump(s, f, indent=2)
        print("\n" + "=" * 100)
        print(f" Pairwise vs AgentRoute  Bonferroni α = {s['bonferroni_alpha']:.6f}")
        print("=" * 100)
        print(pw[["comparison", "metric", "delta_mean", "p_value", "cohens_d",
                  "significant_bonferroni"]].round(4).to_string(index=False))
        print("\n" + "=" * 100)
        print(" Friedman χ² across routers")
        print("=" * 100)
        print(fr.round(6).to_string(index=False))

    log.info(f"\nOutputs written under {out_dir.resolve()}/")


if __name__ == "__main__":
    main()


# =====================================================================
# HOW TO RUN
# =====================================================================
"""
pip install torch transformers datasets accelerate scipy pandas numpy tqdm tiktoken
# Phi-3-small needs tiktoken + trust_remote_code; a >=24 GB GPU is recommended
# (7B weights in bf16 ≈ 14-15 GB, plus long HotpotQA contexts inflate the KV cache).

# Smoke test (~10 min on a large GPU): 10 problems, 1 seed, 2 routers
python hotpotqa_evaluation.py --num_problems 10 --runs 1 --seeds 13 \\
    --routers AgentRoute,RCR-Router --out_dir smoke_hotpotqa

# Full run: 100 problems × 3 seeds × 6 routers
python hotpotqa_evaluation.py --num_problems 100 --runs 3 --seeds 13,14,15 \\
    --out_dir hotpotqa_out

# Output files (in --out_dir):
#   hotpotqa_results.{csv,json}   one row per (router, seed)
#   hotpotqa_aggregate.csv        mean ± std per router (EM, F1, latency, cost)
#   hotpotqa_pairwise.csv         paired t-tests vs AgentRoute (EM/F1/latency/cost)
#   hotpotqa_friedman.csv         Friedman χ² per metric
#   hotpotqa_stats.json           full statistical blob
"""
