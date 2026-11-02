import re
import json
from typing import List, Dict, Any
from pathlib import Path

# ---------------------------------------------------------------------------
# Utility helpers for extracting, normalising and comparing answers in the
# MATH and MATH‑500 datasets.  This version has been *hardened* to cope with
# the quirky LaTeX routinely produced by LLMs (e.g. missing back‑slashes in
# \text{}, stray \left/\right, or \boxed{} spelt without the back‑slash).
# ---------------------------------------------------------------------------

# 1. Answer extraction helpers ------------------------------------------------

def _strip_tex_wrappers(expr: str) -> str:
    """Remove common LaTeX wrappers such as \text{}, text{}, \mathrm{} …"""
    # Handle both escaped and un‑escaped forms, e.g. \text{foo} or text{foo}
    expr = re.sub(r"(?:\\)?(?:text|mathrm|mathbf|mathit|mathsf|mathcal|mathbb)\{([^}]*)\}", r"\1", expr,
                  flags=re.IGNORECASE)
    return expr


def clean_math_answer(answer: str) -> str:
    """Extract the *final* boxed answer from a worked *official* solution."""
    # Accept both \boxed{…} and boxed{…}
    boxed_pattern = r"\\?boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(boxed_pattern, answer)
    if matches:
        final_answer = matches[-1]
        final_answer = final_answer.replace("\\", "")  # strip any remaining back‑slashes
        final_answer = final_answer.replace("$", "").replace(",", "").strip()
        final_answer = _strip_tex_wrappers(final_answer)
        return final_answer

    # Look for an explicit "answer is …" phrase.
    ans_ptrn = r"answer\s+is\s+([^.]+)"
    match = re.search(ans_ptrn, answer.lower())
    if match:
        return _strip_tex_wrappers(match.group(1).strip())

    # Fallback: last numeric token in the text.
    numbers = re.findall(r"-?\d+\.?\d*", answer)
    if numbers:
        return numbers[-1]

    return answer.strip()


# 2. Model prediction parsing -------------------------------------------------

def math_get_predict(text: str) -> str:
    """Extract the answer _the model_ claims – usually inside a \boxed{}."""
    if not text:
        return ""

    boxed_pattern = r"\\?boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(boxed_pattern, text)
    if matches:
        ans = matches[-1]
        # Normalise a few ubiquitous commands **before** further processing.
        ans = (ans.replace("\\frac", "frac")
                  .replace("\\sqrt", "sqrt")
                  .replace("\\pi", "pi")
                  .replace("\\infty", "infinity")
                  .replace("\\", ""))
        ans = _strip_tex_wrappers(ans).strip()
        return ans

    # Common fall‑back phrases
    for ptrn in [r"[Tt]he answer is:?\s*([^.\n]+)",
                 r"[Ff]inal answer:?\s*([^.\n]+)",
                 r"[Aa]nswer:?\s*([^.\n]+)"]:
        m = re.search(ptrn, text)
        if m:
            return _strip_tex_wrappers(re.sub(r"[.,;!?]+$", "", m.group(1).strip()))

    # Scan lines backwards, looking for an equation or lone number.
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if not line or line.startswith(("Note:", "Therefore")):
            continue
        if any(ch in line for ch in "=+-*/^fracsqrt"):
            if "=" in line:
                return _strip_tex_wrappers(line.split("=")[-1].strip())
        numbers = re.findall(r"-?\d+\.?\d*", line)
        if numbers:
            return numbers[-1]
    return ""


# 3. Normalisation ------------------------------------------------------------

def normalize_math_answer(ans: str) -> str:
    """Convert different LaTeX / textual representations to a canonical form."""
    if not ans:
        return ""

    # Strip \boxed if user left it in.
    ans = re.sub(r"\\?boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", ans)

    ans = ans.strip()
    ans = _strip_tex_wrappers(ans)

    # Exhaustive replacement table (extended with un‑escaped variants).
    repl = {
        # Fraction variants
        r"\\dfrac": "\\frac", r"\\tfrac": "\\frac", r"\\cfrac": "\\frac",
        # Display style
        r"\\displaystyle": "",
        # Degree symbols and text
        r"\^\{\\circ\}": "°", r"\\circ": "°", r"\\degree": "°", r"degrees?": "°",
        # Operators
        r"\\pm": "±", r"\\mp": "∓", r"\\times": "*", r"\\cdot": "*", r"\\div": "/",
        r"\\leq?": "<=", r"\\geq?": ">=", r"\\lt": "<", r"\\gt": ">", r"\\neq|\\ne": "!=",
        # Functions
        r"\\sqrt": "sqrt", r"\\log": "log", r"\\ln": "ln", r"\\sin": "sin", r"\\cos": "cos",
        r"\\tan": "tan", r"\\cot": "cot", r"\\sec": "sec", r"\\csc": "csc",
        r"\\arcsin": "arcsin", r"\\arccos": "arccos", r"\\arctan": "arctan",
        # Greek letters / constants
        r"\\pi": "π", r"\\infty": "∞",
        # Bracket sizing – include un‑escaped variants
        r"\\left\(": "(", r"\\right\)": ")", r"left\(": "(", r"right\)": ")",
        r"\\left\[": "[", r"\\right\]": "]", r"left\[": "[", r"right\]": "]",
        r"\\left\\{": "{", r"\\right\\}": "}", r"left\\{": "{", r"right\\}": "}",
        r"\\left\|": "|", r"\\right\|": "|", r"left\|": "|", r"right\|": "|",
    }
    for p, r_ in repl.items():
        ans = re.sub(p, r_, ans, flags=re.IGNORECASE)

    # Handle \frac{a}{b} → (a)/(b) and likewise for bare 'frac'.
    for frac_pat in [r"\\frac\{([^}]+)\}\{([^}]+)\}", r"frac\{([^}]+)\}\{([^}]+)\}"]:
        while re.search(frac_pat, ans):
            ans = re.sub(frac_pat, r"(\1)/(\2)", ans)

    # Strip any *remaining* back‑slashes then normalise whitespace.
    ans = ans.replace("\\", "")
    ans = re.sub(r"\s+", " ", ans).strip()

    # Compact forms like "90 °" → "90°"
    ans = re.sub(r"(\d+)\s*°", r"\1°", ans)
    return ans


# ---------------------------------------------------------------------------
# The comparison logic (unchanged except that it calls the new normaliser).
# ---------------------------------------------------------------------------

def compare_numerical_values(val1: str, val2: str, tol: float = 1e-9) -> bool:
    try:
        return abs(float(val1) - float(val2)) < tol
    except Exception:
        try:
            num1 = complex(val1.replace("i", "j"))
            num2 = complex(val2.replace("i", "j"))
            return abs(num1 - num2) < tol
        except Exception:
            return False


def compare_fractions(frac1: str, frac2: str) -> bool:
    try:
        m1 = re.match(r"\(([^)]+)\)/\(([^)]+)\)", frac1)
        m2 = re.match(r"\(([^)]+)\)/\(([^)]+)\)", frac2)
        if m1 and m2:
            v1 = float(m1.group(1)) / float(m1.group(2))
            v2 = float(m2.group(1)) / float(m2.group(2))
            return abs(v1 - v2) < 1e-9
        if '/' in frac1 and '/' in frac2:
            n1, d1 = map(float, frac1.split('/'))
            n2, d2 = map(float, frac2.split('/'))
            return abs(n1 / d1 - n2 / d2) < 1e-9
    except Exception:
        pass
    return False


def compare_ordered_pairs(p1: str, p2: str) -> bool:
    try:
        m1, m2 = re.search(r"\(([^)]+)\)", p1), re.search(r"\(([^)]+)\)", p2)
        if m1 and m2:
            vals1 = [v.strip() for v in m1.group(1).split(',')]
            vals2 = [v.strip() for v in m2.group(1).split(',')]
            if len(vals1) != len(vals2):
                return False
            return all(compare_math_answers(a, b) for a, b in zip(vals1, vals2))
    except Exception:
        pass
    return False


def compare_math_answers(pred: str, target: str) -> bool:
    if not pred or not target:
        return False

    if pred.strip() == target.strip():
        return True

    p_norm, t_norm = normalize_math_answer(pred), normalize_math_answer(target)
    if p_norm == t_norm or p_norm.lower() == t_norm.lower():
        return True

    # Aggressive de‑punctuated textual compare (e.g. 'evelyn' vs 'text evelyn')
    if re.sub(r"[^\w]", "", p_norm.lower()) == re.sub(r"[^\w]", "", t_norm.lower()):
        return True

    # Degrees
    if '°' in p_norm or '°' in t_norm:
        return compare_numerical_values(p_norm.replace('°', ''), t_norm.replace('°', ''))

    # Numerics
    if compare_numerical_values(p_norm, t_norm):
        return True

    # Fractions
    if ('/' in p_norm or 'frac' in p_norm) and ('/' in t_norm or 'frac' in t_norm):
        if compare_fractions(p_norm, t_norm):
            return True

    # Ordered pairs / tuples
    if '(' in p_norm and '(' in t_norm:
        if compare_ordered_pairs(p_norm, t_norm):
            return True

    # Pi handling
    if 'π' in p_norm or 'π' in t_norm:
        if p_norm.replace('π', 'pi') == t_norm.replace('π', 'pi'):
            return True

    # sqrt normalisation (e.g. '3sqrt13')
    p_s = re.sub(r"(\d+)\s*sqrt\s*\(?\s*(\d+)\s*\)?", r"\1*sqrt(\2)", p_norm)
    t_s = re.sub(r"(\d+)\s*sqrt\s*\(?\s*(\d+)\s*\)?", r"\1*sqrt(\2)", t_norm)
    if p_s == t_s:
        return True

    # Evaluate simple arithmetic expressions
    if re.fullmatch(r"[\d\.\+\-*/()\s]+", p_norm) and re.fullmatch(r"[\d\.\+\-*/()\s]+", t_norm):
        try:
            return abs(eval(p_norm) - eval(t_norm)) < 1e-9
        except Exception:
            pass

    # Mixed numbers (e.g. '1 1/2')
    mix = r"(\d+)\s+(\d+)/(\d+)"
    pm, tm = re.match(mix, p_norm), re.match(mix, t_norm)
    if pm or tm:
        try:
            pv = float(pm.group(1)) + float(pm.group(2))/float(pm.group(3)) if pm else eval(p_norm)
            tv = float(tm.group(1)) + float(tm.group(2))/float(tm.group(3)) if tm else eval(t_norm)
            return abs(pv - tv) < 1e-9
        except Exception:
            pass
    return False


# 4. Dataset loader -----------------------------------------------------------

def math_data_process(data_path: str) -> List[Dict[str, Any]]:
    """Load and harmonise either the original MATH or MATH‑500 JSONL."""
    processed = []
    with open(data_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            if 'problem' in item and 'answer' in item:  # MATH‑500
                processed.append({
                    'task': item['problem'],
                    'answer': item['answer'],
                    'level': item.get('level', 'Unknown'),
                    'type': item.get('subject', 'Unknown'),
                    'original_solution': item.get('solution', ''),
                    'unique_id': item.get('unique_id', '')
                })
            elif 'problem' in item and 'solution' in item:  # Original MATH
                processed.append({
                    'task': item['problem'],
                    'answer': clean_math_answer(item['solution']),
                    'level': item.get('level', 'Unknown'),
                    'type': item.get('type', 'Unknown'),
                    'original_solution': item['solution']
                })
            else:  # Legacy/other
                processed.append({
                    'task': item.get('task', item.get('problem', '')),
                    'answer': clean_math_answer(item.get('solution', '')),
                    'level': item.get('level', 'Unknown'),
                    'type': item.get('type', 'Unknown'),
                    'original_solution': item.get('solution', '')
                })
    return processed


# ---------------------------------------------------------------------------
# Quick self‑test ------------------------------------------------------------
# Run `python math_dataset.py` to sanity‑check the normalisation / comparison.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running quick self‑check …")
    cases = [
        ("text{Evelyn}", "Evelyn", True),
        ("dfrac{14}{3}", "\\frac{14}{3}", True),
        ("(3, π/2)", "\\left( 3, \\frac{\\pi}{2} \right)", True),
        ("90°", "90^{\\circ}", True),
        ("42", "43", False)
    ]
    for pred, true, exp in cases:
        res = compare_math_answers(pred, true)
        print(f" {pred!s:<20} vs {true:<30} → {res} (expected {exp})")
