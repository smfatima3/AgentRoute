"""Gold-free answer parsing for communication and voting; no symbolic execution."""
import re
from .evaluation import final_answer


def answer_key(text, metric='exact'):
    value=final_answer(text)
    if value is None or not value.strip() or len(value)>4096: return None
    value=value.strip()
    if metric=='mcq':
        m=re.fullmatch(r'\(?([A-D])\)?[.]?',value,re.I)
        return m.group(1).upper() if m else None
    if metric=='math':
        if value.startswith('\\boxed{') and value.endswith('}'): value=value[7:-1]
        if value.startswith('$') and value.endswith('$'): value=value.strip('$')
        value=value.replace('\\left','').replace('\\right','').replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
        return re.sub(r'\s+','',value) or None
    # Do not erase meaningful BBH parentheses, brackets, signs or word order.
    return ' '.join(value.split()).casefold()
