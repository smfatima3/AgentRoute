"""Answer-health observations. Never repairs, rescoring, or label-dependent prompts."""
import re
from .evaluation import final_answer


def answer_health(text, ledger):
    final=final_answer(text)
    return {'literal_placeholder':bool(re.search(r'<\s*(?:answer|number)\s*>',text,re.I)),
            'final_marker_present':bool(re.search(r'^\s*(?:FINAL(?: ANSWER)?|ANSWER)\s*:',text,re.M|re.I)),
            'parsed_final':final,
            'truncated_calls':sum(c.get('finish_reason')=='length' for c in ledger),
            'router_errors':sum(c.get('category')=='router' and c.get('status')!='ok' for c in ledger)}
