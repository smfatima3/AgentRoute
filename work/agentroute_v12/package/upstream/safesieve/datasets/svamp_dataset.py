import json
import re
import math

def _strip_string(string):
    # Reuse gsm8k's cleaning logic
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\text{ ", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = string.replace(" ", "")
    return string

def delete_extra_zero(n):
    try:
        n = float(n)
    except:
        return n
    if n == int(n):
        return str(int(n))
    else:
        return str(n).rstrip('0').rstrip('.') if '.' in str(n) else str(n)

def svamp_get_predict(pred_str):
    # Prediction string extraction, similar to gsm8k
    if ('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif ('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    else:
        pattern = r'-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if (len(pred) >= 1):
            pred = pred[-1]
        else:
            pred = ''
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    pred = _strip_string(pred)
    return delete_extra_zero(pred)

def svamp_data_process(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            q = item['question']
            ans = item['answer']
            if '\n####' in ans:
                expr, final = ans.split('\n####')
                expr = expr.strip()
                final = final.strip()
            else:
                expr, final = '', ans.strip()
            # Automatically calculate the expression
            expr_eval = None
            if expr:
                try:
                    # Only allow safe numbers and operators
                    expr_eval = eval(expr, {"__builtins__": None}, {})
                except Exception:
                    expr_eval = None
            # Normalize and clean the answer
            final_clean = delete_extra_zero(_strip_string(final))
            expr_eval_clean = delete_extra_zero(_strip_string(str(expr_eval))) if expr_eval is not None else None
            # Compare, only keep samples where the answer and expression calculation are consistent
            if expr_eval_clean is not None and final_clean != expr_eval_clean:
                continue  # Skip abnormal samples
            data.append({'task': q, 'answer': final_clean, 'expression': expr})
    return data 