"""Fixed parsers. HumanEval is evaluated only in a restricted Docker container."""
import ast, json, re, shutil, subprocess, tempfile, sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

def final_answer(text):
    lines=re.findall(r'^\s*(?:FINAL(?: ANSWER)?|ANSWER)\s*:\s*(.*?)\s*$',text,re.M|re.I)
    return lines[-1].strip() if lines else (text.strip() if '\n' not in text.strip() else None)

def numeric(text):
    try:
        val=Decimal(text.replace(',','').strip())
        return val if val.is_finite() else None
    except (InvalidOperation,AttributeError): return None

def score_text(text,row):
    pred=final_answer(text)
    if pred is None:return {'quality':0.,'format_ok':False,'prediction':None}
    if row['metric']=='math':
        return score_math(pred,row['answers'])
    if row['metric']=='mcq':
        match=re.fullmatch(r'\(?([A-D])\)?[.]?',pred,re.I)
        pred=match.group(1).upper() if match else None
        ok=pred is not None; correct=ok and pred in row['answers']
    elif row['metric']=='numeric':
        p=numeric(pred);ok=p is not None
        correct=ok and any(p==numeric(a) for a in row['answers'])
    else:
        # Preserve punctuation: BBH includes parenthesis sequences and letter labels.
        norm=lambda x:' '.join(x.split()).casefold()
        ok=True;correct=any(norm(pred)==norm(a) for a in row['answers'])
    return {'quality':float(correct),'format_ok':ok,'prediction':pred}


def score_math(pred,answers,timeout_s=15):
    """Offline symbolic grading only. Never used to route or select an answer.

    Bound input size and isolate parser work behind a hard process deadline.
    A scorer/infrastructure failure is not silently converted into a wrong answer.
    """
    if not answers:raise ValueError('Unresolved gold answer; stop before inference')
    result={'quality':0.,'format_ok':False,'prediction':pred,'evaluator':'math-verify-0.9.0'}
    if not pred or len(pred)>4096:
        return dict(result,evaluation_status='invalid_prediction_size')
    if any(len(a)>16384 for a in answers):raise ValueError('Gold answer exceeds scorer limit')
    worker=Path(__file__).resolve().parents[1]/'scripts/math_score_worker.py'
    try:
        run=subprocess.run([sys.executable,'-I',str(worker)],
            input=json.dumps({'prediction':pred,'answers':answers}),
            capture_output=True,text=True,encoding='utf-8',timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Math scorer deadline exceeded; preserve attempt for adjudication') from exc
    if run.returncode:raise RuntimeError('Math scorer process failed')
    scored=json.loads(run.stdout)
    if not scored['gold_parsed']:raise ValueError('Gold did not parse; adjudication required')
    return dict(result,quality=float(scored['correct']),format_ok=scored['prediction_parsed'],
                evaluation_status='scored' if scored['prediction_parsed'] else 'unparsed_prediction')

def code_source(text,row):
    blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',text,re.S)
    if len(blocks)>1: raise ValueError('Multiple code blocks')
    code=blocks[0] if blocks else text
    # Only complete implementations; never infer code from prose or append a gold solution.
    tree=ast.parse(code)
    if not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==row['entry_point'] for n in tree.body):
        raise ValueError('Missing requested function definition')
    return code

def score_code(text,row,image):
    if not shutil.which('docker'):raise RuntimeError('Docker required for HumanEval scoring')
    if not image or '@sha256:' not in image:raise RuntimeError('Pin evaluator image by digest')
    try: code=code_source(text,row)
    except (ValueError,SyntaxError):return {'quality':0.,'format_ok':False,'evaluation_status':'format_error'}
    # The prompt's imports are public task context. Preserve only import statements.
    imports=[]
    try:
        for n in ast.parse(row['question']).body:
            if isinstance(n,(ast.Import,ast.ImportFrom)): imports.append(ast.unparse(n))
    except SyntaxError: pass
    source='\n'.join(imports)+'\n'+code+'\n'+row['test']+'\ncheck('+row['entry_point']+')\n'
    with tempfile.TemporaryDirectory(prefix='agentroute_eval_') as d:
        p=Path(d)/'candidate.py';p.write_text(source);p.chmod(0o444)
        name='agentroute-eval-'+Path(d).name.lower().replace('_','-')
        command=['docker','run','--rm','--name',name,'--network','none','--read-only',
            '--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','64',
            '--memory','512m','--cpus','1','--user','65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,size=16m',
            '--mount',f'type=bind,source={p.resolve()},target=/candidate.py,readonly',image,
            'python','-I','-B','/candidate.py']
        try:
            run=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=20)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            return {'quality':0.,'format_ok':True,'evaluation_status':'timeout'}
        if run.returncode in [125,126,127]:raise RuntimeError('Evaluator infrastructure failure')
        return {'quality':float(run.returncode==0),'format_ok':True,
                'evaluation_status':'passed' if run.returncode==0 else 'failed_tests'}
