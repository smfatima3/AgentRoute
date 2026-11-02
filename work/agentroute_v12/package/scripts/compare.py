"""Fit required TRAIN-only baseline state, then compare every method on common DEV IDs."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agentroute.core import write_json
from agentroute.jobs import select_rows


def command_plan(model,config,datasets,methods,limit,base_url):
    fitted=[m for m in ['safesieve','card','agentnet'] if m in methods]
    common=[sys.executable,str(ROOT/'scripts/suite.py'),'--model',model,'--config',str(config),
            '--base-url',base_url,'--datasets',*datasets,'--jobs-per-gpu','2']
    commands=[]
    if fitted:commands.append(common+['--stage','fit','--methods',*fitted])
    commands.append(common+['--stage','dev','--methods',*methods,'--limit',str(limit)])
    commands.append([sys.executable,str(ROOT/'scripts/analyze.py')])
    return commands


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/study.json')
    p.add_argument('--datasets',nargs='+',default=['mmlu']);p.add_argument('--methods',nargs='+')
    p.add_argument('--limit',type=int,default=50);p.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    c=json.loads(a.config.read_text());methods=a.methods or c['methods']
    if a.model not in c['models'] or a.limit<=0:p.error('Choose the enabled model and a positive question limit')
    if len(set(methods))!=len(methods) or not set(methods)<=set(c['methods']):p.error('Invalid methods')
    if (len(set(a.datasets))!=len(a.datasets) or not set(a.datasets)<=set(c['datasets']) or 'humaneval' in a.datasets):
        p.error('Choose development datasets; HumanEval remains held out')
    plan={'model':a.model,'methods':methods,'datasets':a.datasets,'limit':a.limit,'seed':c['seeds'][0],
          'split':'dev','fitting_uses':'train_only','native_baseline_parity':False,
          'ids':{d:[r['id'] for r in select_rows(ROOT/'data',d,'dev',a.limit)] for d in a.datasets}}
    commands=command_plan(a.model,a.config,a.datasets,methods,a.limit,a.base_url)
    if a.plan_only:print(json.dumps(dict(plan,commands=commands),indent=2));return
    write_json(ROOT/'setup/comparison_plan.json',plan)
    for command in commands:subprocess.run(command,cwd=ROOT,check=True)
    analysis=json.loads((ROOT/'analysis.json').read_text())
    selected=[s for s in analysis['summaries'] if s['model']==a.model and s['split']=='dev' and s['dataset'] in a.datasets and s['method'] in methods]
    health={'scope':'development_only','passed':True,'methods':[]}
    for dataset in a.datasets:
        for method in methods:
            matches=[s for s in selected if s['dataset']==dataset and s['method']==method]
            # A generation gate checks completeness and serialization, never thresholding accuracy.
            ok=(len(matches)==1 and matches[0]['complete'] and matches[0]['failures']==0 and
                matches[0].get('format_error_rate',1)<=.05)
            health['methods'].append({'dataset':dataset,'method':method,'passed':ok})
            health['passed'] &= ok
    write_json(ROOT/'setup/comparison_health.json',health)
    if not health['passed']:
        raise RuntimeError('Comparison saved, but generation/completeness checks failed. Inspect analysis.json and raw attempts before expansion.')
    print('Development comparison complete. Inspect measured quality/cost trade-offs before freeze.')


if __name__=='__main__':main()
