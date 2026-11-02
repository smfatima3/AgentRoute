"""Hash-verified fixed-panel analysis; no model calls, policy selection or raw-result mutation."""
import collections,csv,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1];INPUT=BASE/'outputs/agentroute_v12_completion/analysis_inputs'
sys.path.insert(0,str(ROOT/'package'))
from agentroute.compact_stats import paired,summary
from agentroute.statistics import wilson,paired_token_saving
from agentroute.evaluation import score_text
NAMES={'cascade':'AgentRoute','dense':'Dense','no_comm':'No communication','single':'Single agent','safesieve':'SafeSieve','card':'CARD','selforg':'SelfOrg','agentnet':'AgentNet'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def holm(comparisons):
 ordered=sorted(comparisons,key=lambda k:comparisons[k]['exact_mcnemar_p']);maximum=0
 for i,k in enumerate(ordered):
  maximum=max(maximum,min(1,(len(ordered)-i)*comparisons[k]['exact_mcnemar_p']))
  comparisons[k]['holm_p']=maximum

def load():
 verify=read(INPUT/'DOWNLOAD_VERIFICATION.json');manifest=read(INPUT/'campaign.lock.json');specs={t['task_id']:t for t in manifest['tasks']}
 receipts=[json.loads(s) for s in (INPUT/'results_index.jsonl').read_text().splitlines() if s.strip()]
 rows=[];cache={};seen=set()
 for receipt in receipts:
  spec=specs[receipt['task_id']]
  if spec.get('dataset') not in ['mmlu','bbh'] or spec.get('split')!='test':continue
  if spec['task_id'] in seen:raise ValueError('Duplicate scientific result')
  seen.add(spec['task_id']);ev=next(e for e in receipt['evidence'] if e['role']=='raw')
  path=INPUT/'raw'/(ev['sha256']+Path(ev['path']).suffix)
  if ev['sha256'] not in cache:
   raw=path.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=ev['sha256']:raise ValueError('Corrupt raw result '+str(path))
   x=[json.loads(s) for s in raw.decode().splitlines() if s.strip()] if path.suffix=='.jsonl' else json.loads(raw)
   cache[ev['sha256']]=x if isinstance(x,list) else [x]
  candidates=[r for r in cache[ev['sha256']] if r.get('id')==spec['question_id'] and r.get('method')==spec['method'] and r.get('seed')==spec['seed']]
  if len(candidates)!=1:raise ValueError('Ambiguous record '+spec['task_id'])
  r=dict(candidates[0]);r['_task']=spec;r['_evidence']=ev;r['_receipt_status']=receipt['status'];rows.append(r)
 expected={t['task_id'] for t in manifest['tasks'] if t.get('dataset') in ['mmlu','bbh'] and t.get('split')=='test'}
 if seen!=expected:raise ValueError('Missing fixed-panel inputs: '+str(len(expected-seen)))
 return rows

def main():
 rows=load();stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());dest=BASE/'outputs/agentroute_v12_completion/statistics'/stamp;dest.mkdir(parents=True,exist_ok=False)
 gold={}
 for dataset in ['mmlu','bbh']:
  path=ROOT/'package/compact_data'/dataset/'test.jsonl';gold[dataset]={r['id']:r for r in [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]}
  expected=read(ROOT/'package/compact_data/manifest.json')['files'][dataset+'/test.jsonl']
  if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Dataset hash mismatch')
 rescored=[];disagreements=[]
 for r in rows:
  score=score_text(r.get('final_text',''),gold[r['dataset']][r['id']]) if r['status']=='ok' else {'quality':0.,'format_ok':False}
  item={'task_id':r['_task']['task_id'],'recorded_quality':r['quality'],'rescored_quality':score['quality'],'recorded_format_ok':r.get('format_ok'),'rescored_format_ok':score['format_ok'],'protocol_status':r['status']}
  rescored.append(item)
  if score['quality']!=r['quality'] or score['format_ok']!=r.get('format_ok'):disagreements.append(item)
 save(dest/'INDEPENDENT_RESCORE.json',{'records':rescored,'disagreements':disagreements,'note':'Independent offline recomputation using frozen scorer and hash-verified gold; not a separate human adjudication or alternate scoring algorithm.'})
 if disagreements:raise RuntimeError('Scoring discrepancies; report withheld '+str(dest))
 result={'created_unix_s':time.time(),'held_out_records':len(rows),'primary':{},'ablations':{},'seed_sensitivity':{},'answer_survival':{},'timing_cohorts':{},'limitations':[
  'Fixed selected panels: MMLU 100 questions and BBH 50 questions, not full benchmark populations.',
  'Baseline implementations are documented adaptations; native-paper reproduction is not established.',
  'Accuracy and token comparisons are paired by question. The primary family has seven Holm-corrected comparisons per dataset.',
  'The 2 percentage-point noninferiority margin is a screening threshold; inconclusive does not establish preservation.',
  'Quality-run latency cohorts differ in telemetry overhead. No maximum capacity or universal bottleneck-free claim.',
  'Physical transport replay is a separate subsystem experiment; MATH-based live anchors were excluded by the user.',
  'Inference call tokens include recorded online model calls. Startup, fitting, warmup and failed attempts belong in the separate cost ledger, not these held-out token totals.'
 ]}
 for dataset in ['mmlu','bbh']:
  main=[r for r in rows if r['dataset']==dataset and r['_task']['group'] in ['main_mmlu','short_bbh']]
  groups={m:[r for r in main if r['method']==m] for m in NAMES}
  expected=100 if dataset=='mmlu' else 50
  if any(len(v)!=expected for v in groups.values()):raise ValueError('Incomplete method panel')
  summaries={m:summary(v) for m,v in groups.items()}
  comparisons={m:paired(groups['cascade'],v) for m,v in groups.items() if m!='cascade'};holm(comparisons)
  result['primary'][dataset]={'summaries':summaries,'paired':comparisons,'holm_family_size':7}
  for m,rs in groups.items():
   cohorts=collections.defaultdict(list)
   for r in rs:cohorts[r.get('measurement_mode','legacy_uninstrumented')].append(r)
   result['timing_cohorts'][dataset+'/'+m]={c:{'n':len(v),'p50_p95_p99_s':np.quantile([r.get('inference_wall_s',r['wall_s']) for r in v],[.5,.95,.99]).tolist(),'rss_samples_available':sum(bool(r.get('client_resource_samples')) for r in v)} for c,v in cohorts.items()}
  route={r['id']:r for r in groups['cascade']};abl=collections.defaultdict(list)
  for r in rows:
   if r['dataset']==dataset and r['_task']['group'] in ['component_ablations','stage_ablations','aggregation_controls','self_consistency','selforg_variant']:abl[r['method']].append(r)
  result['ablations'][dataset]={}
  for m,rs in abl.items():
   refs=[route[r['id']] for r in rs]
   result['ablations'][dataset][m]={'summary':summary(rs),'paired_control_minus_agentroute':paired(rs,refs)}
  family={m:v['paired_control_minus_agentroute'] for m,v in result['ablations'][dataset].items()};holm(family)
  records=[]
  for r in groups['cascade']:
   initial=[c for c in r.get('raw_calls',[]) if c.get('category')=='worker'][:4]
   correct=[bool(score_text(c['response'],gold[dataset][r['id']])['quality']) for c in initial]
   records.append({'id':r['id'],'initial_worker_calls':len(initial),'initial_any_correct':any(correct),'initial_correct_count':sum(correct),'final_correct':bool(r['quality']),'local_decision':r.get('local_decision')})
  eligible=[r for r in records if r['initial_worker_calls']==4];had=sum(r['initial_any_correct'] for r in eligible);lost=sum(r['initial_any_correct'] and not r['final_correct'] for r in eligible)
  result['answer_survival'][dataset]={'eligible_questions':len(eligible),'initial_any_correct_questions':had,'correct_answer_lost_questions':lost,'loss_given_initial_correct':lost/had if had else None,'loss_given_initial_correct_wilson95':wilson(lost,had) if had else None,'rescued_questions':sum(not r['initial_any_correct'] and r['final_correct'] for r in eligible),'records':records,'interpretation':'Descriptive initial-candidate-to-final transition; not a causal attribution to aggregation alone.'}
  seeds=[r for r in rows if r['dataset']==dataset and r['_task']['group']=='seed_sensitivity'];by_method={}
  if seeds:
   ids=sorted({r['id'] for r in seeds});rng=np.random.default_rng(20261102);sample=rng.integers(0,len(ids),(10000,len(ids)))
   for method in ['cascade','dense','no_comm']:
    rs=[r for r in seeds if r['method']==method]+[r for r in groups[method] if r['id'] in ids]
    cells={(r['id'],r['seed']):r for r in rs}
    if len(cells)!=len(ids)*3:raise ValueError('Seed panel incomplete')
    means=np.array([np.mean([cells[(q,s)]['quality'] for s in [17,29,43]]) for q in ids])
    by_method[method]={'independent_questions':len(ids),'seed_count':3,'accuracy':float(means.mean()),'question_cluster_bootstrap95':np.quantile(means[sample].mean(axis=1),[.025,.975]).tolist(),'per_seed':{str(s):summary([r for r in rs if r['seed']==s]) for s in [17,29,43]}}
   for method in ['dense','no_comm']:
    a={(r['id'],r['seed']):r for r in seeds+[r for r in groups['cascade'] if r['id'] in ids] if r['method']=='cascade'}
    b={(r['id'],r['seed']):r for r in seeds+[r for r in groups[method] if r['id'] in ids] if r['method']==method}
    by_method[method]['agentroute_paired_tokens']=paired_token_saving(a,b)
   result['seed_sensitivity'][dataset]=by_method
 save(dest/'QUALITY_STATISTICS.json',result)
 compact=[{k:r.get(k) for k in ['id','dataset','method','seed','quality','format_ok','status','total_online_tokens','usage_complete']}|{'task_id':r['_task']['task_id'],'group':r['_task']['group'],'raw_sha256':r['_evidence']['sha256']} for r in rows]
 with (dest/'QUESTION_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
 lines=['# AgentRoute: completed quality-panel results','',f'{len(rows)} held-out method/question/seed evaluations; MATH and SVAMP excluded. Physical transport is reported separately.','']
 for dataset,panel in result['primary'].items():
  lines.extend(['## '+dataset.upper(),'','| Method | Correct / N | Accuracy (95% Wilson CI) | Mean online tokens |','|---|---:|---:|---:|'])
  for m,s in panel['summaries'].items():
   lo,hi=s['accuracy_wilson_ci95'];tokens=f"{s['total_tokens']/s['n']:,.1f}" if s['total_tokens'] is not None else 'Unavailable'
   lines.append(f"| {NAMES[m]} | {s['correct']}/{s['n']} | {100*s['accuracy']:.1f}% ({100*lo:.1f}–{100*hi:.1f}) | {tokens} |")
  lines.extend(['','Positive quality change favors AgentRoute; positive token saving means AgentRoute used fewer tokens.','','| Reference | Accuracy difference (pp) | Conservative 95% CI (pp) | Holm-adjusted p | Token saving (95% paired bootstrap CI) |','|---|---:|---:|---:|---:|'])
  for m,c in panel['paired'].items():
   lo,hi=c['paired_conservative_ci95'];saving='Unavailable'
   if 'token_saving' in c:
    sl,sh=c['token_saving_paired_stratified_ci95'];saving=f"{100*c['token_saving']:.1f}% ({100*sl:.1f}–{100*sh:.1f})"
   lines.append(f"| {NAMES[m]} | {100*c['difference']:+.1f} | {100*lo:.1f} to {100*hi:.1f} | {c['holm_p']:.4g} | {saving} |")
  a=result['answer_survival'][dataset];lines.extend(['',f"Initial-to-final diagnostic: {a['correct_answer_lost_questions']}/{a['initial_any_correct_questions']} questions with a correct initial candidate ended incorrectly. This does not identify aggregation as the sole cause.",''])
 lines.extend(['## Interpretation limits','']+['- '+x for x in result['limitations']])
 lines.extend(['','Rescoring: all recorded quality/format outcomes matched an offline recomputation with the frozen scorer. Raw evidence hashes were checked before analysis. Full ablation, seed-cluster, timing-cohort and answer-survival results are in QUALITY_STATISTICS.json.'])
 (dest/'QUALITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 save(dest/'MANIFEST.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()})
 save(BASE/'outputs/agentroute_v12_completion/STATISTICS_LATEST.json',{'path':str(dest),'status':'QUALITY_ANALYSIS_COMPLETE_PHYSICAL_PENDING','records':len(rows)})
 print(json.dumps({'path':str(dest),'records':len(rows),'rescore_disagreements':len(disagreements),'primary':{d:{m:{k:s[k] for k in ['correct','n','accuracy','total_tokens']} for m,s in p['summaries'].items()} for d,p in result['primary'].items()}}),flush=True)
if __name__=='__main__':main()
