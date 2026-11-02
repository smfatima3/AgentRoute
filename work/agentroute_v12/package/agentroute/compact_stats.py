"""Prospective selected-panel paired inference, with conservative binary bounds."""
import collections
import numpy as np
from scipy.stats import beta,binomtest
from .statistics import wilson

def cp(k,n,alpha=.025):
    return [0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))]

def paired(a,b,draws=10000):
    a={r['id']:r for r in a};b={r['id']:r for r in b}
    if set(a)!=set(b) or not a:raise ValueError('Complete matched question IDs required')
    ids=sorted(a);x=np.array([a[i]['quality'] for i in ids]);y=np.array([b[i]['quality'] for i in ids]);n=len(ids)
    gains=int(np.sum((x==1)&(y==0)));harms=int(np.sum((x==0)&(y==1)))
    g=cp(gains,n);h=cp(harms,n);lower=g[0]-h[1];upper=g[1]-h[0]
    strata=collections.defaultdict(list)
    for k,i in enumerate(ids):strata[a[i].get('stratum','all')].append(k)
    rng=np.random.default_rng(20261029)
    sample=np.concatenate([rng.choice(v,size=(draws,len(v)),replace=True) for v in strata.values()],axis=1)
    delta=(x-y)[sample].mean(axis=1)
    result={'n':n,'difference':float((x-y).mean()),'gains':gains,'harms':harms,
        'exact_mcnemar_p':float(binomtest(gains,gains+harms,.5).pvalue) if gains+harms else 1.,
        'paired_stratified_bootstrap_ci95':np.quantile(delta,[.025,.975]).tolist(),
        'paired_conservative_ci95':[lower,upper],
        'conservative_interval_assumption':'Independent sampled question outcomes; separate 97.5% two-sided gain/harm Clopper-Pearson intervals with union-bound coverage. Stratified unequal-category mixture is not an exact iid binomial draw; interpret as conservative sensitivity under independent-question binomial sampling assumption.',
        'noninferiority_margin':.02,'noninferiority_status':'supported_at_screening_margin' if lower>-.02 else 'inconclusive',
        'strict_zero_loss_lower_bound_positive':lower>0,
        'strata':len(strata),'singleton_strata':sum(len(v)==1 for v in strata.values()),
        'uncertainty_note':'Bootstrap is exploratory and conditional on sampled strata. Zero discordance [0,0] is not evidence of population equivalence; use conservative interval and assumptions.'}
    if all(r.get('usage_complete') and r.get('total_online_tokens') is not None for r in list(a.values())+list(b.values())):
        ta=np.array([a[i]['total_online_tokens'] for i in ids]);tb=np.array([b[i]['total_online_tokens'] for i in ids])
        savings=1-ta[sample].sum(axis=1)/tb[sample].sum(axis=1)
        result.update(token_saving=float(1-ta.sum()/tb.sum()),token_saving_paired_stratified_ci95=np.quantile(savings,[.025,.975]).tolist(),
            candidate_tokens=int(ta.sum()),reference_tokens=int(tb.sum()))
    else:result['token_saving_status']='withheld_incomplete_metering'
    return result

def summary(rows,slo=None):
    n=len(rows);correct=sum(r['quality'] for r in rows)
    latency=[r.get('inference_wall_s',r['wall_s']) for r in rows]
    calls=[c for r in rows for c in r.get('calls',[])]
    deliveries=[d for r in rows for d in r.get('deliveries',[])]
    valid_usage=all(r.get('usage_complete') for r in rows)
    return {'n':n,'correct':int(correct),'accuracy':correct/n,'accuracy_wilson_ci95':wilson(correct,n),
        'invalid':sum(not r.get('format_ok') for r in rows),'failed':sum(r['status']!='ok' for r in rows),
        'prompt_tokens':sum(c['prompt_tokens'] for c in calls) if valid_usage else None,
        'completion_tokens':sum(c['completion_tokens'] for c in calls) if valid_usage else None,
        'total_tokens':sum(r['total_online_tokens'] for r in rows) if valid_usage else None,
        'calls_by_purpose':dict(collections.Counter(c['category'] for c in calls)),
        'deliveries_by_kind':dict(collections.Counter(d['kind'] for d in deliveries)),
        'serialized_bytes_by_kind':{k:sum(d.get('serialized_bytes',d['payload_bytes']) for d in deliveries if d['kind']==k) for k in {d['kind'] for d in deliveries}},
        'latency_p50_p95_p99_s':np.quantile(latency,[.5,.95,.99]).tolist(),
        'latency_tail_note':'Small quality panel at query concurrency two; not a capacity or reliable p99 estimate',
        'slo_s':slo,'correct_on_time':sum(r['quality']==1 and r.get('inference_wall_s',r['wall_s'])<=slo and r['status']=='ok' for r in rows) if slo else None,
        'length_truncated_calls':sum(c.get('finish_reason')=='length' for c in calls),
        'missing_cached_token_details':sum(c.get('cached_tokens') is None for c in calls)}

def analyze(groups,candidate='cascade',slo=None):
    out={'summaries':{m:summary(rows,slo) for m,rows in groups.items()},'paired':{}}
    if candidate not in groups:return out
    for m,rows in groups.items():
        if m!=candidate:out['paired'][m]=paired(groups[candidate],rows)
    ordered=sorted(out['paired'],key=lambda m:out['paired'][m]['exact_mcnemar_p']);maximum=0
    for i,m in enumerate(ordered):
        maximum=max(maximum,min(1.,(len(ordered)-i)*out['paired'][m]['exact_mcnemar_p']))
        out['paired'][m]['holm_p']=maximum
    out['holm_family_size']=len(ordered)
    return out
