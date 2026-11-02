"""Question-level uncertainty, not proof of provenance or causal attribution."""
import math
import numpy as np

def bootstrap_mean(values,seed=97,draws=10000):
    x=np.asarray(values,dtype=float)
    if not len(x):raise ValueError('Empty sample')
    rng=np.random.default_rng(seed)
    means=np.array([x[rng.integers(0,len(x),len(x))].mean() for _ in range(draws)])
    return np.quantile(means,[.025,.975]).tolist()

def wilson(correct,n,z=1.959963984540054):
    if n<=0 or not 0<=correct<=n:raise ValueError('Invalid binomial counts')
    p=correct/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-width),min(1.,center+width)]

def quality_interval(rows):
    groups={}
    for row in rows:
        groups.setdefault(row['id'],[]).append(float(row['quality']))
    values=[float(np.mean(v)) for _,v in sorted(groups.items())]
    single_binary=all(len(v)==1 and v[0] in [0.,1.] for v in groups.values())
    ci=wilson(sum(values),len(values)) if single_binary else bootstrap_mean(values)
    return {'accuracy_question_mean':float(np.mean(values)),'accuracy_ci95':ci,
            'accuracy_interval_method':'Wilson score' if single_binary else 'question bootstrap after seed averaging',
            'independent_questions':len(values),'seeds':sorted({r['seed'] for r in rows})}

def paired_token_saving(route,baseline,seed=97,draws=10000):
    """Paired ratio of means; cluster all seeds of a question before resampling."""
    if set(route)!=set(baseline):raise ValueError('Paired cells differ')
    ids=sorted({id for id,_ in route});a=[];b=[]
    for id in ids:
        cells=[p for p in route if p[0]==id]
        if any(not route[p]['usage_complete'] or not baseline[p]['usage_complete'] for p in cells):
            return {'token_saving_status':'withheld_incomplete_metering'}
        a.append(np.mean([route[p]['total_online_tokens'] for p in cells]))
        b.append(np.mean([baseline[p]['total_online_tokens'] for p in cells]))
    a=np.asarray(a);b=np.asarray(b)
    if b.mean()<=0:raise ValueError('Zero baseline tokens')
    rng=np.random.default_rng(seed);values=[]
    for _ in range(draws):
        indices=rng.integers(0,len(ids),len(ids));den=b[indices].mean()
        if den<=0:return {'token_saving_status':'withheld_zero_resampled_denominator'}
        values.append(100*(1-a[indices].mean()/den))
    return {'token_saving_status':'exploratory_paired_estimate','token_saving_percent':float(100*(1-a.mean()/b.mean())),
            'token_saving_percent_paired_ci95':np.quantile(values,[.025,.975]).tolist()}
