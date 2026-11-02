"""Plot only complete measured result groups; never generate placeholder values."""
import argparse,json
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=json.loads(a.analysis.read_text())
    rows=[r for r in report['summaries'] if r['complete']]
    if not rows:raise RuntimeError('No complete measured results to plot')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.output.mkdir(parents=True,exist_ok=True)
    for model,dataset,split in sorted({(r['model'],r['dataset'],r['split']) for r in rows}):
        group=[r for r in rows if (r['model'],r['dataset'],r['split'])==(model,dataset,split)]
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for ax,metric,title in zip(axes,['accuracy','peer_messages','total_online_tokens'],['Accuracy / pass@1','Peer messages / query','All online tokens / query']):
            available=[r for r in group if r.get(metric) is not None]
            ax.bar([r['method'] for r in available],[r[metric] for r in available],color=['#076b8b' if r['method']=='cascade' else '#879aa5' for r in available])
            ax.set_title(title);ax.tick_params(axis='x',rotation=55);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if metric=='accuracy':ax.set_ylim(0,1)
        fig.suptitle(f'{model} · {dataset} · {split}\nExploratory common-task adapters; native parity pending',fontsize=11)
        stem=f'{model.split("/")[-1]}_{dataset}_{split}'
        fig.savefig(a.output/(stem+'.pdf'));fig.savefig(a.output/(stem+'.png'),dpi=180);plt.close(fig)
