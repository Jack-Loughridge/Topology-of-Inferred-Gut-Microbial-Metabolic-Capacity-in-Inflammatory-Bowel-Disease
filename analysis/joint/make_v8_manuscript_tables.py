"""Generate the v8 paper tables from the archived aggregate CSVs only.

No fitted models or individual predictions are needed. This does not recompute
AUC or log loss from raw observations and is not a classifier training script.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

METRICS = [('accuracy','Accuracy'),('balanced_accuracy','Balanced accuracy'),
           ('macro_f1',r'Macro $F_1$'),('roc_auc','ROC AUC'),('log_loss','Log loss')]

def table(caption,label,columns,header,rows,note):
    return '\n'.join([r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption{\jointfinal{'+caption+'}}',r'\label{'+label+'}',
        r'\begingroup\bfseries\boldmath',r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{'+columns+'}',r'\toprule',
        header+r' \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',
        r'\par\smallskip\begin{minipage}{0.98\linewidth}\footnotesize '+note,
        r'\end{minipage}',r'\endgroup',r'\end{table}',''])

def generate(source: Path, dest: Path):
    dest.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(source/'performance_summary.csv')
    delta=pd.read_csv(source/'paired_summary.csv')
    repeat=pd.read_csv(source/'repetition_metrics.csv')
    conf=pd.read_csv(source/'consensus_SEPARATE.csv')
    assert len(p)==20 and set(p['count'])=={5}
    def get(arm,level,metric):
        a=p[(p.model==arm)&(p.level==level)&(p.metric==metric)]
        assert len(a)==1
        return a.iloc[0]
    for level in ['participant','sample']:
        rows=[]
        for met,name in METRICS:
            a=get('joint',level,met); b=get('ricci',level,met)
            d=delta[(delta.level==level)&(delta.metric==met)].iloc[0]
            assert np.isclose(a['mean']-b['mean'],d['mean'],rtol=0,atol=1e-14)
            rows.append(fr'{name} & ${a["mean"]:.3f} \pm {a["std"]:.3f}$ & ${b["mean"]:.3f} \pm {b["std"]:.3f}$ & ${d["mean"]:+.3f}$ \\')
        n='106 participants' if level=='participant' else '1,317 samples'
        text=table(f'Exploratory regularised joint $H_0$ and Ricci performance for IBD versus non-IBD at {level} level. The joint model used fixed requested $M=32$, $H_0$ ridge $0.001$ and Ricci $C=0.02$. The control used the same Ricci penalty and matched outer partitions. Values are mean $\\pm$ sample SD across five repetitions.',
            'tab:joint-v8-'+level,'lrrr',r'Metric & Joint $H_0$ + Ricci & Matched Ricci & Mean difference',rows,
            f'Each repetition includes the same {n}. '+'Within each repetition, all five outer test folds are pooled. '+
            ('Sample probabilities are averaged within participant before evaluation. ' if level=='participant' else '')+
            'Differences are joint minus control. Lower log loss is better. Repetition variation is not uncertainty from independent cohorts. Historical SAGA results are not substituted for this matched control.')
        (dest/(level+'.tex')).write_text(text)
    rows=[]
    vals=repeat[(repeat.level=='participant')&(repeat.metric=='roc_auc')].pivot(index='repeat',columns='model',values='value')
    for r,a in vals.iterrows(): rows.append(fr'${r}$ & ${a["joint"]:.3f}$ & ${a["ricci"]:.3f}$ & ${a["joint"]-a["ricci"]:+.3f}$ \\')
    (dest/'auc_repeats.tex').write_text(table('Participant ROC AUC in each of the five matched repetitions of the regularised joint comparison.',
        'tab:joint-v8-auc-repeats','rrrr','Repetition & Joint & Matched Ricci & Difference',rows,
        'The paired difference is positive in all five repetitions. These are overlapping partitions of one cohort, not five independent validation cohorts.'))
    rows=[]
    for arm,name in [('joint','Joint'),('ricci','Matched Ricci')]:
        a=conf[(conf.model==arm)&(conf.level=='participant')].iloc[0]
        rows.append(fr'{name} & ${int(a.TN)}$ & ${int(a.FP)}$ & ${int(a.FN)}$ & ${int(a.TP)}$ & ${a.balanced_accuracy:.3f}$ & ${a.roc_auc:.3f}$ \\')
    (dest/'consensus.tex').write_text(table('Separate participant consensus results for the regularised joint comparison. Probabilities were averaged across repetitions before assigning one label per participant.',
        'tab:joint-v8-consensus','lrrrrrr','Model & TN & FP & FN & TP & Bal. acc. & AUC',rows,
        'TN and TP are correct non-IBD and IBD classifications. FP and FN are their respective errors. There are 26 non-IBD and 80 IBD participants. These consensus estimates do not replace the mean repetition results.'))
    # Retain every available repetition metric for both evaluation levels.
    rows=[]
    for level in ['participant','sample']:
        for arm,name in [('joint','Joint'),('ricci','Ricci')]:
            a=repeat[(repeat.level==level)&(repeat.model==arm)].pivot(index='repeat',columns='metric',values='value')
            for r,v in a.iterrows():
                rows.append(f'{level.title()} & {name} & ${r}$ & '+ ' & '.join(f'${v[m]:.3f}$' for m,_ in METRICS)+r' \\')
    (dest/'all_repetitions.tex').write_text(table('All regularised joint and matched Ricci repetition metrics. Each row pools the five held out folds within one repetition.',
        'tab:joint-v8-all-repetitions','llrrrrrr','Level & Model & Repeat & Accuracy & Bal. acc. & Macro $F_1$ & AUC & Log loss',rows,
        'Participant probabilities are means of sample probabilities. Lower log loss is better. No unfinished fold or repetition is omitted.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    generate(args.source,args.output)
