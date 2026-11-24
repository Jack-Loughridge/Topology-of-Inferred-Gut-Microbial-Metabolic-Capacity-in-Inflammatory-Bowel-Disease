"""Fit decision thresholds from complete, group-excluded pipelines in parent training."""
import json

import pandas as pd

from .statistics import select_inner_oof_threshold


def select_thresholds(runner,parent):
    """A child selection repeats the whole search with its own sub-inner partitions."""
    train,_=runner.proc.frames(parent)
    classes=runner.config['classes']
    if len(classes)!=2:
        return runner.store.record(dict(rule='argmax',class_order=classes))
    probabilities={}
    source_records={}
    for child,a,b in runner.proc.children(parent,train):
        # No parameters or candidate configuration selected on parent validation
        # are passed down. Fit/selection/combination all repeat inside a.
        result=runner.run(child)
        for method,fit_id in result['outputs'].items():
            fit=json.loads(runner.store.read(fit_id))
            if (fit['split']!=child or list(fit['samples'])!=list(b.index) or
                    set(fit['train_groups'])!=set(a.group) or set(a.group)&set(b.group)
                    or fit['classes']!=classes):
                raise ValueError('Invalid complete-pipeline OOF provenance')
            probs=pd.DataFrame(runner.store.load_array(fit['probabilities']),index=b.index,columns=classes)
            probabilities.setdefault(method,[]).append((fit['train_groups'],probs))
            source_records.setdefault(method,[]).append(fit_id)
    values={method:select_inner_oof_threshold(train,classes,parts) for method,parts in probabilities.items()}
    return runner.store.record(dict(rule='max-participant-balanced-accuracy',parent=parent,
                   class_order=classes,thresholds=values,heldout_fits=source_records))
