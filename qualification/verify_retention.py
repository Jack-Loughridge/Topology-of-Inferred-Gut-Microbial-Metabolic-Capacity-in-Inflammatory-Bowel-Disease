"""Verify retained evidence and physical allocation on the restored invented archive."""
from collections import Counter
import json
from pathlib import Path

from analysis.alpha_matched import comparator_run as driver
from analysis.alpha_matched import report_development as report
from analysis.alpha_matched.artifacts import Store
from analysis.alpha_matched.partitions import canonical_json
from qualification.integrated_storage import OUT, digest, guard, read, save


def checkpoint_inventory(store, objects):
    count = 0
    for marker in store.root.glob('unit-*.json'):
        name = read(marker)['record']
        assert name in objects
        state = driver.read(store,name)
        assert marker.name == 'unit-'+state['key']+'.json'
        assert driver.references(store, state['result']) <= objects
        count += 1
    return count


def challenge_report(store, objects, fit, frames, cfg):
    """Rewrite an authentic invented OOF wrapper without editing any stored evidence."""
    for mutation in ('omit_blocks', 'unpublished_wrapper'):
        altered = dict(fit)
        if mutation == 'omit_blocks':
            altered.pop('blocks')
        else:
            altered['invented_extra'] = True
        data = canonical_json(altered)
        key = digest(data) + '.json'

        class Overlay(Store):
            def read(self, name):
                return data if name == key else super().read(name)
        try:
            report.fit_table(Overlay(store.root),objects | {key},key,fit['arm'],fit['name'],fit['split'],*frames,cfg)
        except ValueError as exc:
            expected = 'Missing standalone blocks' if mutation == 'omit_blocks' else 'completed source-bound fit'
            if expected not in str(exc):
                raise
        else:
            raise AssertionError('Accepted rewritten OOF evidence')


def verify():
    root = OUT/'restored'
    guard()
    core = driver.core
    fold = core.FINAL
    rec, rid = core.release(root, fold)
    cfg, splits, meta, _, _ = core.read_inputs(root, rec)
    proc = core.Procedure(meta,splits,None,{},[],None)
    s = Store(root/'jobs/development-final-refits'/rid/fold)
    body = driver.read(s,read(s.root/'receipt.json')['record'])
    objects = set(body['objects'])
    classes = Counter()
    units = checkpoint_inventory(s,objects)
    all_blocks = {}
    for arm, result in body['arms'].items():
        guard()
        rule = driver.read(s,result['decision_rules'])
        for fits in rule['heldout_fits'].values():
            assert set(fits) <= set(result['fits'])
        assert set(result['outputs'].values()) <= set(result['fits'])
        for key in set(result['trials']):
            assert key in objects
            assert set(driver.read(s,key)['fold_fits']) <= set(result['fits'])
        for key in set(result['fits']):
            assert key in objects
            fit = driver.read(s,key)
            assert driver.references(s,fit) <= objects
            frames = proc.frames(fit['split'])
            if fit['split'] == fold:
                assert set(s.deployment_files(fit['deployment'])) <= objects
                classes['deployment_fits'] += 1
            else:
                report.fit_table(s,objects,key,arm,fit['name'],fit['split'],*frames,cfg)
                classes['oof_fits_with_participant_means'] += 1
                if fit['name']=='Ricci' and not classes['rejected_report_rewrites']:
                    challenge_report(s,objects,fit,frames,cfg)
                    classes['rejected_report_rewrites'] = 2
            for b in fit['blocks']:
                all_blocks[digest(canonical_json(b))] = b
        guard()
        report.decision_rules(s,objects,result['decision_rules'],arm,list(result['outputs']),fold,proc.frames(fold)[0],
                              {k:(a,b) for k,a,b in proc.children(fold,proc.frames(fold)[0])},cfg)
    classes['unique_blocks'] = len(all_blocks)
    classes['compact_blocks'] = sum('contract' in b for b in all_blocks.values())
    anchor = read(OUT/'evidence/package.json')['result']
    _, _, _, cmpid = driver.release(root,fold)
    cmp, cb = driver.report_fold(root/'jobs/comparator-folds'/cmpid/fold,cmpid,rid,fold,anchor['comparator'])
    cmp_objects = set(cb['objects'])
    cmp_units = checkpoint_inventory(cmp,cmp_objects)
    for name in cmp_objects:
        cmp.read(name)
    def file_inodes(directory):
        return {(st.st_dev, st.st_ino) for path in directory.rglob('*') if path.is_file() for st in [path.stat()]}
    shared_inodes = file_inodes(root) & file_inodes(OUT/'clean')
    assert not shared_inodes
    original = read(OUT/'original-replay.json')
    review = read(OUT.parents[3]/'reviews/FINAL-REFIT-EXPORT-ADDENDUM-20261125.json')
    assert original['marker']==review['independent_evidence']['binary_primary_marker']
    return dict(archive_source_inodes_shared=len(shared_inodes),core_objects=len(objects), comparator_objects=len(cmp_objects), core_units=units,
                comparator_units=cmp_units, comparator_results=len(cb['results']), counts=dict(classes),
                original_marker_independently_pinned=original['marker'])


if __name__=='__main__':
    result=verify()
    save(OUT/'retention.json',result)
    print(json.dumps(result,sort_keys=True))
