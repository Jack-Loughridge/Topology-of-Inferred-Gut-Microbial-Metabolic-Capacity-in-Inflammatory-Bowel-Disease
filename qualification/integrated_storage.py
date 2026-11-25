"""Private, resumable invented-only integration. Run from the isolated candidate checkout."""
import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
RUN = BASE.parents[1]
OUT = BASE / 'outputs/integrated-storage-final'
TARGET = '4d6b80a70430fcd058af551b7f5d950905c05196'
THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(encoded(obj))
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def read(path):
    return json.loads(path.read_bytes())


def guard(allowance=0):
    if os.path.lexists(RUN / 'STOP') or read(RUN / 'status.json')['status'] != 'ACTIVE':
        raise RuntimeError('Alpha stopped')
    if shutil.disk_usage(BASE).free < 5 * 1024**3 + allowance + 65536:
        raise OSError('Private durable reserve reached')


def usage(root):
    seen = {}
    if root.exists():
        for directory, _, files in os.walk(root):
            for p in [Path(directory), *(Path(directory)/f for f in files)]:
                try:
                    s = p.lstat()
                except FileNotFoundError:
                    continue
                seen[s.st_dev, s.st_ino] = s.st_blocks * 512
    return sum(seen.values())


def fixture(root, dense):
    from analysis.alpha_matched import run_fold
    from analysis.alpha_matched.test_artifacts import fixture as make
    if root.exists():
        return
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        staged = Path(tmp) / 'invented'
        make(staged, final=True, masked=True)
        if dense:
            path = staged/'manifests/development-fold-inputs/config.json'
            config = read(path)
            config['ricci_storage'] = 'dense'
            save(path, config)
            review = staged/'reviews/test.json'
            released = read(review)
            released['inputs']['config']['sha256'] = digest(path.read_bytes())
            save(review, released)
            status = read(staged/'status.json')
            status['final_refit_release']['sha256'] = digest(review.read_bytes())
            save(staged/'status.json', status)
        run_fold.release(staged, run_fold.FINAL)
        staged.rename(root)


def inventory(root):
    """All regular bytes and hardlink equivalence classes in an independent archive."""
    files = {}
    links = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Unexpected symlink')
        if p.is_file():
            name = p.relative_to(root).as_posix()
            files[name] = digest(p.read_bytes())
            st = p.stat()
            links.setdefault((st.st_dev, st.st_ino), []).append(name)
    return dict(files=digest(encoded(files)), hardlinks=digest(encoded(sorted(links.values()))),
                names=len(files), distinct_files=len(links))


def audit(root):
    from analysis.alpha_matched import run_fold as core
    from analysis.alpha_matched.artifacts import Store
    from analysis.alpha_matched.partitions import array_id
    rec, rid = core.release(root, core.FINAL)
    directory = root/'jobs/development-final-refits'/rid/core.FINAL
    s = Store(directory)
    marker = directory/'receipt.json'
    body = json.loads(s.read(read(marker)['record']))
    objects = set(body['objects'])
    for name in objects:
        s.read(name)
    results = {}
    vectors, fits, choices, trials, decisions, proof_counts = {}, {}, {}, {}, {}, {}
    for arm, ledger in body['arms'].items():
        names = {}
        for key in set(ledger['fits']):
            assert key in objects
            fit = json.loads(s.read(key))
            kinds = []
            for block in fit['blocks']:
                model = json.loads(s.read(block['model']))
                source = json.loads(s.read(model['source'])) if model['name'] == 'K0' else model
                config = source['state'].get('setting', {})
                kinds.append([model['name'], config])
                for side in ('train', 'validate'):
                    label = encoded([arm, fit['split'], *kinds[-1], side]).decode()
                    if label not in vectors:
                        v = s.load_vector(block[side])
                        vectors[label] = dict(identity=v.identity(), values=array_id(v.values),
                                              layout=digest(encoded([v.block, v.arm, v.partition, v.ids, v.features])),
                                              generator=v.generator)
            label = encoded([arm, fit['name'], fit['split'], kinds, fit['head']]).decode()
            names[key] = label
            fits[label] = {k:fit[k] for k in ('classes', 'participants', 'samples', 'train_groups',
                                                'probabilities', 'participant_means', 'fitted') if k in fit}
        choices[arm] = {k:names[v] for k,v in ledger['outputs'].items()}
        trial = []
        for key in ledger['trials']:
            assert key in objects
            row = json.loads(s.read(key))
            trial.append(dict(row, fold_fits=[names[v] for v in row['fold_fits']]))
        trials[arm] = trial
        assert ledger['decision_rules'] in objects
        rule = json.loads(s.read(ledger['decision_rules']))
        decisions[arm] = dict(rule, heldout_fits={k:[names[v] for v in vv] for k,vv in rule['heldout_fits'].items()})
        proof_counts[arm] = dict(fits=len(set(ledger['fits'])), trials=len(set(ledger['trials'])),
                                 outputs=len(choices[arm]), thresholds=len(rule['thresholds']))
    results.update(source=digest(encoded(rec['source_sha256'])), release=rid, marker=digest(marker.read_bytes()),
                   allocated=usage(directory), objects=len(objects), extensions=dict(Counter(Path(k).suffix for k in objects)),
                   geometry_entries=len(body['geometry']), units=len(list(directory.glob('unit-*.json'))),
                   counts=proof_counts, vectors=vectors,
                   fits=digest(encoded(fits)), trials=digest(encoded(trials)),
                   outputs=digest(encoded(choices)), rules=digest(encoded(decisions)))
    return results


def worker(phase):
    from analysis.alpha_matched import comparator_run as driver
    from analysis.alpha_matched import run_fold as core
    from analysis.alpha_matched.test_comparator_run import register
    label = {'reference':'reference', 'dense':'dense', 'clean':'clean', 'break':'interrupted',
             'resume':'interrupted', 'seal':'clean', 'package':'clean', 'restore':'restored'}[phase]
    root = OUT / label
    if phase in ('resume', 'seal', 'package', 'restore') and not root.exists():
        raise ValueError('Missing earlier invented phase')
    fixture(root, phase == 'dense')
    rec, rid = core.release(root, core.FINAL)
    directory = root/'jobs/development-final-refits'/rid/core.FINAL
    assert rec['source_sha256'] == core.source_hashes()
    deny = AssertionError('Recomputed an already sealed scientific unit')
    if phase == 'break':
        if (OUT/'break-state.json').exists():
            return read(OUT/'break-state.json')
        original = core.Fold._fit
        def interrupt(self, name, *args):
            if self.factory.arm == 'RELATIVE' and name == 'Ricci':
                raise InterruptedError('invented mid-run failure')
            return original(self, name, *args)
        with patch.object(core.Fold, '_fit', interrupt):
            try:
                core.execute(root, core.FINAL)
            except InterruptedError:
                assert not (directory/'receipt.json').exists()
            else:
                raise AssertionError('Interruption did not fire')
        units = {p.name:digest(p.read_bytes()) for p in directory.glob('unit-*.json')}
        assert units
        result = dict(completed_units=units)
        save(OUT/'break-state.json', result)
        return result
    if phase in ('seal', 'restore'):
        with patch.object(core.Fold, '_build', side_effect=deny), patch.object(core.Fold, '_fit', side_effect=deny):
            core.execute(root, core.FINAL)
    elif phase != 'package':
        core.execute(root, core.FINAL)
    result = audit(root)
    if phase == 'resume':
        saved = read(OUT/'break-state.json')['completed_units']
        assert all(digest((directory/k).read_bytes()) == v for k,v in saved.items())
    if phase in ('package', 'restore'):
        from analysis.alpha_matched.export_final import export
        if phase == 'package':
            cmp = register(root, core.FINAL, primary=result['marker'])
            driver.execute(root, core.FINAL)
        else:
            _, _, _, cmpid = driver.release(root, core.FINAL)
            cmp = driver.Store(root/'jobs/comparator-folds'/cmpid/core.FINAL)
        marker = digest((cmp.root/'receipt.json').read_bytes())
        with patch.object(core.Fold, '_build', side_effect=deny), patch.object(core.Fold, '_fit', side_effect=deny), patch.object(driver.heads,'fit',side_effect=deny):
            exported = export(root, result['marker'], marker)
        result.update(comparator=marker, export=digest(exported.read_bytes()), procedures=len(read(exported)['procedures']),
                      private_tree=inventory(root), private_total=usage(root))
        assert result['procedures'] == 42
    return result


def archive_restore():
    root = OUT/'clean'
    expected = inventory(root)
    archive = OUT/'invented.tar.gz'
    guard(usage(root) + 2**20)
    if not archive.exists():
        with tempfile.NamedTemporaryFile(dir=OUT, delete=False) as raw:
            with gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as compressed, tarfile.open(
                    mode='w|', fileobj=compressed, dereference=False) as tar:
                def fixed(info):
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    return info
                tar.add(root, arcname='invented', filter=fixed)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(raw.name, archive)
    restored = OUT/'restored'
    if not restored.exists():
        with tempfile.TemporaryDirectory(dir=OUT) as tmp, tarfile.open(archive) as tar:
            tar.extractall(tmp, filter='data')
            tree = Path(tmp)/'invented'
            if inventory(tree) != expected:
                raise ValueError('Archive inventory or hardlink topology mismatch')
            tree.rename(restored)
    if inventory(restored) != expected:
        raise ValueError('Restored tree differs')
    return dict(archive=digest(archive.read_bytes()), archive_file=archive.stat().st_size,
                archive_allocated=archive.stat().st_blocks * 512, restored_allocated=usage(restored), **expected)


def launch(phase, source):
    rootname = {'break':'interrupted','resume':'interrupted','seal':'clean','package':'clean','restore':'restored'}.get(phase, phase)
    root = OUT/rootname
    env = dict(os.environ, **THREADS, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               TMPDIR=str(OUT/'tmp'), XDG_CACHE_HOME=str(OUT/'tmp'))
    with (OUT/'logs'/(phase+'.log')).open('ab') as log:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', phase],
                                 cwd=source, env=env, stdout=log, stderr=log)
        peak = 0
        samples = 0
        try:
            while child.poll() is None:
                guard()
                peak = max(peak, usage(root))
                samples += 1
                time.sleep(.5)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
        if child.returncode:
            raise RuntimeError('Check private worker log for '+phase)
    return dict(peak_sampled=max(peak, usage(root)), samples=samples, end_allocated=usage(root))


def main():
    guard(32*2**20)
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ('logs','tmp','evidence'):
        (OUT/sub).mkdir(exist_ok=True)
    ref = OUT/'reference-source'
    if not ref.exists():
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            path = Path(tmp)/'code.tar'
            subprocess.run(['git','archive','-o',str(path),TARGET,'analysis/alpha_matched'],cwd=BASE,check=True)
            tree = Path(tmp)/'source'
            tree.mkdir()
            with tarfile.open(path) as tar:
                tar.extractall(tree, filter='data')
            tree.rename(ref)
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', TARGET, 'analysis/alpha_matched'], cwd=BASE).decode().splitlines()
    for name in paths:
        original = subprocess.check_output(['git', 'show', TARGET+':'+name], cwd=BASE)
        if (ref/name).read_bytes() != original:
            raise ValueError('Reference source snapshot changed')
    context = dict(target=TARGET, harness=digest(Path(__file__).read_bytes()), python=sys.executable,
                   source={p.name:digest(p.read_bytes()) for p in sorted((BASE/'analysis/alpha_matched').glob('*.py'))},
                   threads=THREADS)
    context_path = OUT/'context.json'
    if context_path.exists() and read(context_path)!=context:
        raise ValueError('Run context changed; retain earlier evidence and use a fresh candidate evidence directory')
    save(context_path,context)
    for phase in ('reference','dense','clean','break','resume','seal','package','archive','restore'):
        guard()
        state = OUT/'evidence'/(phase+'.json')
        if state.exists():
            continue
        if phase == 'archive':
            row = archive_restore()
            row['archive_and_two_trees_allocated'] = row['archive_allocated'] + usage(OUT/'clean') + usage(OUT/'restored')
        else:
            measurement = launch(phase, ref if phase == 'reference' else BASE)
            row = dict(measurement=measurement, result=read(OUT/(phase+'-result.json')))
        save(state, row)
        print('Completed',phase,flush=True)
    ev = {p.stem:read(p) for p in (OUT/'evidence').glob('*.json')}
    core = {k:ev[k]['result'] for k in ('reference','dense','clean','resume','seal','package','restore')}
    clean = core['clean']
    for label, rec in core.items():
        assert rec['vectors'].keys()==clean['vectors'].keys()
        for k,v in clean['vectors'].items():
            for field in (('values','layout') if label=='reference' else v):
                assert v[field] == rec['vectors'][k][field], (label,k,field)
        assert all(rec[k]==clean[k] for k in ('fits','trials','outputs','rules','counts'))
    assert len({core[k]['marker'] for k in ('clean','resume','seal','restore')})==1
    assert core['package']['export']==core['restore']['export'] and core['package']['comparator']==core['restore']['comparator']
    result = dict(reference_source=core['reference']['source'], candidate_source=clean['source'],vectors=len(clean['vectors']),
                  original_identities_equal=sum(v['identity']==core['reference']['vectors'][k]['identity'] for k,v in clean['vectors'].items()),
                  candidate_identities_equal=len(clean['vectors']), counts=clean['counts'], objects=clean['objects'],
                  extensions=clean['extensions'], geometry_entries=clean['geometry_entries'], units=clean['units'],
                  interrupted_units=len(ev['break']['result']['completed_units']), archive=ev['archive'],
                  marker=clean['marker'], export=core['restore']['export'], comparator=core['restore']['comparator'],
                  phases={k:v['measurement'] for k,v in ev.items() if k!='archive'},
                  core_allocated={k:v['allocated'] for k,v in core.items()})
    save(OUT/'summary.json', result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == '__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument('--worker', choices=['reference','dense','clean','break','resume','seal','package','restore'])
    args=arg.parse_args()
    guard()
    if args.worker:
        save(OUT/(args.worker+'-result.json'), worker(args.worker))
    else:
        main()
