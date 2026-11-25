"""Export verified development final-refit procedures for later G6 assembly."""
import argparse
import json
from pathlib import Path

from . import comparator_run as driver
from . import report_comparators as amended
from . import report_development as development
from . import run_fold as core
from .artifacts import Store, sha
from .nesting import Procedure
from .partitions import canonical_json


class FinalEvidence(amended.FitEvidence):
    def predictions(self, ref, method, entry, node, candidate):
        if node != core.FINAL:
            return super().predictions(ref, method, entry, node, candidate)
        fit = self.record(ref)
        train, valid = self.proc.frames(node)
        head = self.record(fit['fitted'])
        if (len(valid) or 'probabilities' in fit or fit['name'] != method or fit['split'] != node
                or fit['endpoint'] != self.config['endpoint'] or fit['arm'] != entry['arm']
                or fit['classes'] != self.config['classes'] or fit['candidate'] != candidate
                or fit['train_groups'] != sorted(set(train.group)) or head['config'] != candidate
                or head['classes'] != list(range(len(fit['classes'])))
                or not set(driver.heads.dependencies(self.store,fit['fitted'])) <= self.objects):
            raise ValueError('Final head differs from selected procedure')
        self.unit(['fit',method,node,fit['blocks'],candidate],ref)
        x = self.vectors(fit['blocks'],entry,node,train,valid)
        if x.shape != (0,head['n_features']):
            raise ValueError('Final application shape differs from frozen feature width')
        return None


class SealedStore(Store):
    """Allow existing producer bytes to be checked; prohibit creating an absent record."""
    def publish(self, name, data):
        path = self.root/name
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError('Sealed core replay differs from producer artifacts')


def replay_core(root, parent, parent_id, store, body, config, splits, meta, graphs, species):
    """Reconstruct both producer outputs, including every nested threshold choice, in sealed mode."""
    if not (store.root/'receipt.json').is_file():
        raise ValueError('Sealed final core receipt required')
    runtime = json.loads((store.root/'runtime.json').read_bytes())
    version = sha(canonical_json(parent['source_sha256']))
    certificates = set()

    class CheckedUnits(core.unitcache.Units):
        def run(self, identity, compute):
            core.active(root)
            return super().run(identity,compute)

    for arm in driver.ARMS:
        factory = core.Factory(graphs,arm,store.root/'geometry',config['tolerances'])
        factory.version = version
        proc = Procedure(meta,splits,factory,{},[],None)
        checker = core.Fold(proc,factory,store,config,species,lambda: core.active(root))
        # Reuse the producing runtime identity. Loading comparator libraries for
        # export can change the live threadpool inventory in this process.
        checker.units = CheckedUnits(store,[parent_id,arm,version,config,runtime])
        if not checker.units.sealed:
            raise ValueError('Refusing unsealed core replay')
        result = checker.run(core.FINAL)
        if config.get('fit_thresholds'):
            result['decision_rules'] = core.select_thresholds(checker,core.FINAL)
        if canonical_json(result) != canonical_json(body['arms'][arm]):
            raise ValueError('Core final selection or threshold differs from producer units')
        certificates.update(checker.certificates)
    if not store.objects.issubset(body['objects']):
        raise ValueError('Sealed core dependency absent from final receipt inventory')
    if core.geometry_files(store.root,certificates) != body['geometry']:
        raise ValueError('Core final geometry evidence differs from sealed producer inventory')


def export(root, primary_marker, comparator_marker):
    root = Path(root).resolve(strict=True)
    fold = core.FINAL
    parent, parent_id = core.release(root,fold)
    _, rec, budget, rid = driver.release(root,fold)
    config, splits, meta, graphs, species = core.read_inputs(root,parent)
    proc = Procedure(meta,splits,None,{},[],None)
    train, valid = proc.frames(fold)
    if len(valid) or set(train.index) != set(meta.index):
        raise ValueError('Incomplete development final-refit roster')
    children = {k:(a,b) for k,a,b in proc.children(fold,train)}
    pdir = root/'jobs/development-final-refits'/parent_id/fold
    marker_bytes = (pdir/'receipt.json').read_bytes()
    if sha(marker_bytes) != primary_marker:
        raise ValueError('Changed primary final-refit marker')
    primary = SealedStore(pdir)
    body = driver.read(primary,json.loads(marker_bytes)['record'])
    if (body['release'],body['fold'],body['inputs'],set(body['arms'])) != (
            parent_id,fold,parent['inputs'],set(driver.ARMS)):
        raise ValueError('Wrong core final receipt')
    replay_core(root,parent,parent_id,primary,body,config,splits,meta,graphs,species)
    cstore, cmp = driver.report_fold(root/'jobs/comparator-folds'/rid/fold,rid,parent_id,fold,comparator_marker)
    expected = driver.registry(budget)
    registration = driver.read(cstore,cmp['registration'])
    if (cmp['status'] != 'complete' or cmp['inputs'] != parent['inputs'] or expected != cmp['registry']
            or registration != dict(entries=expected, implementation=rec['source_sha256'], parent=parent_id,
                                    release=rid,packages=budget['packages'],neural_review=rec['neural_review'],
                                    mandatory_complete=True,budget=rec['budget'])):
        raise ValueError('Incomplete or changed comparator registry')
    selected = driver.FrozenPrimary(root,fold,parent,rec) if budget['frozen_vectors'] else None
    panel = parent['inputs']['bundle']
    bundle = driver.document(root/'manifests/development-fold-inputs',dict(file=panel['path'],sha256=panel['sha256']))
    ce = FinalEvidence(cstore,set(cmp['objects']),proc,config,budget,rid,fold,selected,bundle)
    validators = {Path(m.__file__).name:core.bundle.sha(Path(m.__file__)) for m in (amended,development)}
    version = core.bundle.sha(Path(__file__))
    identity = sha(canonical_json([parent_id,rid,primary_marker,comparator_marker,version,validators]))
    dest = root/'jobs/final-procedure-exports'/identity
    output = Store(dest/'objects')
    names = development.methods_by_arm(config)
    procedures = []
    files = {}
    model_paths = set()
    decision_sources = set()
    training_ids = sorted(sha(canonical_json(i)) for i in train.index)

    def add(store,model,closure,objects,arm,method,requires_curvature,rule,fit_id,decision_id):
        if not (set(closure) | {decision_id}) <= objects:
            raise ValueError('Incomplete final dependency inventory')
        spec = driver.read(store,model)
        if (spec['classes'],spec['endpoint'],spec['method']) != (config['classes'],config['endpoint'],method):
            raise ValueError('Deployment differs from final fit')
        gen = driver.read(store,spec['generators'])
        if driver.read(store,gen['training_visits']) != training_ids:
            raise ValueError('Deployment roster differs from all-development refit')
        for name in [*closure,decision_id]:
            # Store.read decodes compressed JSON. Transport its exact stored bytes.
            if Path(name).name != name:
                raise ValueError('Invalid portable object name')
            payload = (store.root/name).read_bytes()
            checksum = name.split('.')[0]
            if sha(payload) != checksum:
                raise ValueError('Corrupt portable object')
            output.publish(name,payload)
            files['objects/'+name] = checksum
        model_paths.add('objects/'+model)
        decision_sources.add('objects/'+decision_id)
        procedures.append(dict(endpoint=spec['endpoint'],arm=arm,method=method,class_order=spec['classes'],
            model_sha256=model.split('.')[0],generator_sha256=spec['generators'].split('.')[0],
            requires_curvature=requires_curvature,rule=rule,fit=fit_id,decision_source_sha256=decision_id.split('.')[0]))

    for arm in driver.ARMS:
        results = body['arms'][arm]
        if set(results['outputs']) != set(names[arm]):
            raise ValueError('Incomplete core final procedures')
        objects = set(body['objects'])
        rules = development.decision_rules(primary,objects,results['decision_rules'],arm,names[arm],
                                            fold,train,children,config)
        for method, fit_id in results['outputs'].items():
            fit = driver.read(primary,fit_id)
            model = fit['deployment']
            spec = driver.read(primary,model)
            gen = driver.read(primary,spec['generators'])
            if (fit_id not in objects or fit_id not in results['fits'] or fit['split'] != fold or 'probabilities' in fit
                    or fit['name'] != method or fit['arm'] != arm or spec['arm'] != arm
                    or fit['classes'] != config['classes'] or fit['endpoint'] != config['endpoint']
                    or fit['train_groups'] != sorted(set(train.group)) or spec['head'] != fit['fitted']
                    or gen['blocks'] != [b['model'] for b in fit['blocks']] or spec['generators'] != fit['generators']):
                raise ValueError('Unbound core final fit or deployment')
            add(primary,model,primary.deployment_files(model),objects,arm,method,method in ('Ricci','joint','K0'),
                dict(rule='argmax' if len(config['classes'])>2 else 'max-participant-balanced-accuracy',
                     threshold=rules[method]),fit_id,results['decision_rules'])

    for method, item in expected.items():
        entry = cmp['results'][method]
        fit = ce.record(entry['fit'])
        rule = ce.record(entry['decision'])
        ce.selected(dict(method=method,rule=rule,decision=entry['decision'],fit=entry['fit']),
                    item,fold,entry['trials'])
        model = entry['deployment']
        spec = ce.record(model)
        compact = {k:rule[k] for k in ('rule','threshold')}
        if (spec['registry'] != cmp['registration'] or spec['decision'] != compact or spec['head'] != fit['fitted']
                or ce.record(spec['generators'])['blocks'] != [b['model'] for b in fit['blocks']]):
            raise ValueError('Unbound comparator final deployment')
        for arm in item['applies_to']:
            add(cstore,model,driver.deployment_files(cstore,model),ce.objects,arm,method,
                item['representation'] in ('Ricci','joint'),compact,entry['fit'],entry['decision'])
    # Exercise the production mixed-model loader against the exported transitive files.
    # This verifies exact frozen fusion and identical shared-species descriptors.
    driver.load_deployments(dest,dict(files=files,roles=dict(models=sorted(model_paths))),procedures)
    if core.release(root,fold)[1] != parent_id or driver.release(root,fold)[3] != rid:
        raise ValueError('Development final-refit release changed')
    if (core.bundle.sha(pdir/'receipt.json') != primary_marker
            or core.bundle.sha(cstore.root/'receipt.json') != comparator_marker):
        raise ValueError('Final-refit marker changed during export')
    data = dict(endpoint=config['endpoint'],classes=config['classes'],parent_release=parent_id,comparator_release=rid,
                primary_marker=primary_marker,comparator_marker=comparator_marker,exporter_sha256=version,validator_sha256=validators,files=files,
                models=sorted(model_paths),decision_sources=sorted(decision_sources),procedures=procedures)
    Store(dest).publish('export.json',canonical_json(data))
    return dest/'export.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('primary_marker')
    parser.add_argument('comparator_marker')
    args = parser.parse_args()
    print(export(args.root,args.primary_marker,args.comparator_marker))
