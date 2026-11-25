"""Local heads for a single registered candidate and training partition."""
from contextlib import contextmanager
from importlib.metadata import version
import json
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .artifacts import sha
from .nesting import weights as participant_weights

FIELDS = {
    'logistic': {'C', 'max_iter'},
    'random_forest': {'n_estimators', 'max_depth', 'min_samples_leaf'},
    'xgboost': {'n_estimators', 'max_depth', 'learning_rate', 'reg_lambda'},
    'tabm': {'k', 'n_blocks', 'd_block', 'dropout', 'epochs', 'batch_size', 'learning_rate', 'weight_decay'},
}
INTEGER = {'max_iter', 'n_estimators', 'max_depth', 'min_samples_leaf', 'k', 'n_blocks',
           'd_block', 'epochs', 'batch_size'}


def _config(cfg):
    if set(cfg) != {'family', 'seed', 'params'} or cfg['family'] not in FIELDS:
        raise ValueError('Expected a registered head family, seed and explicit params')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**31:
        raise ValueError('Invalid CPU seed')
    params = cfg['params']
    if not isinstance(params, dict) or set(params) != FIELDS[cfg['family']]:
        raise ValueError('Missing or unexpected candidate parameters')
    for key, value in params.items():
        if key == 'max_depth' and value is None and cfg['family'] == 'random_forest':
            continue
        if key in INTEGER:
            ok = type(value) is int and value >= 1
        else:
            ok = isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)
            if ok:
                ok = 0 <= value < 1 if key == 'dropout' else (
                    value >= 0 if key in {'reg_lambda', 'weight_decay'} else value > 0)
        if not ok:
            raise ValueError('Invalid parameter ' + key)


def _versions(family):
    packages = {'logistic': ['scikit-learn'], 'random_forest': ['scikit-learn'],
                'tabm': ['tabm', 'torch', 'rtdl-num-embeddings'], 'xgboost': ['xgboost']}[family]
    versions = {p: version(p) for p in ['numpy', *packages]}
    for p, expected in [('tabm', '0.0.3'), ('xgboost', '3.2.0')]:
        if p in versions and versions[p] != expected:
            raise ValueError('Unqualified package version for ' + p)
    return versions


def _matrix(x):
    a = np.asarray(x)
    if a.ndim != 2 or a.shape[1] == 0 or a.dtype.kind not in 'biuf' or not np.isfinite(a).all():
        raise ValueError('Expected finite two-dimensional numeric representation')
    return a


def _training(x, y, people):
    indices = [v.index for v in (x, y, people) if isinstance(v, (pd.DataFrame, pd.Series))]
    if indices and (not indices[0].is_unique or any(not i.equals(indices[0]) for i in indices[1:])):
        raise ValueError('Unaligned training visit indices')
    a, labels, people = _matrix(x), np.asarray(y), np.asarray(people)
    if (labels.shape != (len(a),) or labels.dtype.kind not in 'iu' or people.shape != labels.shape
            or pd.isna(people).any()):
        raise ValueError('Invalid labels or participant roster')
    classes, encoded = np.unique(labels, return_inverse=True)
    groups, _ = pd.factorize(people)
    if len(classes) < 2 or pd.Series(labels).groupby(groups).nunique().max() != 1:
        raise ValueError('Need consistent participant labels and at least two classes')
    return a, encoded, participant_weights(groups), classes.tolist()


@contextmanager
def _torch_cpu(seed):
    import torch
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.default_generator.manual_seed(seed)
            yield
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn)
            torch.set_num_threads(previous)


def _tabm(n_features, n_classes, params):
    from tabm import TabM
    return TabM.make(n_num_features=n_features, d_out=n_classes,
                     **{k: params[k] for k in ('k', 'n_blocks', 'd_block', 'dropout')})


def _tensor(a):
    import torch
    with np.errstate(over='ignore'):
        data = np.ascontiguousarray(a, dtype=np.float32)
    values = torch.from_numpy(data.copy() if not data.flags.writeable else data)
    if not torch.isfinite(values).all():
        raise ValueError('Representation outside finite float32 range')
    return values


def _train_tabm(a, labels, weights, cfg, n_classes):
    import torch
    from torch.nn.functional import cross_entropy
    p = cfg['params']
    model = _tabm(a.shape[1], n_classes, p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    batch = min(p['batch_size'], len(a))
    # A fixed denominator keeps a visit's coefficient independent of its batch
    # composition, including the final short batch of each epoch.
    denominator = batch * weights.sum() / len(a)
    x, y, w = _tensor(a), torch.tensor(labels, dtype=torch.long), _tensor(weights / denominator)
    for _ in range(p['epochs']):
        for rows in torch.randperm(len(a)).split(batch):
            scores = model(x[rows])
            target = y[rows, None].expand(-1, p['k'])
            losses = cross_entropy(scores.transpose(1, 2), target, reduction='none').mean(1)
            loss = (w[rows] * losses).sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite TabM training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def fit(store, x, y, people, config):
    """Fit one head on the supplied representation without refitting preprocessing."""
    _config(config)
    a, labels, weights, classes = _training(x, y, people)
    family, seed, params = config['family'], config['seed'], config['params']
    versions = _versions(family)
    with threadpool_limits(limits=1):
        if family == 'tabm':
            with _torch_cpu(seed):
                model = _train_tabm(a, labels, weights, config, len(classes))
                state = {k: store.array(v.detach().cpu().numpy()) for k, v in model.state_dict().items()}
        elif family == 'xgboost':
            from xgboost import XGBClassifier
            model = XGBClassifier(**params, random_state=seed, n_jobs=1, tree_method='hist', device='cpu')
            model.fit(a, labels, sample_weight=weights)
            state = store.put(bytes(model.get_booster().save_raw(raw_format='ubj')), '.ubj')
        else:
            if family == 'logistic':
                model = LogisticRegression(**params, solver='lbfgs', random_state=seed)
            else:
                model = RandomForestClassifier(**params, random_state=seed, n_jobs=1)
            model.fit(a, labels, sample_weight=weights)
            state = store.head(model)
    return store.record(dict(schema=1, config=config, classes=classes, n_features=a.shape[1],
                             versions=versions, state=state))


def _descriptor(store, name):
    spec = json.loads(store.read(name))
    _config(spec['config'])
    if spec['schema'] != 1 or _versions(spec['config']['family']) != spec['versions']:
        raise ValueError('Incompatible comparator runtime or descriptor')
    return spec


def predict(store, model_id, x):
    """Return local probabilities in the descriptor's training-derived class order."""
    spec = _descriptor(store, model_id)
    a = _matrix(x)
    if a.shape[1] != spec['n_features']:
        raise ValueError('Comparator feature width changed')
    cfg, state = spec['config'], spec['state']
    count = len(spec['classes'])
    if not len(a):
        return np.empty((0, count))
    with threadpool_limits(limits=1):
        if cfg['family'] == 'tabm':
            import torch
            with _torch_cpu(cfg['seed']), torch.inference_mode():
                model = _tabm(a.shape[1], count, cfg['params'])
                model.load_state_dict({k: torch.from_numpy(store.load_array(ref)) for k, ref in state.items()})
                model.eval()
                outputs = []
                for start in range(0, len(a), cfg['params']['batch_size']):
                    rows = a[start:start + cfg['params']['batch_size']]
                    outputs.append(model(_tensor(rows)).softmax(-1).mean(1).numpy())
                result = np.concatenate(outputs)
        elif cfg['family'] == 'xgboost':
            from xgboost import Booster
            booster = Booster(params={'nthread': 1, 'device': 'cpu'})
            booster.load_model(bytearray(store.read(state)))
            booster.set_param({'nthread': 1, 'device': 'cpu'})
            result = booster.inplace_predict(a)
            if count == 2:
                result = np.column_stack((1 - result, result))
        else:
            result = store.predict(state, a)
    if (result.shape != (len(a), count) or not np.isfinite(result).all() or (result < 0).any()
            or (result > 1).any() or not np.allclose(result.sum(1), 1, atol=2e-6, rtol=0)):
        raise ValueError('Invalid restored comparator probabilities')
    if result.dtype == np.float32:
        result = result.astype(np.float64)
        result /= result.sum(axis=1, keepdims=True)
    return result


def dependencies(store, model_id):
    spec = _descriptor(store, model_id)
    family, state = spec['config']['family'], spec['state']
    names = {model_id}
    if family == 'tabm':
        names.update(state.values())
    else:
        names.add(state)
        if family in ('logistic', 'random_forest'):
            head = json.loads(store.read(state))
            if family == 'logistic':
                names.update([head['coef'], head['intercept']])
            else:
                for tree in head['trees']:
                    names.update(tree.values())
    for name in names:
        store.read(name)
    return sorted(names)


def export(store, model_id, destination_store):
    names = dependencies(store, model_id)
    names.remove(model_id)
    for name in [*names, model_id]:
        data = (store.root / name).read_bytes()
        if sha(data) != name.split('.')[0]:
            raise ValueError('Comparator object changed during export')
        destination_store.publish(name, data)
        destination_store.objects.add(name)
    return model_id
