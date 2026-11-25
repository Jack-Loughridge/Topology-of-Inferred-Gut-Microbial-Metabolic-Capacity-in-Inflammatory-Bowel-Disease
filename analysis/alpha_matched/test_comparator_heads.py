"""Synthetic local comparator training, participant weights and portable replay."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .artifacts import Store
from . import comparator_heads as heads


PARAMS = {
    'logistic': dict(C=.75, max_iter=200),
    'random_forest': dict(n_estimators=9, max_depth=3, min_samples_leaf=1),
    'xgboost': dict(n_estimators=5, max_depth=2, learning_rate=.2, reg_lambda=.5),
    'tabm': dict(k=3, n_blocks=1, d_block=12, dropout=.1, epochs=3, batch_size=7,
                 learning_rate=.003, weight_decay=.01),
}


def fixture(count):
    person = np.repeat(np.arange(18), [1, 3, 2] * 6)
    x = np.random.default_rng(46).normal(size=(len(person), 5))
    y = person % count
    x[:, 0] += y * .5
    return x, y, person, np.tile([1., 1/3, 1/3, 1/3, .5, .5], 6)


class ComparatorHeads(unittest.TestCase):
    def test_native_probabilities_and_portability(self):
        from xgboost import XGBClassifier
        for count in (2, 3):
            x, y, people, weights = fixture(count)
            held = (x * .91)[::-3]
            for family, params in PARAMS.items():
                with self.subTest(count=count, family=family), tempfile.TemporaryDirectory() as tmp:
                    source, dest = Store(Path(tmp) / 'source'), Store(Path(tmp) / 'export')
                    cfg = dict(family=family, seed=19, params=params)
                    model_id = heads.fit(source, x, y * 10 + 4, people, cfg)
                    spec = json.loads(source.read(model_id))
                    self.assertEqual(spec['classes'], sorted(set(y * 10 + 4)))
                    self.assertEqual(heads.fit(source, x, y * 10 + 4, people, cfg), model_id)
                    p = heads.predict(source, model_id, held)
                    self.assertEqual(heads.export(source, model_id, dest), model_id)
                    self.assertEqual(set(heads.dependencies(source, model_id)), {f.name for f in dest.root.iterdir()})
                    np.testing.assert_array_equal(p, heads.predict(Store(dest.root), model_id, held))
                    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
                    if family == 'tabm':
                        native = self.tabm_reference(x, y, weights, held, cfg)
                    else:
                        cls = {'logistic': LogisticRegression, 'random_forest': RandomForestClassifier,
                               'xgboost': XGBClassifier}[family]
                        extras = dict(tree_method='hist', device='cpu') if family == 'xgboost' else {}
                        clf = cls(**params, random_state=19, n_jobs=1, **extras)
                        clf.fit(x, y, sample_weight=weights)
                        native = clf.predict_proba(held)
                    np.testing.assert_allclose(p, native, rtol=3e-6, atol=1e-7)
                    dependency = next(n for n in heads.dependencies(dest, model_id) if n != model_id)
                    (dest.root / dependency).chmod(0o600)
                    (dest.root / dependency).write_bytes(b'corrupt')
                    with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                        heads.predict(dest, model_id, held)

    @staticmethod
    def tabm_reference(x, y, weights, held, cfg):
        import torch
        from tabm import TabM
        import torch.nn.functional as F
        p = cfg['params']
        with heads._torch_cpu(cfg['seed']):
            net = TabM.make(n_num_features=x.shape[1], d_out=len(set(y)), k=p['k'], n_blocks=p['n_blocks'],
                            d_block=p['d_block'], dropout=p['dropout'])
            optimizer = torch.optim.AdamW(net.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
            # Derive each visit coefficient directly from the known fixture roster.
            # Both full and short batches use the same per-visit coefficient.
            coeff = torch.tensor(weights * len(y) / (18 * p['batch_size']), dtype=torch.float32)
            for _ in range(p['epochs']):
                for ids in torch.randperm(len(y)).split(p['batch_size']):
                    logits = net(torch.tensor(x[ids.numpy()], dtype=torch.float32))
                    logp = F.log_softmax(logits, dim=2)
                    selected = logp[torch.arange(len(ids)), :, torch.tensor(y[ids.numpy()])]
                    loss = -(selected.mean(1) * coeff[ids]).sum()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                net.eval()
                return net(torch.tensor(held.copy(), dtype=torch.float32)).softmax(2).mean(1).numpy()

    def test_reject_misalignment_and_unregistered_settings(self):
        x, y, people, _ = fixture(2)
        cfg = dict(family='logistic', seed=19, params=PARAMS['logistic'])
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            with self.assertRaisesRegex(ValueError, 'indices'):
                heads.fit(store, pd.DataFrame(x), pd.Series(y).iloc[::-1], people, cfg)
            y[2] = 1 - y[1]
            with self.assertRaisesRegex(ValueError, 'consistent'):
                heads.fit(store, x, y, people, cfg)
            with self.assertRaisesRegex(ValueError, 'parameters'):
                heads.fit(store, x, y, people, {**cfg, 'params': {**cfg['params'], 'other': 0}})
