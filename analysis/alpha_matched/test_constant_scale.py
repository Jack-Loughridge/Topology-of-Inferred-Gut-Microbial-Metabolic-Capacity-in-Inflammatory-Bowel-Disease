import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .artifacts import Store
from .comparators import Baseline
from .features import Factory, Graphs
from .partitions import canonical_json, digest, fitted_location_scale

class ConstantScale(unittest.TestCase):
    def test_constant_and_nearly_constant_coordinates(self):
        v=np.array([[.1,0],[.1,1e-12],[.1,2e-12]])
        mean,sd=fitted_location_scale(v,[1,1,1])
        self.assertEqual(mean[0],.1)
        self.assertEqual(sd[0],1)
        self.assertGreater(sd[1],0)
        self.assertLess(sd[1],1e-12)
        np.testing.assert_array_equal((v[:,0]-mean[0])/sd[0],np.zeros(3))

    def test_tiny_and_adjacent_variation(self):
        adjacent = np.nextafter(.1, 1.)
        values = np.array([[0, .1], [1e-200, adjacent], [2e-200, adjacent]])
        _, scale = fitted_location_scale(values, [1, 1, 1])
        np.testing.assert_allclose(scale, [np.sqrt(2/3)*1e-200,
                                           (adjacent-.1)*np.sqrt(2)/3], atol=0, rtol=1e-14)

    def test_direct_saved_transfer_and_species_imputation(self):
        samples=tuple('abcde')
        graph=Graphs(csr_matrix([[.1],[.1],[.1],[.1],[.2]]),np.array([[0,1]]),
                     tuple(np.arange(2) for _ in samples),samples,
                     csr_matrix([[.1],[.1],[.1],[0],[.2]]),'catalogue','mapping',False,('s',),
                     np.array([[1],[1],[1],[0],[1]],bool))
        train=pd.DataFrame({'person':list('ABCD')},index=list('abcd'))
        species=Baseline(graph,'species',train)
        self.assertEqual(species.species_impute[0],.1)
        self.assertEqual(species.scale[0],1)
        self.assertEqual(species.mean[0],.1)
        # Three people with equal weights reproduce the original 0.1 averaging defect.
        train=train.iloc[:3]
        raw=Baseline(graph,'direct-raw',train)
        self.assertEqual(raw.transform(['e'])[0,0],.1)
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp))
            factory=Factory(graph,'RAW',Path(tmp)/'cache',{})
            spec=dict(name='direct-raw',arm='RAW',partition=raw.partition,reference=None,
                      axis=digest(canonical_json(graph.axis)),catalogue=graph.catalogue_digest,
                      mapping=graph.mapping_digest,features=store.record(['edge']),
                      state={'mean':store.array(raw.mean),'scale':store.array(raw.scale)})
            restored=store.transform(store.record(spec),factory,['e'],['s'])
            np.testing.assert_array_equal(restored.values,raw.transform(['e']))
