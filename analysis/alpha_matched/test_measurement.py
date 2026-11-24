import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .features import Graphs
from .comparators import Baseline
from .gm_bundle import measurement_for_panel


class Measurement(unittest.TestCase):
    def test_observed_zero_and_unreported_have_different_vector(self):
        # A has two visits, and B and C have one each. Only C's first taxon is
        # unreported. The held-out visit cannot affect the pooled training mean.
        abundance=csr_matrix([[0,.2,0],[.4,0,0],[.8,.1,0],[0,.1,0],[0,.1,.5]])
        observed=np.array([[1,1,0],[1,1,0],[1,1,0],[0,1,0],[0,1,1]],dtype=bool)
        panel=Graphs(csr_matrix((5,0)),np.empty((0,2),dtype=int),
                     tuple(np.arange(0) for _ in range(5)),tuple('abcde'),abundance,'c','m',False,
                     ('x','y','z'),observed)
        train=pd.DataFrame({'person':['A','A','B','C']},index=list('abcd'))
        b=Baseline(panel,'species',train)
        np.testing.assert_array_equal(b.species_axis,[0,1])
        # E[x | emitted] = ((0+.4)/2+.8)/2 = .5. C's missing cell has no
        # contribution. Its two-block vector differs from the observed zero.
        self.assertAlmostEqual(b.species_impute[0],.5)
        np.testing.assert_array_equal(b.values(['a','e'])[:,[0,2]],[[0,0],[.5,1]])

    def test_positive_outside_native_roster_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            Graphs(csr_matrix((1,0)),np.empty((0,2),dtype=int),(np.arange(0),),('a',),
                   csr_matrix([[.1]]),'c','m',False,('species',),np.array([[False]]))

    def test_mixed_measurement_contract(self):
        spec={'key':'sample','unreported_lineages_by_release':{'release':1}}
        axes={'source_lineage':['x','y']}
        config={'panel_key':'sample','mapping_sha256':'a'*64,'source_species_key':'source_lineage',
                'coverage_rule':'positive reported support','unlisted_is_measured_nondetection':False}
        with self.assertRaisesRegex(ValueError,'Unresolved native'):
            measurement_for_panel(spec,axes,config)
        result=measurement_for_panel(spec,axes,{**config,'preserve_unreported_mask':True})
        self.assertFalse(result[-1])
        self.assertEqual(result[0],['x','y'])


if __name__=='__main__':
    unittest.main()
