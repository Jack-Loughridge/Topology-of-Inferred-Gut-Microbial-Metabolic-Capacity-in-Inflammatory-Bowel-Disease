import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from .cache import CurvatureCache,FIELDS


class RawNumericalReuse(unittest.TestCase):
    def test_only_exact_fixed_raw_operator_shares_bytes(self):
        e=np.array([[1,2]],dtype='<i8')
        values=np.array([-.0])
        cert=dict.fromkeys(('first_hit_error','transport_error','marginal_error','curvature_error'),0.)
        identity=dict.fromkeys(FIELDS,'stable')
        identity.update(arm='RAW',reference_digest='identity',training_people_digest='fold-a',rho=.9,
                        active_pairs_digest=hashlib.sha256(e.tobytes()).hexdigest(),
                        tolerance=dict.fromkeys(cert,1e-7))
        with TemporaryDirectory() as tmp:
            d=Path(tmp)
            cache=CurvatureCache(d,identity)
            raw=CurvatureCache(d/'.raw-numerics',{**identity,'training_people_digest':'fixed-RAW-numerical-operator'})
            raw.save(values,e,cert)
            linked=cache.bind_raw_calculation(raw,e)
            self.assertEqual(linked.tobytes(),values.tobytes())
            self.assertEqual(cache.read(e).tobytes(),values.tobytes())
            self.assertNotEqual(cache.folder.name,raw.folder.name)
            self.assertTrue(next(cache.folder.glob('*.npy')).samefile(next(raw.folder.glob('*.npy'))))
            for changes in [dict(rho=.95),dict(arm='RELATIVE'),dict(reference_digest='learned')]:
                with self.assertRaises(ValueError):
                    CurvatureCache(d,{**identity,**changes}).bind_raw_calculation(raw,e)
