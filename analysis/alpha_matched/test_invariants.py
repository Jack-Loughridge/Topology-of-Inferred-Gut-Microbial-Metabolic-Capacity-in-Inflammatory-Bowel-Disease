import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from .cache import CurvatureCache,FIELDS
from .partitions import FrozenVectors,RelativeReference,fusion_input
from .support import sample_support


class Invariants(unittest.TestCase):
    def test_cache(self):
        pairs=np.array([[0,1]],dtype='<i8')
        key=dict.fromkeys(FIELDS,'fixture')
        key.update(tolerance=dict.fromkeys(('first_hit_error','transport_error','marginal_error','curvature_error'),1e-7),
                   active_pairs_digest=hashlib.sha256(pairs.tobytes()).hexdigest())
        with TemporaryDirectory() as folder:
            c=CurvatureCache(folder,key)
            cert=dict.fromkeys(key['tolerance'],0.)
            with self.assertRaises(ValueError):
                c.save(np.array([np.nan]),pairs,cert)
            self.assertIsNone(c.read(pairs))
            c.save(np.array([0.]),pairs,cert)
            self.assertEqual(c.read(pairs)[0],0.)
            for field in FIELDS:
                alternate=CurvatureCache(folder,{**key,field:'changed'})
                self.assertIsNone(alternate.read(pairs))
            receipt=c.folder/'receipt.json'
            msg=json.loads(receipt.read_bytes())
            msg['record']['certificate']['curvature_error']=1e-4
            receipt.write_text(json.dumps(msg))
            with self.assertRaises(ValueError):
                c.read(pairs)

    def test_relative_reference(self):
        a=RelativeReference.fit([[.25,0,0],[1,1,0],[1,0,0]],['1','1','2'],list('abc'),list('ijk'))
        expected=[.25**.25,1,(1+.25**.25)/2]
        np.testing.assert_allclose(np.exp(a.log_reference),expected)
        np.testing.assert_array_equal(a.unseen,[False,False,True])
        empty=RelativeReference.fit([[0,0]],['1'],['1'],['a','b'])
        np.testing.assert_array_equal(empty.log_reference,[0,0])
        self.assertNotEqual(empty.partition,a.partition)

    def test_fusion(self):
        b=[FrozenVectors(np.eye(2),('1','2'),('a','b'),'RAW','fit','gen'+s,s) for s in ('H0','Ricci')]
        data=fusion_input(*b)
        np.testing.assert_array_equal(data[:,:2],b[0].values)
        np.testing.assert_array_equal(data[:,2:],b[1].values)
        data[:]=0
        self.assertEqual(b[0].values[0,0],1)
        with self.assertRaises(ValueError):
            b[0].values.setflags(write=True)

    def test_reject_stale_parent_fit(self):
        from .nesting import Procedure
        import pandas as pd
        data=pd.DataFrame({'person':['p1','p2']},index=['s1','s2'])
        for arm in ('RAW','RELATIVE'):
            for block in ('H0','Ricci'):
                class Stale:
                    def transform(self,keys):
                        return FrozenVectors(np.ones((len(keys),1)),keys,('x',),arm,'parent','g',block)
                proc=Procedure(data,{},lambda *args:Stale(),{},[],None)
                with self.assertRaises(ValueError):
                    proc.build(block,{},data.iloc[:1],data.iloc[1:])

    def test_source_bin_tie(self):
        from .h0 import Image
        for support in ('.125','.01','.05','.07','.4'):
            d=np.array([np.log(np.longdouble(support))])
            image=Image.initial([d],25)
            reloaded=Image(image.bounds,image.bandwidth,image.alpha,
                           tuple(json.loads(json.dumps(image.source_cuts))))
            np.testing.assert_array_equal(np.flatnonzero(reloaded.basis(d).sum(axis=1)>0),[1])

    def test_dense_percentile_grid(self):
        from .h0 import Image
        s=np.linspace(-1,1,1001,dtype=np.longdouble)
        im=Image.initial([-s],.4)
        cut=np.array(im.source_cuts,dtype=np.longdouble)
        np.testing.assert_array_equal(np.searchsorted(cut,s[4:1000:4],side='right'),np.arange(1,250))

    def test_training_species_axis(self):
        import pandas as pd
        from types import SimpleNamespace
        from .comparators import Baseline
        train=pd.DataFrame({'person':['p','q']},index=['x','y'])
        g=SimpleNamespace(lookup={'x':0,'y':1,'z':2},species_abundance=csr_matrix([[1,0],[.5,0],[.25,.75]]))
        fit=Baseline(g,'species',train)
        np.testing.assert_array_equal(fit.species_axis,[0])
        np.testing.assert_allclose(fit.transform(['z']),[[-2.]])

    def test_activity(self):
        subnormal=np.nextafter(0.,1.)
        raw,verts=sample_support([subnormal,0],csr_matrix([[1,0],[0,1]]),csr_matrix([[1,1,1],[0,0,1]]))
        np.testing.assert_array_equal(raw.indices,[0])
        np.testing.assert_array_equal(verts,[0,1,2])

if __name__=='__main__':
    unittest.main()
