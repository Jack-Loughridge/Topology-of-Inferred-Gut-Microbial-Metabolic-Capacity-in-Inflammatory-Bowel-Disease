"""Production numerical contracts on invented graphs."""
from unittest import TestCase
from unittest.mock import patch
import numpy as np

from . import geometry


class GeometryTests(TestCase):
    def test_biological_underflow(self):
        pairs=np.array([[0,1],[0,2],[2,1],[1,3]])
        logs=np.array([-1000,0,1000,0],dtype=np.longdouble)
        g=geometry.compute(5,pairs,logs,family="power",parameter=4,rho=.99)
        self.assertTrue(np.array_equal(g.pairs,pairs))
        self.assertTrue(np.isfinite(g.curvature).all())
        self.assertLessEqual(g.error["curvature_error"],1e-7)

    def test_forced_rational_repair(self):
        pairs=np.array([[0,1],[0,2],[1,2],[2,0]])
        logs=np.log(np.array([1,2,3,4],dtype=np.longdouble))
        kw=dict(family="power",parameter=.25,rho=.99)
        regular=geometry.compute(3,pairs,logs,**kw)
        with patch.object(geometry,"_certify_fast",side_effect=ArithmeticError):
            repaired=geometry.compute(3,pairs,logs,**kw)
        self.assertEqual(repaired.error["rational_repair_edges"],4)
        self.assertTrue(np.all(np.abs(regular.curvature-repaired.curvature)<=
                               regular.error["curvature_error"]+repaired.error["curvature_error"]))
        self.assertLessEqual(repaired.error["curvature_error"],1e-7)

    def test_integer_fallback_independently_of_two_float_proposals(self):
        pairs=np.array([[0,1],[0,2],[1,2],[2,0]])
        logs=np.zeros(4,dtype=np.longdouble)
        kw=dict(family="linear",parameter=None,rho=.95)
        regular=geometry.compute(3,pairs,logs,**kw)
        with patch.object(geometry,"_certify_fast",side_effect=ArithmeticError), patch.object(
            geometry.ref,"transport",side_effect=ArithmeticError):
            repaired=geometry.compute(3,pairs,logs,**kw)
        self.assertEqual(repaired.error["rational_repair_edges"],4)
        self.assertTrue(np.all(np.abs(regular.curvature-repaired.curvature)<=
                               regular.error["curvature_error"]+repaired.error["curvature_error"]))

    def test_high_degree_analytic_sink_star_under_bounded_cache(self):
        n=157
        pairs=np.array([(0,j) for j in range(1,n)])
        g=geometry.compute(n,pairs,np.zeros(n-1),family="uniform",parameter=None,rho=.99,
                           memory_bytes=256*n+8192)
        m=-np.log(1e-4)
        d=-np.log(1e-4+(1-1e-4)*.99/(n-1))
        # Centre and leaves each carry 1/n; exactly one leaf is the target.
        exact=1-(d+(n-2)*m)/(n*d)
        self.assertEqual(len(g.curvature),n-1)
        self.assertLessEqual(max(abs(g.curvature-exact)),g.error["curvature_error"]+1e-14)

    def test_incorrect_inputs(self):
        for edges in ([(0.5,1)],[(0,1),(0,1)],[(1,1)]):
            with self.assertRaises(ValueError):
                geometry.compute(2,edges,[0]*len(edges),family="linear",parameter=None,rho=.9)
        with self.assertRaises(ValueError):
            geometry.compute(2,[],[],family="power",parameter=-1,rho=.9)
