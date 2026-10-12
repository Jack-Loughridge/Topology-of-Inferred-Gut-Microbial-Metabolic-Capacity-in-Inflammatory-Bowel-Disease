#!/usr/bin/env python3
import os,sys,time,unittest
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
if __name__=='__main__':
    start=time.monotonic();suite=unittest.defaultTestLoader.discover(str(root/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print({'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
           'skips':len(result.skipped),'seconds':time.monotonic()-start},flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)
