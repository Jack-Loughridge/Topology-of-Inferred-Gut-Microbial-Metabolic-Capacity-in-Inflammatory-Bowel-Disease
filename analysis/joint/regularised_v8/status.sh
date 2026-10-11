#!/usr/bin/env bash
set -euo pipefail
OUT="$HOME/Real_Data/H0_Ricci_IBD_Regularised_v8"
python3 - "$OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);print('Output:',out)
p=out/'STATUS.json'
print(p.read_text() if p.is_file() else 'No status yet. Check the startup log.')
for arm in ('joint','ricci'):
 print(arm,'completed:',len(list(out.glob(f'jobs/repeat_*/fold_*/{arm}/COMPLETE.json'))))
files=sorted(out.glob('jobs/repeat_*/fold_*/STATUS.json'),key=lambda x:x.stat().st_mtime)
for p in files[-3:]:print(str(p.parent.relative_to(out)),p.read_text())
PY
