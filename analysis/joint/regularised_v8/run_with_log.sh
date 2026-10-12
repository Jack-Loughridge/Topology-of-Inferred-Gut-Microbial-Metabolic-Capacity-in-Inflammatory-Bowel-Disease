#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$HOME/Real_Data"
mkdir -p "$BASE/logs"
exec >> "$BASE/logs/joint_ibd_regularised_v8.log" 2>&1
printf '\n=== V8 START/RESUME %s ===\n' "$(date -Is)"
PYTHON=$(python3 - "$BASE/H0_Ricci_JointSparse_RepeatedCV_AllTasks/orchestrator_config.json" <<'PY'
import json,sys
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text())
p=Path(cfg['joint_repo']).expanduser()/'.venv/bin/python'
if not p.is_file():raise SystemExit('Original virtual environment is missing. No installation attempted.')
# Do not resolve this interpreter symlink to the system Python.
print(str(p))
PY
)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$HERE"
"$PYTHON" -u verify_package.py
if ! "$PYTHON" -u run_tests.py; then
    echo 'SELF-TEST FAILED. No production fitting started.'
    exit 2
fi
set +e
"$PYTHON" -u run.py
STATUS=$?
set -e
echo "=== V8 EXIT status=$STATUS $(date -Is) ==="
exit "$STATUS"
