#!/usr/bin/env bash
set -euo pipefail
REAL_DATA_DIR="${REAL_DATA_DIR:-$HOME/Real_Data}"
CORE_DIR="${CORE_DIR:-$REAL_DATA_DIR/h0_ricci_joint_sparse}"
ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$REAL_DATA_DIR/h0_ricci_joint_classifier_github}"
[[ -d "$CORE_DIR" ]] || { echo "[error] Core directory missing: $CORE_DIR" >&2; exit 1; }
rm -rf "$DEST"
mkdir -p "$DEST/core" "$DEST/orchestration"
rsync -a \
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='build/' --exclude='dist/' --exclude='*.egg-info/' --exclude='*.pyc' --exclude='*.log' \
  "$CORE_DIR/" "$DEST/core/"
rsync -a \
  --exclude='.git/' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.pyc' \
  "$ORCH_DIR/" "$DEST/orchestration/"
cat > "$DEST/README.md" <<'EOF'
# Joint H0 Alpha-Pi and Ricci classifier

- `core/`: sparse joint WKPI + faithful Ricci solver.
- `orchestration/`: locked five-task execution, validation, aggregation and stability analysis.

The manuscript run uses repetition 1 only for all five tasks (25 outer folds), while the orchestration CLI supports larger repeated ranges.
EOF
cat > "$DEST/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
*.log
*.npy
*.npz
*.joblib
runs/
outputs/
results/
checkpoints/
EOF
echo "Prepared GitHub repository at: $DEST"
find "$DEST" -type f -size +20M -print
