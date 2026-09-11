from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest

from joint_repeated_cv.config import RunConfig


@pytest.fixture
def small_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        joint_repo=str(tmp_path / "core"),
        ibd_split_manifest=str(tmp_path / "ibd_manifest.csv"),
        h0_results_dir=str(tmp_path / "h0_results"),
        h0_pd_dir=str(tmp_path / "pds"),
        ricci_original_dir=str(tmp_path / "ricci_reference"),
        ricci_feature_dir=str(tmp_path / "ricci_features"),
        output_dir=str(tmp_path / "output"),
        expected_repeats=2,
        expected_folds=5,
        start_repeat=1,
        end_repeat=2,
    )
