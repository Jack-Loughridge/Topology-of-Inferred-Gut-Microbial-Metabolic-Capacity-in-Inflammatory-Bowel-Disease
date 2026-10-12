#!/usr/bin/env python3
"""Fast functional self-test for the repeated H0 Alpha-Pi package."""

from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from h0_alpha_pi_repeated_cv import (
    build_adaptive_bounds,
    check_influence_identity,
    choose_candidate,
    combine_loss,
    create_model,
    load_and_validate_manifest,
    load_h0_bundle,
    make_ce_loss,
    set_all_seeds,
)


def minimal_args() -> Namespace:
    return Namespace(
        quad_points=2,
        point_chunk=16,
        margin_between=10.0,
        lambda_ce=1000.0,
        lambda_within=1.0,
        lambda_radius=1.0,
        lambda_between=200.0,
    )


def build_small_inputs(base: Path) -> Path:
    rng = np.random.default_rng(20260720)
    pd_dir = base / "out_pds"
    pd_dir.mkdir(parents=True)
    rows = []
    labels = []
    metadata = []
    for participant_index in range(8):
        label = 0 if participant_index < 4 else 1
        condition = "nonIBD" if label == 0 else ("UC" if participant_index % 2 == 0 else "CD")
        sample_id = f"S{participant_index:02d}"
        participant_id = f"P{participant_index:02d}"
        deaths = rng.beta(2.0 + label, 6.0 - label, size=12).astype(np.float32)
        np.save(pd_dir / f"{sample_id}_H0.npy", np.column_stack([np.zeros_like(deaths), deaths]))
        rows.append(
            {
                "sample_id": sample_id,
                "participant_id": participant_id,
                "label": label,
                "label_name": "non-IBD" if label == 0 else "IBD",
                "condition": condition,
            }
        )
        labels.append({"sample_id": sample_id, "label": condition})
        metadata.append({"External ID": sample_id, "Participant ID": participant_id})

    samples = pd.DataFrame(rows)
    # Deliberately include conflicting records outside the analysis manifest.
    # These must not block a manifest-locked run.
    labels.extend([
        {"sample_id": "UNRELATED", "label": "CD"},
        {"sample_id": "UNRELATED", "label": "nonIBD"},
    ])
    metadata.extend([
        {"External ID": "UNRELATED", "Participant ID": "PX"},
        {"External ID": "UNRELATED", "Participant ID": "PY"},
    ])

    pd.DataFrame(labels).to_csv(base / "sample_labels.csv", index=False)
    pd.DataFrame(metadata).to_csv(base / "hmp2_metadata.csv", index=False)

    manifest_rows = []
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=13)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(samples)), samples["label"], samples["participant_id"]), start=1
    ):
        for role, indices in [("train", train_index), ("test", test_index)]:
            for row in samples.iloc[indices].itertuples(index=False):
                manifest_rows.append(
                    {
                        "repeat": 1,
                        "fold": fold,
                        "split_seed": 13,
                        "role": role,
                        "sample_id": row.sample_id,
                        "participant_id": row.participant_id,
                        "label": row.label,
                        "label_name": row.label_name,
                        "condition": row.condition,
                    }
                )
    manifest_path = base / "sample_split_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return manifest_path


def main() -> None:
    set_all_seeds(123, deterministic=True)
    with tempfile.TemporaryDirectory(prefix="h0_alpha_pi_fast_selftest_") as temp:
        base = Path(temp)
        manifest_path = build_small_inputs(base)
        manifest, sample_meta, n_repeats, n_splits = load_and_validate_manifest(manifest_path)
        assert n_repeats == 1 and n_splits == 2
        bundle = load_h0_bundle(
            base / "out_pds",
            base / "sample_labels.csv",
            base / "hmp2_metadata.csv",
            sample_meta,
            require_exact_pd_set=True,
        )
        assert len(bundle.sample_ids) == 8

        # Conflicts for a required sample must still be rejected.
        bad_metadata = pd.read_csv(base / "hmp2_metadata.csv")
        bad_metadata = pd.concat(
            [
                bad_metadata,
                pd.DataFrame([{"External ID": "S00", "Participant ID": "CONFLICT"}]),
            ],
            ignore_index=True,
        )
        bad_metadata_path = base / "hmp2_metadata_required_conflict.csv"
        bad_metadata.to_csv(bad_metadata_path, index=False)
        try:
            load_h0_bundle(
                base / "out_pds",
                base / "sample_labels.csv",
                bad_metadata_path,
                sample_meta,
                require_exact_pd_set=True,
            )
        except ValueError as exc:
            assert "Conflicting participant IDs for required samples" in str(exc)
        else:
            raise AssertionError("A participant conflict for a manifest sample was not rejected")

        bounds = build_adaptive_bounds(bundle.deaths_list, np.arange(8), 25.0, 1e-8)
        assert np.all(np.diff(bounds) > 0)
        assert bounds[0] == 0.0 and bounds[-1] == 1.0

        args = minimal_args()
        device = torch.device("cpu")
        model = create_model(bounds, device, args)
        ce_loss = make_ce_loss(bundle.labels, 2, device)

        # One forward/backward/optimizer step on two diagrams.
        deaths = torch.tensor([[0.05, 0.2, 0.0], [0.25, 0.6, 0.8]], dtype=torch.float32)
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        bins = torch.from_numpy(
            np.clip(np.searchsorted(bounds, deaths.numpy(), side="right") - 1, 0, len(bounds) - 2).astype(np.int64)
        )
        y = torch.tensor([0, 1], dtype=torch.long)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        z, logits = model(deaths, bins, mask)
        ce = ce_loss(logits, y)
        _, parts = model.alpha_pi_loss(z, y, args.margin_between)
        loss = combine_loss(
            ce,
            parts,
            args.lambda_ce,
            args.lambda_within,
            args.lambda_radius,
            args.lambda_between,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()

        identity_error = check_influence_identity(model, bounds, device)
        assert identity_error < 1e-5, identity_error

        # The final tie-break must prefer fewer intervals, not the denser model.
        selected = choose_candidate(
            [
                {
                    "percentile_step": 0.3,
                    "n_intervals": 300,
                    "best_epoch": 10,
                    "best_epoch_at_limit": False,
                    "validation_balanced_accuracy": 0.7,
                    "validation_macro_f1": 0.7,
                    "validation_roc_auc": 0.75,
                },
                {
                    "percentile_step": 0.7,
                    "n_intervals": 140,
                    "best_epoch": 10,
                    "best_epoch_at_limit": False,
                    "validation_balanced_accuracy": 0.7,
                    "validation_macro_f1": 0.7,
                    "validation_roc_auc": 0.75,
                },
            ]
        )
        assert selected["n_intervals"] == 140

    print("SELF-TEST PASSED: manifest-scoped metadata validation, relevant-conflict rejection, exact input alignment, adaptive bounds, training gradients, influence identity and sparse tie-break all succeeded.")


if __name__ == "__main__":
    main()
