# Verification record

Repository verification performed on 2026-07-14:

- Python modules compile successfully.
- All shell launch/monitor scripts pass `bash -n`.
- Editable package installation succeeds with the current dependencies.
- Unit tests: **10 passed**.
- A synthetic all-five-task end-to-end run completed through nested participant-grouped CV and generated the complete CSV, JSON, PNG, PDF and LaTeX output tree.

The synthetic run used reduced dimensions and a reduced hyperparameter grid solely to exercise the full execution path. It did not alter repository defaults. The real project data paths under `~/Real_Data` were not available in the artifact-building container, so no real-data results are bundled or claimed here. Run `bash scripts/validate_inputs.sh` on the TDA VM before launching the full analysis.
