# Source inventory for compatibility review

This is a path and source-text inventory, not proof of output provenance.

## h0_descriptive_names

No matching source path in this checkout.

## later_representation_audit_names

No matching source path in this checkout.

## all_task_external_followup_names

No matching source path in this checkout.

## canonical_interpretation

- `analysis/interpretation/build_mapped_ricci_interpretation_tables.py`
- `analysis/interpretation/build_ricci_reaction_interpretability.py`
- `analysis/interpretation/make_h0_ricci_joint_manuscript_outputs.py`
- `analysis/interpretation/make_ricci_ibd_cpath_manuscript_outputs.py`
- `analysis/interpretation/make_species_benchmark_manuscript_outputs.py`
- `analysis/interpretation/rebuild_process_tables_split_B_K0.py`

## locked_run_configs

- `provenance/locked_run_configs/directed_clustering_config.json`
- `provenance/locked_run_configs/edge_concentration_weight_top001_config.json`
- `provenance/locked_run_configs/h0_CD_vs_UC_run_config.json`
- `provenance/locked_run_configs/h0_ibd_run_config.json`
- `provenance/locked_run_configs/h0_nonIBD_vs_CD_run_config.json`
- `provenance/locked_run_configs/h0_nonIBD_vs_UC_run_config.json`
- `provenance/locked_run_configs/h0_three_way_nonIBD_UC_CD_run_config.json`
- `provenance/locked_run_configs/joint_orchestrator_config.json`
- `provenance/locked_run_configs/outdegree_hist_active_participant_mean_config.json`
- `provenance/locked_run_configs/path_backbone_longrange_config.json`
- `provenance/locked_run_configs/ricci_ibd_b_only_run_config.json`
- `provenance/locked_run_configs/ricci_ibd_k0_only_run_config.json`
- `provenance/locked_run_configs/ricci_ibd_run_config.json`
- `provenance/locked_run_configs/ricci_remaining_tasks_run_config.json`
- `provenance/locked_run_configs/ricci_three_binary_tasks_run_config.json`
- `provenance/locked_run_configs/sequential_routing_attack_config.json`
- `provenance/locked_run_configs/species_ibd_run_config.json`
- `provenance/locked_run_configs/species_remaining_tasks_run_config.json`
- `provenance/locked_run_configs/weight_density_participant_average_config.json`

## canonical_external

- `external_validation/serrano_gomez/scripts/apply_frozen_h0_alphapi_ensemble_external_ibd_v1.py`
- `external_validation/serrano_gomez/scripts/apply_frozen_h0_alphapi_ensemble_external_ibd_v1_host_subject.py`
- `external_validation/serrano_gomez/scripts/apply_frozen_ricci_ibd_model_expanded_v2.py`
- `external_validation/serrano_gomez/scripts/apply_frozen_ricci_ibd_model_v1.py`
- `external_validation/serrano_gomez/scripts/apply_frozen_species_benchmarks_external_ibd_last_occurrence_v1.py`
- `external_validation/serrano_gomez/scripts/apply_frozen_species_benchmarks_external_ibd_v1.py`
- `external_validation/serrano_gomez/scripts/apply_frozen_species_benchmarks_external_v2_last_occurrence_all184.py`
- `external_validation/serrano_gomez/scripts/audit_expanded_external_v2_prerequisites.py`
- `external_validation/serrano_gomez/scripts/build_expanded_external_v2.py`
- `external_validation/serrano_gomez/scripts/build_ge50_external_ec_graphs.py`
- `external_validation/serrano_gomez/scripts/build_ge50_species_proxy_graphs_033.py`
- `external_validation/serrano_gomez/scripts/check_ge50_md5.py`
- `external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py`
- `external_validation/serrano_gomez/scripts/stream_build_h0_v260.py`

## entropy_code


## ricci_path_settings

- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:34`: `d(u,v) = -log(epsilon_dist + q(u,v)) otherwise`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:81`: `epsilon_dist: float = 1e-4`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:82`: `c_single_out: float = 0.001`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:89`: `n_paths: int = 1000`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:299`: `probs[nbrs[0]] = 1.0 - cfg.c_single_out`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:417`: `# 1 - c_single_out for outdegree-one vertices.`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:462`: `- path probability falls below epsilon_dist,`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:481`: `for _ in range(cfg.n_paths):`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:501`: `if new_p < cfg.epsilon_dist:`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:524`: `"mean_steps": float(total_steps / cfg.n_paths),`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:539`: `return reach_cache.get(u, {}).get(v, 0) / float(cfg.n_paths)`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:544`: `return float(-math.log(cfg.epsilon_dist + q))`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:740`: `"n_paths": int(cfg.n_paths),`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:776`: `"n_paths": cfg.n_paths,`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:777`: `"epsilon_dist": cfg.epsilon_dist,`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:779`: `"c_single_out": cfg.c_single_out,`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:823`: `"n_paths": int(cfg.n_paths),`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:856`: `ap.add_argument("--n-paths", type=int, default=Config.n_paths)`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:861`: `ap.add_argument("--epsilon-dist", type=float, default=Config.epsilon_dist)`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:862`: `ap.add_argument("--c-single-out", type=float, default=Config.c_single_out)`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:885`: `epsilon_dist=args.epsilon_dist,`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:886`: `c_single_out=args.c_single_out,`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:890`: `n_paths=args.n_paths,`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:924`: `print(f"    n_paths:   {cfg.n_paths}")`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:926`: `print(f"    epsilon:   {cfg.epsilon_dist}")`
- `pipeline/topology/compute_ricci_faithful_pairwise_active.py:927`: `print(f"    c_single:  {cfg.c_single_out}")`

## ricci_availability

- `analysis/ricci/feature_builder/faithful_ricci_classifier.py:127`: `if not math.isfinite(x):`
- `analysis/ricci/feature_builder/faithful_ricci_classifier.py:261`: `ok = kappa.notna() & np.isfinite(kappa.to_numpy(dtype=float))`
- `analysis/ricci/feature_builder/faithful_ricci_classifier.py:313`: `ok = kappa.notna() & np.isfinite(kappa.to_numpy(dtype=float))`
- `analysis/ricci/feature_builder/faithful_ricci_classifier.py:745`: `all_vals = all_vals[np.isfinite(all_vals)]`
- `analysis/ricci/feature_builder/faithful_ricci_classifier.py:771`: `vals = vals[np.isfinite(vals)]`
- `analysis/ricci/feature_builder/faithful_ricci_classifier.py:833`: `sd = np.where(np.isfinite(sd), sd, 0.0)`
- `analysis/ricci/feature_builder/faithful_ricci_classifier.py:986`: `vals = vals[np.isfinite(vals)]`
- `analysis/ricci/feature_builder/faithful_ricci_classifier.py:1016`: `vals = vals[np.isfinite(vals)]`
- `analysis/ricci/five_task_repeated_cv/ricci_all_tasks.py:421`: `if not np.isfinite(dense).all():`
- `analysis/ricci/five_task_repeated_cv/ricci_all_tasks.py:657`: `finite = values[np.isfinite(values)]`
- `analysis/ricci/ibd_block_ablation/run_ibd_block_ablation.py:383`: `if not np.isfinite(X_block).all():`
- `analysis/ricci/ibd_block_ablation/run_ibd_block_ablation.py:761`: `if not np.isfinite(K0).all() or not np.any(K0 != 0):`
- `analysis/ricci/ibd_cpath/repeated_ricci_ibd_cpath.py:145`: `if not math.isfinite(value):`
- `analysis/ricci/ibd_cpath/repeated_ricci_ibd_cpath.py:221`: `process = str(proc_value) if pd.notna(proc_value) else "Unannotated"`
- `analysis/ricci/ibd_cpath/repeated_ricci_ibd_cpath.py:222`: `display = str(display_value) if pd.notna(display_value) else decode_edge_for_display(edge)`
- `analysis/ricci/ibd_cpath/repeated_ricci_ibd_cpath.py:854`: `finite = array[np.isfinite(array)]`
- `external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py:305`: `if not math.isfinite(v) or v < 0:`
- `external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py:439`: `if not math.isfinite(w):`
- `external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py:467`: `d = d[np.isfinite(d)]`
- `external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py:654`: `frac = float(vals.notna().mean())`
- `external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py:667`: `if not math.isfinite(float(val)) if pd.notna(val) else True:`
- `external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py:914`: `if not math.isfinite(w) or w < 0.0 or w > 1.0 + 1e-12:`
- `external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py:1502`: `if not math.isfinite(kval):`
- `external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1/external_structural_validation.py:1513`: `if not math.isfinite(kval):`
