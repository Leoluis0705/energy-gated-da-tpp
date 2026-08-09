# Structural-Group Gate Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a frozen 50-job held-out GPU experiment that tests whether label-blind structural groups and a fixed 5% quality safeguard prevent the Mn/Mg target losses caused by element-system grouping.

**Architecture:** Extend the existing frozen interval-runner rather than create a new training pipeline. Keep method behavior isolated in the selector, generate immutable structural-group and initial-set assets locally, route each method through a method-specific frozen protocol, then run the existing audited queue and analyze all five held-out seeds together.

**Tech Stack:** Python 3, PyTorch/CGCNN, pandas, NumPy, pytest, Matplotlib, existing audited GPU queue, SSH/SFTP, SHA-256 manifests.

## Global Constraints

- Candidate pool, CIFs, candidate ordering, oracle table, checkpoint, proxy labels, target intervals, model hyperparameters, batch size 4, budget 320, and MC-dropout passes 30 remain unchanged.
- Mn interval is `[-2.1, -1.9] eV atom^-1`; Mg interval is `[-2.3, -2.1] eV atom^-1`.
- Held-out seeds are exactly `111, 112, 113, 114, 115`.
- Methods are exactly Greedy, Legacy Element-System Gate, Structural-Group Gate, Structural-Group Gate with fixed 95% quality safeguard, and Gradient-Norm Hybrid.
- No oracle label, DFT outcome, or hidden evaluability score may enter acquisition.
- No VASP or new DFT job may start.
- No existing result or staging directory may be overwritten.
- GPU-server occupancy must not exceed five hours; the 50-job estimate is 3.92 hours at the previously observed throughput.
- All results are retained even when revised Gate methods lose to Greedy.
- This workspace has no Git repository. Do not initialize Git without user authorization; create SHA-256 checkpoint manifests in place of commits.

---

### Task 1: Add a distinct frozen protocol cohort

**Files:**
- Modify: `experiments/active_learning/experiments/reproducibility/formal_protocol.py`
- Modify: `experiments/active_learning/experiments/reproducibility/protocol_artifacts.py`
- Modify: `experiments/active_learning/experiments/reproducibility/run_paired_dataset_job.py`
- Test: `experiments/active_learning/tests/test_structural_gate_feasibility_protocol.py`

**Interfaces:**
- Produces: protocol phase `structural_group_feasibility`, method tokens `structural_group_gate` and `structural_group_gate_q95`, group mode `structure_matcher_cluster`, and protocol version `egdatpp_structgate_feas_v1`.
- Consumes: existing `FormalProtocol`, `METHOD_SPECS`, `score_artifact_path`, and `trace_artifact_path` behavior.

- [ ] **Step 1: Write failing protocol tests**

```python
def test_structural_feasibility_protocol_accepts_only_heldout_cohort(tmp_path):
    protocol = load_formal_protocol(write_structural_protocol(tmp_path))
    base = runner.dataset_configs()["limo"]
    protocol.resolve_dataset_config(base, method="structural_group_gate", seed=111)
    protocol.resolve_dataset_config(base, method="structural_group_gate_q95", seed=115)
    with pytest.raises(FormalProtocolError, match="not allowed"):
        protocol.resolve_dataset_config(base, method="structural_group_gate", seed=110)


def test_new_protocol_version_is_collision_safe():
    old = score_artifact_path("out", method_name="m", iteration=1)
    new = score_artifact_path(
        "out", method_name="m", iteration=1,
        protocol_version="egdatpp_structgate_feas_v1",
    )
    assert old != new
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_structural_gate_feasibility_protocol.py -v`

Expected: failure because the phase, methods, group mode, and protocol version are not yet supported.

- [ ] **Step 3: Add the minimal protocol vocabulary**

Add:

```python
PHASE_COHORTS["structural_group_feasibility"] = range(111, 116)
FORMAL_METHODS.update({"structural_group_gate", "structural_group_gate_q95"})
GROUP_KEY_MODES.add("structure_matcher_cluster")
SUPPORTED_PROTOCOL_VERSIONS = {
    "egdatpp_psfix_v1",
    "egdatpp_structgate_feas_v1",
}
```

Change `selector_protocol_arguments` to accept an explicit validated `protocol_version` and pass `formal_protocol.protocol_version` into each selector invocation. Preserve `egdatpp_psfix_v1` as the default for every existing call.

- [ ] **Step 4: Add runner method specifications**

```python
"structural_group_gate": {
    "display_name": "Structural-Group Gate",
    "selection_method_name": "structural_group_gate",
    "ablation_mode": "full",
    "quality_safeguard_fraction": None,
},
"structural_group_gate_q95": {
    "display_name": "Structural-Group Gate + Q95",
    "selection_method_name": "structural_group_gate_q95",
    "ablation_mode": "full",
    "quality_safeguard_fraction": 0.95,
},
```

Pass `--quality-safeguard-fraction 0.95` only for the safeguarded method.

- [ ] **Step 5: Run protocol and existing interval tests**

Run: `python -m pytest tests/test_structural_gate_feasibility_protocol.py tests/test_interval_gpu_protocol.py -v`

Expected: all tests pass, including the previous 180-job manifest test.

- [ ] **Step 6: Record a source checkpoint**

Generate SHA-256 rows for the three modified files and the new test in `provenance/structural_gate_feasibility_source_checkpoints.csv` with checkpoint name `task_1_protocol`.

### Task 2: Implement the fixed quality safeguard

**Files:**
- Modify: `experiments/active_learning/active_learning_energy_gate_ablation.py`
- Test: `experiments/active_learning/tests/test_structural_gate_quality_safeguard.py`

**Interfaces:**
- Produces: `apply_quality_safeguard(direct_idx, proposed_idx, p_hit, min_fraction)` returning `(selected_idx, fallback_used, direct_sum, proposed_sum)`.
- Consumes: the direct top-`b` ordering and structural-diversity proposal already computed in the selector.

- [ ] **Step 1: Write boundary tests**

```python
def test_q95_accepts_exact_boundary():
    chosen, fallback, direct_sum, proposed_sum = apply_quality_safeguard(
        [0, 1], [2, 3], np.array([0.5, 0.5, 0.50, 0.45]), 0.95
    )
    assert chosen == [2, 3]
    assert fallback is False
    assert (direct_sum, proposed_sum) == pytest.approx((1.0, 0.95))


def test_q95_returns_direct_order_below_boundary():
    chosen, fallback, _, _ = apply_quality_safeguard(
        [1, 0], [2, 3], np.array([0.5, 0.5, 0.49, 0.45]), 0.95
    )
    assert chosen == [1, 0]
    assert fallback is True


def test_q95_accepts_zero_sum_proposal_when_direct_is_zero():
    chosen, fallback, _, _ = apply_quality_safeguard(
        [0, 1], [2, 3], np.zeros(4), 0.95
    )
    assert chosen == [2, 3]
    assert fallback is False
```

- [ ] **Step 2: Verify the new tests fail**

Run: `python -m pytest tests/test_structural_gate_quality_safeguard.py -v`

Expected: import failure for `apply_quality_safeguard`.

- [ ] **Step 3: Implement the pure comparator**

```python
def apply_quality_safeguard(direct_idx, proposed_idx, p_hit, min_fraction):
    direct = [int(value) for value in direct_idx]
    proposed = [int(value) for value in proposed_idx]
    values = np.asarray(p_hit, dtype=float)
    direct_sum = float(values[direct].sum())
    proposed_sum = float(values[proposed].sum())
    accepted = proposed_sum + EPS >= float(min_fraction) * direct_sum
    return (proposed if accepted else direct), (not accepted), direct_sum, proposed_sum
```

Validate `0 < min_fraction <= 1`. Apply it only after the structural-diversity batch is built and only when the gate selected the correction route.

- [ ] **Step 4: Add audit fields without reading oracle labels**

Add the following columns to the trace row:

```python
"quality_safeguard_fraction": args.quality_safeguard_fraction,
"quality_safeguard_fallback": int(quality_fallback),
"direct_batch_p_hit_sum": direct_p_hit_sum,
"proposed_batch_p_hit_sum": proposed_p_hit_sum,
"selected_batch_p_hit_sum": float(p_hit[selected_idx].sum()),
```

- [ ] **Step 5: Run selector regression tests**

Run: `python -m pytest tests/test_structural_gate_quality_safeguard.py tests/test_interval_gpu_protocol.py -v`

Expected: Q95 boundary tests pass and Greedy/legacy baseline tests remain unchanged.

- [ ] **Step 6: Record a source checkpoint**

Append hashes for the selector and test with checkpoint name `task_2_safeguard`.

### Task 3: Generate and freeze label-blind assets

**Files:**
- Create: `experiments/active_learning/analysis/build_structural_gate_feasibility_assets.py`
- Create: `experiments/active_learning/tests/test_build_structural_gate_feasibility_assets.py`
- Generate: `experiments/active_learning/configs/structural_group_feasibility/structure_matcher_group_map.csv`
- Generate: `experiments/active_learning/configs/structural_group_feasibility/heldout_initial_sets.csv`
- Generate: `experiments/active_learning/configs/structural_group_feasibility/*.json`

**Interfaces:**
- Consumes: `experiments/hidden_evaluability/inputs/three_system/candidate_pool_master.csv` and frozen 640-pool IDs.
- Produces: `build_group_map(frame) -> DataFrame`, `deterministic_initial_ids(pool_ids, seed) -> list[str]`, three method-family protocols, and two task overlays.

- [ ] **Step 1: Write asset-invariant tests**

```python
def test_group_map_is_exact_and_label_blind(pool_master):
    result = build_group_map(pool_master)
    assert result.columns.tolist() == ["candidate_id", "group_key"]
    assert result.shape == (640, 2)
    assert result["candidate_id"].is_unique
    assert result["group_key"].nunique() == 148
    assert not any("target" in column.lower() for column in result.columns)


def test_initial_sets_are_distinct_and_reproducible(pool_ids):
    first = {seed: deterministic_initial_ids(pool_ids, seed) for seed in range(111, 116)}
    second = {seed: deterministic_initial_ids(pool_ids, seed) for seed in range(111, 116)}
    assert first == second
    assert len({tuple(ids) for ids in first.values()}) == 5
    assert all(len(ids) == len(set(ids)) == 4 for ids in first.values())
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/test_build_structural_gate_feasibility_assets.py -v`

- [ ] **Step 3: Implement deterministic initial sets**

```python
INITIAL_NAMESPACE = "structural_gate_holdout_v1"

def deterministic_initial_ids(pool_ids, seed):
    def key(candidate_id):
        payload = f"{INITIAL_NAMESPACE}:{int(seed)}:{candidate_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    return sorted((str(value) for value in pool_ids), key=key)[:4]
```

Store `initial_set_sha256` using the existing `candidate_order_digest(sorted(ids))` convention required by `load_initial_set_ids`.

- [ ] **Step 4: Generate method-specific protocols**

Create:

- `legacy_protocol.json`: element-system mode; allows Greedy, legacy Gate, and Gradient-Norm Hybrid.
- `structural_protocol.json`: structure-cluster map; allows Structural-Group Gate.
- `structural_q95_protocol.json`: same map; allows Structural-Group Gate + Q95.
- `mn_task.json` and `mg_task.json`: unchanged intervals, batch 4, budget 320, checkpoints 80/160/240/320, and new held-out initial-set path.

- [ ] **Step 5: Generate assets and hashes**

Run:

```text
python analysis/build_structural_gate_feasibility_assets.py \
  --pool-master ../../hidden_evaluability/inputs/three_system/candidate_pool_master.csv \
  --output-dir configs/structural_group_feasibility
```

Expected: 640 group rows, 20 initial-set rows, five distinct initial hashes, 148 groups, 84 singleton groups, maximum group size 41.

- [ ] **Step 6: Run asset tests against generated files**

Run: `python -m pytest tests/test_build_structural_gate_feasibility_assets.py -v`

- [ ] **Step 7: Freeze assets**

Write `configs/structural_group_feasibility/SHA256SUMS.csv` and do not regenerate assets after the first held-out job starts.

### Task 4: Build the collision-safe 50-job manifest

**Files:**
- Create: `experiments/active_learning/analysis/build_structural_gate_feasibility_manifest.py`
- Create: `experiments/active_learning/tests/test_build_structural_gate_feasibility_manifest.py`
- Generate: `experiments/active_learning/configs/structural_group_feasibility/jobs_manifest.csv`

**Interfaces:**
- Consumes: the three frozen protocols, two task overlays, two available GPU IDs confirmed by server preflight, and Python/project/execution paths.
- Produces: exactly 50 audited queue rows with method-specific protocol paths and noncolliding result directories.

- [ ] **Step 1: Write the manifest-grid test**

```python
def test_manifest_has_complete_50_job_grid():
    rows = build_manifest_rows(
        project_root="/project", execution_root="/execution",
        python="/venv/bin/python", gpu_ids=(0, 1),
    )
    assert len(rows) == 50
    assert len({row["job_id"] for row in rows}) == 50
    assert len({row["output_path"] for row in rows}) == 50
    assert {row["seed"] for row in rows} == {str(value) for value in range(111, 116)}
    assert {row["method"] for row in rows} == {
        "predicted_target_greedy", "energy_gated_da_tpp",
        "structural_group_gate", "structural_group_gate_q95",
        "gradient_norm_hybrid",
    }
```

- [ ] **Step 2: Verify the manifest test fails**

Run: `python -m pytest tests/test_build_structural_gate_feasibility_manifest.py -v`

- [ ] **Step 3: Implement the exact grid**

Use two tasks × five methods × five seeds. Map each method to the appropriate frozen protocol. Set cohort to `structural_group_feasibility_v1`; route outputs only under a new execution root; initialize every status to `PENDING`; assign GPU IDs round-robin from the preflight-confirmed list.

- [ ] **Step 4: Generate and validate the manifest**

Run the builder, then execute a validation that rejects duplicate job IDs, duplicate output paths, missing commands, a job count other than 50, any seed outside 111–115, or any result path containing the previous `formal_w0p2` directory.

- [ ] **Step 5: Freeze the manifest**

Append the manifest hash and all referenced config hashes to `configs/structural_group_feasibility/SHA256SUMS.csv`.

### Task 5: Implement paired feasibility analysis

**Files:**
- Create: `experiments/active_learning/analysis/analyze_structural_gate_feasibility.py`
- Create: `experiments/active_learning/tests/test_analyze_structural_gate_feasibility.py`

**Interfaces:**
- Consumes: completed run directories, structural group map, frozen oracle, and hidden evaluability score table.
- Produces: paired per-seed metrics, decision status, figures, Chinese report, and claim boundary.

- [ ] **Step 1: Write metric and decision-rule tests**

```python
def test_strong_go_requires_both_tasks():
    rows = synthetic_rows(mn_delta=0.01, mg_delta=0.01, r80_delta=0, cluster_delta=0, gain=0)
    assert classify_decision(rows) == "STRONG_GO"


def test_conditional_go_is_not_superiority():
    rows = synthetic_rows(mn_delta=-0.004, mg_delta=-0.003, r80_delta=-1,
                          cluster_delta=1, gain=-0.1, worst_seed=-0.01)
    assert classify_decision(rows) == "CONDITIONAL_GO"


def test_stop_keeps_losing_tasks_visible():
    rows = synthetic_rows(mn_delta=-0.02, mg_delta=-0.015, r80_delta=-3,
                          cluster_delta=0, gain=-4)
    assert classify_decision(rows) == "STOP"
```

- [ ] **Step 2: Verify analysis tests fail**

Run: `python -m pytest tests/test_analyze_structural_gate_feasibility.py -v`

- [ ] **Step 3: Implement metric extraction**

For every task/method/seed, calculate normalized AUTC, recovery at 80/160/240/320, queries per recovered target, correction rounds, replaced positions, post-selection correction target gain, direct/proposed/selected `P_hit` sums, safeguard fallbacks, all-structure cluster coverage, target-cluster coverage, group repetition, and hidden expected DFT evaluability.

- [ ] **Step 4: Implement paired comparisons**

Pair every method with Greedy by task and seed. Reject analysis if any pair is missing, if fewer than five seeds exist, or if initial-set hashes differ within a task/seed.

- [ ] **Step 5: Produce fixed outputs**

Write:

- `per_seed_metrics.csv`
- `paired_gate_vs_greedy.csv`
- `mechanism_metrics.csv`
- `decision.json`
- `STRUCTURAL_GATE_FEASIBILITY_REPORT_ZH.md`
- `STRUCTURAL_GATE_CLAIM_BOUNDARY.md`
- `figure_recovery.pdf/png`
- `figure_autc.pdf/png`
- `figure_target_cluster_coverage.pdf/png`
- `figure_correction_loss.pdf/png`

- [ ] **Step 6: Run analysis tests**

Run: `python -m pytest tests/test_analyze_structural_gate_feasibility.py -v`

### Task 6: Complete local verification and package staging

**Files:**
- Generate: `provenance/structural_gate_feasibility_local_manifest.csv`
- Generate: `provenance/structural_gate_feasibility_source.tar.gz`

**Interfaces:**
- Consumes: modified source, frozen assets, tests, and manifests.
- Produces: a hash-verifiable archive that contains no prior result trees and no credentials.

- [ ] **Step 1: Run focused tests**

Run:

```text
python -m pytest \
  tests/test_structural_gate_feasibility_protocol.py \
  tests/test_structural_gate_quality_safeguard.py \
  tests/test_build_structural_gate_feasibility_assets.py \
  tests/test_build_structural_gate_feasibility_manifest.py \
  tests/test_analyze_structural_gate_feasibility.py -q
```

- [ ] **Step 2: Run relevant regression tests**

Run: `python -m pytest tests/test_interval_gpu_protocol.py tests/test_formal_protocol.py tests/test_mc_dropout_seed_policy.py -q`

- [ ] **Step 3: Run the complete active-learning test suite**

Run: `python -m pytest tests -q`

Expected: no failed test. Existing unrelated skips are recorded, not converted to passes.

- [ ] **Step 4: Build the staging archive**

Include active source, `cgcnn`, frozen 640-pool assets required by the runner, new configs, and manifests. Exclude all previous result directories, caches, credentials, DFT outputs, and paper files.

- [ ] **Step 5: Verify local hashes**

Recompute every manifest entry from the archive extraction and require exact equality before upload.

### Task 7: Preflight, smoke test, and launch on the GPU server

**Files:**
- Remote create: a new timestamped staging directory under the existing server workspace.
- Remote create: `SERVER_INVENTORY.json`, `SMOKE_RESULTS.csv`, and the live audited manifest.

**Interfaces:**
- Consumes: SSH endpoint supplied by the user and the verified staging archive.
- Produces: a validated server environment and resumable 50-job queue.

- [ ] **Step 1: Perform read-only inventory**

Record hostname, GPU count/model/free memory, driver/CUDA, CPU, RAM, disk space, Python path, environment path, PyTorch/CUDA visibility, and currently running GPU processes. Do not kill or alter existing processes.

- [ ] **Step 2: Enforce the five-hour feasibility check**

Require two usable GPUs or a measured smoke throughput that forecasts completion within five hours. If the forecast exceeds five hours, stop before the 50-job launch and report the measured estimate.

- [ ] **Step 3: Upload to a new staging directory**

Upload the archive and local SHA manifest, extract once, and compare remote hashes with local hashes. Never place the supplied password in a remote file, shell history, manifest, or log.

- [ ] **Step 4: Run one CPU selector smoke test**

Exercise structural group-map loading and Q95 accept/fallback paths without training. Require exit code 0 and expected trace columns.

- [ ] **Step 5: Run two short GPU integration jobs**

Run Mn/seed111 Greedy and Structural-Q95 with the smoke task budget. Require both jobs to finish, use the same initial-set hash, use different output paths, produce nonempty histories, and expose CUDA usage.

- [ ] **Step 6: Launch the formal audited queue**

Start all 50 jobs using the existing resumable queue. Use one job per GPU at a time. Preserve `PENDING/RUNNING/DONE/FAILED/CANCELLED`, commands, environment, timestamps, exit code, output path, and hashes.

- [ ] **Step 7: Monitor without blocking communication**

Poll status at short intervals, report meaningful state changes, preserve isolated failures, and stop a method configuration only after three consecutive identical failures.

### Task 8: Copy back, verify, and classify the result

**Files:**
- Local create: `work/gpu_structural_gate_feasibility_20260804_results/`
- Local create: all analysis outputs from Task 5.

**Interfaces:**
- Consumes: completed remote execution tree and remote SHA manifest.
- Produces: complete local evidence, a decision status, and a concise Chinese interpretation.

- [ ] **Step 1: Freeze remote outputs**

After the queue reaches a terminal state, generate a remote file inventory and SHA-256 manifest. Archive configs/logs/core tabular artifacts separately from large checkpoints.

- [ ] **Step 2: Download all evidence**

Copy both archives and the unhashed manifest to the new local result directory. Do not copy only summary tables.

- [ ] **Step 3: Verify local copyback**

Require exact remote/local hash equality and confirm all 50 jobs have a terminal status. List every missing or failed job before analysis.

- [ ] **Step 4: Run paired analysis**

Execute `analysis/analyze_structural_gate_feasibility.py` against the verified local tree. Do not change rules, thresholds, intervals, or methods.

- [ ] **Step 5: Report the decision**

Return exactly one of `STRONG_GO`, `CONDITIONAL_GO`, or `STOP`, followed by Mn and Mg method tables, Gate-versus-Greedy paired differences, mechanism explanation, hidden-evaluability audit clearly marked as predicted, and the resulting manuscript claim boundary.

- [ ] **Step 6: Preserve server credentials safely**

Do not repeat the password in reports. Advise the user to rotate the password after all evidence has been verified locally.
