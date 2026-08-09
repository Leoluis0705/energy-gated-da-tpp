# Structural-Group Gate Held-Out Feasibility Experiment

## 1. Objective

Test whether replacing the current element-system redundancy definition with a label-blind structural-cluster definition prevents the target losses observed in the Mn- and Mg-anchored interval tasks.

This is a held-out-seed feasibility experiment, not a preregistered independent confirmation of the overall scientific hypothesis. The revised selector is motivated by the completed 180-run diagnostic. All revised definitions are frozen before any held-out result is generated.

## 2. Decision Supported

The experiment decides whether to:

1. proceed with a structure-aware revision of Energy-Gated DA-TPP;
2. report an efficiency-versus-coverage tradeoff without claiming superior target recovery; or
3. stop revising the selector and retain the observed applicability boundary.

## 3. Time and Compute Limit

- Maximum GPU-server window: 5 hours.
- Observed server throughput: 12.75 completed jobs per hour.
- Planned jobs: 50.
- Estimated GPU wall time: 3.92 hours.
- Code preparation, tests, manifest generation, and upload occur locally before the GPU queue starts.
- Result download and statistical analysis occur after GPU completion and do not require the GPU server to remain open.

## 4. Frozen Data Scope

- Candidate pool: the existing frozen 640-candidate Li–M–O pool.
- Candidate IDs, CIF files, candidate order, oracle table, checkpoint, and proxy-label source remain unchanged.
- Mn-anchored proxy interval: `[-2.1, -1.9] eV atom^-1`.
- Mg-anchored proxy interval: `[-2.3, -2.1] eV atom^-1`.
- Both tasks continue to use the complete 640-candidate pool.
- Existing DFT values are not used by acquisition.
- No VASP or other new DFT calculation is launched.

## 5. Held-Out Initial Sets

- Evaluation seeds: `111, 112, 113, 114, 115`.
- Each seed receives four initial labeled candidates.
- Initial candidates are selected label-blind by sorting all eligible candidate IDs by `SHA256("structural_gate_holdout_v1:<seed>:<candidate_id>")` and taking the first four unique IDs.
- The same initial set and order are shared across all five methods within a task and seed.
- The same seed-specific initial set is used for the Mn- and Mg-anchored tasks so that task differences do not arise from different starting IDs.
- The generated initial-set CSV is frozen and hashed before any GPU run.
- The initial-set audit must confirm five distinct set hashes and exact within-seed equality across methods.

## 6. Structural Group Definition

- Source field: `structure_matcher_cluster` in the frozen candidate-pool audit table.
- The field is derived from CIF geometry and does not use target labels, DFT success, proxy energy, method results, or acquisition history.
- Each candidate maps to exactly one structural group.
- Current inventory: 148 groups, 84 singleton groups, and maximum group size 41.
- The formal map contains exactly two columns: `candidate_id,group_key`.
- Candidate IDs must match the 640-pool ID set exactly; duplicates, missing IDs, or extra IDs are fatal errors.
- The group-map SHA-256 is recorded in every revised run configuration.

## 7. Compared Methods

Five methods are evaluated on both tasks and all five held-out seeds:

1. **Predicted-Target Greedy**  
   Select the four candidates with the highest interval-hit probability.

2. **Legacy Element-System Gate**  
   Retain the existing implementation and element-system group key. This is the same-seed mechanism control.

3. **Structural-Group Gate**  
   Retain the existing gate thresholds and diversity score, changing only the group key from element system to structural cluster.

4. **Structural-Group Gate with Quality Safeguard**  
   Construct the structural-diversity batch as in method 3, then compare its summed interval-hit probability with the direct Greedy batch from the same round. Accept the structural batch only when

   `sum(P_hit_structural) >= 0.95 * sum(P_hit_direct)`.

   Otherwise select the direct Greedy batch. The 5% loss tolerance is frozen before execution and is not adjusted after results are observed.

5. **Gradient-Norm Hybrid**  
   Retain the current corrected implementation as the strongest observed non-Gate baseline.

All methods share the same model, training schedule, MC-dropout masks within a seed and round, batch size, budget, candidate pool, initial set, and refit protocol.

## 8. Fixed Protocol

- Batch size: 4.
- Query budget: 320.
- Acquisition rounds after the four-point initial set: 79.
- MC-dropout passes: 30.
- Current `M0`, `G0`, `alpha`, `beta`, and `gamma` remain unchanged.
- The current candidate prefilter multiplier remains unchanged.
- No interval, parameter, seed, or metric may be changed after the first held-out job starts.
- No method may read oracle labels before selection.

## 9. Required Implementation Isolation

The change is divided into two independently testable units:

### 9.1 Group-map input

The existing selector receives the frozen structural group map through the formal group-map interface. This path changes no scoring or route logic.

### 9.2 Quality safeguard

The safeguard is a post-selection batch comparator. It receives the direct and proposed structural batches plus their `P_hit` values and returns either the structural batch or the direct batch. It does not access oracle labels or DFT information.

This separation ensures that any difference between methods 2 and 3 is attributable to group representation, while any difference between methods 3 and 4 is attributable to the safeguard.

## 10. Tests Required Before Upload

1. Structural map has exactly 640 unique candidate IDs and no target-related columns.
2. Structural map is unchanged when oracle labels are permuted.
3. Greedy behavior is byte-for-byte unchanged.
4. Legacy Gate behavior is byte-for-byte unchanged on an existing fixture.
5. Structural Gate differs from Legacy Gate only through the supplied group keys.
6. Safeguard accepts a structural batch at exactly the 95% boundary.
7. Safeguard rejects a structural batch below the 95% boundary and returns the direct batch in the original deterministic order.
8. Zero-sum direct `P_hit` accepts the structural batch when its sum is also zero.
9. Same seed and round produce identical results on repeated execution.
10. All five methods within a seed load the identical frozen initial set.
11. Output paths include a new protocol version and cannot overwrite the completed 180-run results.

No GPU job may start unless all targeted and relevant regression tests pass.

## 11. Metrics

### Primary feasibility metric

- Paired normalized AUTC difference relative to Greedy for each revised Gate.

### Early retrieval metrics

- Recovery count and rate at 80, 160, 240, and 320 queries.
- Queries per recovered target.

### Mechanism metrics

- Direct and correction round counts.
- Number of replaced batch positions.
- Per-round and cumulative correction target gain, calculated after acquisition for analysis only.
- Sum and mean `P_hit` loss relative to the direct batch.
- Number of safeguard fallbacks.

### Diversity metrics

- All-candidate structural clusters covered at each checkpoint.
- Target structural clusters covered at each checkpoint.
- Group repetition rate and largest-group fraction.

### Hidden audit

- Post-selection expected DFT-evaluable target count and score coverage.
- These remain predicted audit values and are not treated as observed DFT outcomes.

## 12. Feasibility Decision Rules

The decision is based on the five held-out seeds and is descriptive; no strong significance claim is made from five seeds.

### Strong go

At least one revised Gate satisfies all of the following on both tasks:

- mean Gate-minus-Greedy AUTC is positive;
- mean target recovery at 80 queries is not lower than Greedy;
- mean target structural-cluster coverage at 80 queries is not lower than Greedy; and
- cumulative correction target gain is non-negative.

### Conditional go

At least one revised Gate satisfies all of the following:

- mean Gate-minus-Greedy AUTC is at least `-0.005` on both tasks;
- mean target recovery difference at 80 queries is at least `-1` on both tasks;
- mean target structural-cluster coverage increases by at least one cluster in at least one task and does not decrease in the other; and
- no seed has Gate-minus-Greedy AUTC below `-0.02`.

This outcome supports a coverage-efficiency tradeoff claim, not superiority in target recovery.

### Stop

Stop algorithm revision if both revised Gates fail the Conditional-go rule and each revised Gate also meets at least one of these failure conditions:

- mean Gate-minus-Greedy AUTC is below `-0.01` in at least one task; or
- cumulative correction target gain is materially negative in at least one task.

No threshold, interval, or task is changed in response to a stop result.

## 13. Reporting Boundary

- The revised design may be described as motivated by preliminary diagnostics of group representation.
- The group map and safeguard are frozen before held-out seed evaluation.
- The experiment must not be described as preregistered or independent of the earlier task-level findings.
- Earlier diagnostic results remain fully reported as group-representation sensitivity evidence.
- All five methods and all five seeds are reported regardless of which method wins.
- Expected DFT evaluability is never described as observed DFT success.

## 14. Server Execution Flow

1. Complete code and tests locally.
2. Produce a source/config SHA-256 manifest.
3. Upload a new staging directory; do not modify the previous staging or result directory.
4. Run one CPU-only selector smoke test and one short GPU integration test.
5. Submit the 50-job manifest with resumable statuses: `PENDING`, `RUNNING`, `DONE`, `FAILED`, or `CANCELLED`.
6. Preserve commands, logs, environment, start/end times, exit codes, output hashes, and GPU assignment.
7. Continue independent jobs after an isolated failure.
8. Stop a method configuration after three consecutive identical failures and preserve the error evidence.
9. Download analysis/core archives immediately after completion.
10. Verify local archive hashes against server hashes before server shutdown.

## 15. Outputs

- Frozen structural group map and SHA-256.
- Frozen held-out initial-set CSV and SHA-256.
- Protocol and job manifests.
- Per-run summaries, histories, score tables, mode traces, configs, and logs.
- Paired five-seed metric table.
- Group-representation and safeguard ablation table.
- Target recovery, target-cluster coverage, and correction-loss figures.
- Chinese feasibility report with one of: `STRONG_GO`, `CONDITIONAL_GO`, or `STOP`.
- Updated claim-boundary note.

## 16. Non-Goals

- No new DFT.
- No new target interval.
- No removal of Greedy or Gradient-Norm Hybrid.
- No further parameter, threshold, group-map, or metric selection using outcomes from seeds 101–110 or 111–115 after this design is frozen. The diagnostic role of seeds 101–110 remains disclosed.
- No attempt to guarantee a positive result.
- No manuscript rewrite before the feasibility decision is available.
