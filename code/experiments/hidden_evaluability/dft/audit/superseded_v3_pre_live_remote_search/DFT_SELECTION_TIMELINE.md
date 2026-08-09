# DFT Selection Timeline

## Scope and set identity

The unified manifest contains 20 unique candidates: eight pilot candidates and twelve new candidates. The three main-text candidates are flags within the new-candidate set; they are not a third candidate set.

## Evidence-backed timeline

1. The eight-pilot selection table records query rank, iteration, an ALIGNN interval score, structural checks, and diversity terms. No contemporaneous result-blind freeze record or independently verifiable freeze timestamp was found. Their `freeze_timestamp` is therefore blank and `result_known_at_freeze=unknown`.
2. The pilot `query_rank/iteration` values do not agree with the retained v14 illustrative Gate/Greedy histories for the same IDs. They remain in dedicated `pilot_selection_*` columns and are not relabelled as `Gate_round` or `Greedy_round`.
3. The twelve-new prelaunch manifest has filesystem UTC mtime `2026-07-12T16:19:06.310516+00:00`. The adjacent prelaunch status contains twelve `pending/awaiting_user_review` rows, and the protocol audit explicitly records launch authorization as false. This supports `result_known_at_freeze=false`, while the timestamp itself is explicitly only filesystem metadata.
4. A later launch-authorization note records authorization on 2026-07-13. The first retained per-candidate completion update follows the prelaunch filesystem timestamp. No more precise signed authorization time was found.
5. The final new12 results identify three main-text candidates: job_120 (Cr), job_214 (Cr), and job_044 (Mn). Their selection reasons are retained verbatim from the main-text shortlist CSV.

## Magnetic-sampling boundary

For candidates with strict magnetic reruns, the evidence covers two tested magnetic initializations. A selected state is described only as the lower-energy configuration among the two tested initializations.

## Pilot relaxation-artifact search

All 8 pilot rows are marked `original_relaxation_artifact_unavailable`: the retained top-level OUTCAR/OSZICAR files are static outputs (`NSW=0`), while relaxation evidence is limited to `relax.log` plus the final relaxed structure.

| Scope | Location | Result |
|---|---|---|
| archive_candidate_evidence | `materials_maintext_candidates/candidate_001..008` | retained OUTCAR/OSZICAR are static (NSW=0); eight relax.log files retained |
| archive_server_mirror | `tmp/dft_candidate_inventory/remote/vasp_inputs` | server mirror contains the same post-static candidate outputs, not pre-static relaxation artifacts |
| archive_bundles | `outputs/dft_candidate_selection/dft_candidate_selection_bundle.tar.gz;outputs/dft_candidate_selection/final_dft_values_bundle.tar.gz;outputs/dft_candidate_selection/gga_u_values_bundle.tar.gz` | input bundle has no outputs; result bundles retain summaries/logs but no pilot relaxation OUTCAR/OSZICAR |
| historical_repository | `D:/CGCNN` | no pilot VASP OUTCAR/OSZICAR or candidate scheduler records found |
| historical_remote_server | `/root/dft_limo/outputs/dft_candidate_selection/vasp_inputs` | live historical server directory not accessible in this workspace |
| recycle_bin_current_user | `D:/$Recycle.Bin/current-user-SID` | no relevant payload or original-path metadata hit |
| recycle_bin_other_sids | `D:/$Recycle.Bin/other-SIDs` | access denied for three other SID directories; not searched |
| baidu_sync | `D:/BaiduSyncdisk` | no relevant file found |
| onedrive | `C:/Users/leoluis0705/OneDrive` | no relevant file found |
| wps_sync | `C:/Users/leoluis0705/WPS Cloud Files; C:/Users/leoluis0705/WPSDrive` | no relevant file found |

A reconstructed rerun may be useful for protocol verification or to regenerate stage-separated outputs, but it would be a new calculation and must never be represented as the original historical relaxation. It is not required to interpret the retained static energies, but the absent original relaxation force/stress trajectory remains a declared provenance limitation.

## Main-text dependency check

The current v33 comparison PDF uses the three new12 main-text candidates rather than the eight pilot candidates for its main DFT shortlist. The pilot calculations nevertheless support historical DFT tables and methodological claims elsewhere in the archive, so their missing relaxation-stage artifacts remain relevant to full reproducibility.

## Primary source hashes

- `outputs/dft_candidate_selection/selected_dft_candidates.csv`: SHA-256 `2b03042921df1ff7be001c51c5f9423012cfbf80e3a777b35aa2291d1228e944`
- `materials_maintext_candidates/candidate_summary.csv`: SHA-256 `fbf4bcdfcf7a16fd9ccb85dcc20d14905ee520b88342d837b143d942190e905a`
- `new12_dft_prelaunch/NEW12_DFT_CANDIDATE_MANIFEST.csv`: SHA-256 `7266d52bdff54e1ded6d12cb2ef04a3ae48424362f27f3ea1a648788c2d25b85`
- `new12_dft_prelaunch/NEW12_DFT_STATUS.csv`: SHA-256 `66d65a1fb637a34dfc1583d97c68f2d29516be0e483984303c5092009c39c0f2`
- `new12_dft_prelaunch/NEW12_DFT_PROTOCOL_AUDIT.md`: SHA-256 `0f2e60f313c0f9244e6de31eaa64bbff6a4d5a8832414c3da7e72000ca20b9f0`
- `new12_dft_screening_snapshot/extracted/LAUNCH_AUTHORIZATION.md`: SHA-256 `8357b63f7a829d67ce783cdd31049ecf20627f365edd776fd48a51dac60c91f5`
- `new12_dft_final/NEW12_DFT_RESULTS.csv`: SHA-256 `d95c21ed08c7f2c4fdae4057c6e04b8f11020edb94eb83af75e45fdd6983c8f5`
- `new12_dft_final/NEW12_DFT_MAIN_TEXT_CANDIDATES.csv`: SHA-256 `b4bff1fc8ee4dfde9aeec33755f862c4d430b047423c2b20883cf129a286534f`
