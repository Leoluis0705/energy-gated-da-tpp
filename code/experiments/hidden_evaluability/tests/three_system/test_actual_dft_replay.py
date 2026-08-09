import pandas as pd

from analysis.three_system.retrospective_actual_dft import build_actual_replay


def test_actual_replay_uses_observed_dft_and_policy_rounds():
    labels = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "dft_evaluable": [1, 0, 1],
            "dft_formation_energy_eV_atom": [-2.0, None, -1.7],
        }
    )
    manifest = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "pilot_or_new": ["new", "new", "new"],
            "Gate_round": [1, 2, 3],
            "Greedy_round": [3, 1, 2],
        }
    )

    candidates, curve, summary = build_actual_replay(labels, manifest)

    assert len(candidates) == 3
    assert set(curve["policy"]) == {"Gate", "Greedy"}
    at_two = curve.loc[curve["round"].eq(2)].set_index("policy")
    assert at_two.loc["Gate", "cumulative_DFT_evaluable"] == 1
    assert at_two.loc["Greedy", "cumulative_DFT_evaluable"] == 1
    assert summary.set_index("policy").loc["Gate", "final_DFT_evaluable"] == 2
