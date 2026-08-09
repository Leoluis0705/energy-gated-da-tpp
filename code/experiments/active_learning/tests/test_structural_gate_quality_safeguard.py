import numpy as np
import pytest

from active_learning_energy_gate_ablation import apply_quality_safeguard


def test_q95_accepts_exact_boundary():
    chosen, fallback, direct_sum, proposed_sum = apply_quality_safeguard(
        [0, 1], [2, 3], np.array([0.5, 0.5, 0.50, 0.45]), 0.95
    )

    assert chosen == [2, 3]
    assert fallback is False
    assert (direct_sum, proposed_sum) == pytest.approx((1.0, 0.95))


def test_q95_returns_direct_batch_in_original_order_below_boundary():
    chosen, fallback, direct_sum, proposed_sum = apply_quality_safeguard(
        [1, 0], [2, 3], np.array([0.5, 0.5, 0.49, 0.45]), 0.95
    )

    assert chosen == [1, 0]
    assert fallback is True
    assert (direct_sum, proposed_sum) == pytest.approx((1.0, 0.94))


def test_q95_accepts_zero_sum_proposal_when_direct_is_zero():
    chosen, fallback, direct_sum, proposed_sum = apply_quality_safeguard(
        [0, 1], [2, 3], np.zeros(4), 0.95
    )

    assert chosen == [2, 3]
    assert fallback is False
    assert direct_sum == proposed_sum == 0.0


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01, float("nan")])
def test_quality_safeguard_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="quality safeguard fraction"):
        apply_quality_safeguard([0], [1], np.array([0.5, 0.5]), fraction)
