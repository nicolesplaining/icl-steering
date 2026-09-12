from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.gsm8k_conditional_geometry import crossfit_predictions, summarize


def test_held_out_targets_cannot_change_their_own_predictions():
    g = torch.Generator().manual_seed(19)
    x = torch.randn(32, 6, generator=g, dtype=torch.double)
    y = torch.randn(32, 6, generator=g, dtype=torch.double)
    before, assignment = crossfit_predictions(x, y)
    changed = y.clone()
    changed[assignment == 0] += 1000
    after, second_assignment = crossfit_predictions(x, changed)
    assert torch.equal(assignment, second_assignment)
    for kind in before:
        torch.testing.assert_close(before[kind][assignment == 0], after[kind][assignment == 0])


def test_linear_map_recovers_signal_beyond_scalar_query_rescaling():
    g = torch.Generator().manual_seed(31)
    x = torch.randn(128, 3, generator=g, dtype=torch.double)
    transform = torch.tensor([[0., 2., 0.], [0., 0., -3.], [4., 0., 0.]], dtype=torch.double)
    y = x @ transform + torch.tensor([8., 9., 10.])
    predictions, assignment = crossfit_predictions(x, y)
    stats = summarize(y, predictions, assignment)["predictors"]
    assert stats["ridge"]["error_reduction_vs_mean"] > 0.99
    assert stats["scalar"]["error_reduction_vs_mean"] < 0.1


def test_scalar_control_exactly_recovers_isotropic_rescaling():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(64, 8, generator=g, dtype=torch.double)
    y = -0.25*x + 3
    predictions, _ = crossfit_predictions(x, y)
    torch.testing.assert_close(predictions["scalar"], y)


def test_linear_map_with_model_width_features_matches_known_signal():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(128, 3)) @ rng.normal(size=(3, 3584))
    y = np.roll(x, 17, axis=1) + 4
    predictions, assignment = crossfit_predictions(torch.from_numpy(x), torch.from_numpy(y))
    stats = summarize(torch.from_numpy(y), predictions, assignment)["predictors"]
    assert stats["ridge"]["error_reduction_vs_mean"] > 0.99
