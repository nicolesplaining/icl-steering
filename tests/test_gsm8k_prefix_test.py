import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.gsm8k_prefix_test import file_hash, locked_selection, matched_direction


@pytest.mark.parametrize("mutation", ["alpha", "eligibility", "directions"])
def test_supplement_waits_for_parent_lock_and_rejects_retuned_selection(tmp_path, mutation):
    assert locked_selection(tmp_path) is None
    selection = {"eligible": True, "chosen": {"layer": 7, "alpha": 0.5}}
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    (tmp_path / "directions.pt").write_bytes(b"reference")
    assert locked_selection(tmp_path) is None
    (tmp_path / "test_lock.json").write_text(json.dumps({
        "selection_sha256": file_hash(tmp_path / "selection.json"),
        "directions_sha256": file_hash(tmp_path / "directions.pt")}))
    assert locked_selection(tmp_path) == selection
    if mutation == "alpha": selection["chosen"]["alpha"] = 1
    if mutation == "eligibility": selection["eligible"] = False
    if mutation == "directions": (tmp_path / "directions.pt").write_bytes(b"changed")
    else: (tmp_path / "selection.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="changed"):
        locked_selection(tmp_path)


def test_failed_parent_gate_does_not_require_or_start_test(tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps({"eligible": False}))
    assert locked_selection(tmp_path) == {"eligible": False}
    assert not (tmp_path / "test_lock.json").exists()


def test_controls_match_reference_norm_without_changing_direction():
    zero = torch.zeros(3, 2)
    control = torch.tensor([[3., 4.]]).repeat(3, 1)
    actual = matched_direction(control, zero, torch.tensor([0., 10.]))
    torch.testing.assert_close(actual, torch.tensor([6., 8.]))
    with pytest.raises(ValueError, match="Invalid"):
        matched_direction(zero, zero, torch.ones(2))
