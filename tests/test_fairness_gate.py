"""Unit gates for the fairness gate: violations must BLOCK, fair passes."""
import pytest

from qdepipe.fairness_gate import FairnessViolation, assert_fair_comparison


def _run(**over):
    base = dict(dataset_id="d1", split_id="s1", scaler_scope="train",
                horizon=1, n_features=96, seed_set="0-4", tuning_budget=12)
    base.update(over)
    return base


def test_fair_pair_passes():
    assert assert_fair_comparison([_run(), _run()])


def test_leakage_scope_blocks():
    with pytest.raises(FairnessViolation, match="leakage-unsafe"):
        assert_fair_comparison([_run(), _run(scaler_scope="global")])


def test_unmatched_features_block():
    with pytest.raises(FairnessViolation, match="unmatched measured feature"):
        assert_fair_comparison([_run(n_features=64), _run(n_features=32)])


def test_different_split_blocks():
    with pytest.raises(FairnessViolation, match="unequal split_id"):
        assert_fair_comparison([_run(), _run(split_id="s2")])


def test_different_dataset_blocks():
    with pytest.raises(FairnessViolation, match="unequal dataset_id"):
        assert_fair_comparison([_run(), _run(dataset_id="d2")])


def test_unequal_seed_set_blocks():
    with pytest.raises(FairnessViolation, match="seed_set"):
        assert_fair_comparison([_run(), _run(seed_set="0-9")])


def test_unequal_tuning_budget_blocks():
    with pytest.raises(FairnessViolation, match="tuning_budget"):
        assert_fair_comparison([_run(), _run(tuning_budget=24)])


def test_missing_metadata_blocks():
    incomplete = {k: v for k, v in _run().items() if k != "scaler_scope"}
    with pytest.raises(FairnessViolation, match="missing"):
        assert_fair_comparison([_run(), incomplete])


def test_declared_tolerance_allows_small_f_gap():
    assert assert_fair_comparison(
        [_run(n_features=96), _run(n_features=97)], f_tolerance=1)
