"""Behavioral tests for the v3 male-genuine coverage validator fixtures.

Runs the validator's pure fixture suite (no artifacts, no network) and pins
the required rejection/acceptance behaviors, so a regression in the coverage
audit logic fails the test suite directly.  The full artifact audit remains a
separate, explicitly invoked step:

    ./.venv-data/bin/python -m scripts.validate_dataset_v3_coverage
"""
from scripts.validate_dataset_v3_coverage import run_fixtures

REQUIRED_CASES = {
    "benchmark_male_genuine_attempted_in_train",
    "benchmark_synthetic_participant_attempted",
    "old_exposure_identity_attempted",
    "speaker_shared_across_added_splits",
    "clean_additions_accepted",
    "wrong_path_base_rejected",
    "transitive_bridge_to_benchmark_rejected",
    "renamed_duplicate_hash_rejected",
    "single_speaker_reserved_for_dev_accepted",
    "failed_read_charge_retained",
    "allowance_exhaustion_blocks_before_io",
}


def test_fixture_suite_passes_and_covers_required_cases():
    report = run_fixtures()
    assert report["passed"] is True
    names = {case["case"] for case in report["cases"]}
    missing = REQUIRED_CASES - names
    assert not missing, f"missing required fixture cases: {sorted(missing)}"
    failed = [case for case in report["cases"] if not case["passed"]]
    assert not failed, f"failing fixture cases: {failed}"
