"""Adversarial fixture tests for the independent v2 split validator.

These exercise the leak-detection logic directly (reject leaks, permit
legitimate cases) without touching the real dataset.  Coverage includes the
transitive-closure audit (bridge vs disjoint), quota-overflow components,
parent verification/evidence requirements, cross-split components, and the
near-duplicate fingerprint detector on re-encoded audio.
"""
from scripts.validate_dataset_v2 import (
    _genuine_row,
    _near_duplicate_similarity,
    _synth_row,
    audit_closure,
    audit_manifests,
    run_fixtures,
)


def test_adversarial_fixtures_reject_leaks_and_permit_legitimate_cases():
    report = run_fixtures()
    failures = [case for case in report["cases"] if not case["passed"]]
    assert not failures, failures
    audio_failures = [case for case in report["audio_cases"] if not case["passed"]]
    assert not audio_failures, audio_failures
    assert report["passed"] is True


def test_required_behavioral_cases_present_and_effective():
    """The fixture matrix must include, and actually exercise, the cases the
    review asked for (transitive bridge, quota overflow, re-encode duplicate)."""
    report = run_fixtures()
    names = {case["case"] for case in report["cases"]}
    required = {
        "transitive_closure_bridge_rejected",
        "transitive_closure_disjoint_accepted",
        "oversized_component_split_across_splits",
        "oversized_component_within_test_accepted",
        "unverified_source_parent_rejected",
        "unverified_target_parent_rejected",
        "parent_evidence_missing_rejected",
        "parent_record_not_in_scanned_inventories_rejected",
    }
    assert required <= names, required - names
    audio_names = {case["case"] for case in report["audio_cases"]}
    assert {"reencoded_fingerprint_detected", "unrelated_audio_not_flagged"} == audio_names
    # sanity: the bridge case must fail while its disjoint control passes
    by_name = {case["case"]: case for case in report["cases"]}
    assert "test_closure_component_intersects_exposure" in by_name["transitive_closure_bridge_rejected"]["issues"]
    assert "test_closure_component_intersects_exposure" not in by_name["transitive_closure_disjoint_accepted"]["issues"]


def test_closure_audit_rejects_multihop_bridge_and_permits_disjoint():
    """Direct unit test of the transitive-closure audit over a two-hop chain."""
    row = _synth_row("is-te-hop", "Bengali", "700", "500", "777-500-f.wav", "888-700-f.wav")
    edges = [
        (("j", "bengali", "777"), ("r", "bengali", "444-444-f")),
        (("r", "bengali", "444-444-f"), ("s", "bengali", "444")),
    ]
    anchored = audit_closure([row], edges, [("s", "bengali", "444")])
    assert anchored["n_hits"] == 1 and anchored["hits"][0]["window_id"] == "is-te-hop"
    disjoint = audit_closure([row], edges, [("s", "bengali", "555")])
    assert disjoint["n_hits"] == 0


def test_reencoded_audio_exercises_fingerprint_not_hash():
    """The near-duplicate fixture must flag a re-encoded signal that is not
    byte-identical (hash equality alone cannot catch transcodes)."""
    import numpy as np

    from scripts.validate_dataset_v2 import _fixture_clip

    base = _fixture_clip(11, 260.0, 0.2)
    reencoded = np.round(base.astype(np.float64) * 32767).astype(np.int16).astype(np.float64) / 32767.0
    assert not np.array_equal(base, reencoded.astype(np.float32))  # genuinely different bytes
    _, _, hit = _near_duplicate_similarity(base, 16000, reencoded, 16000)
    assert hit
    _, _, miss = _near_duplicate_similarity(base, 16000, _fixture_clip(3, 750.0, 0.5), 16000)
    assert not miss
