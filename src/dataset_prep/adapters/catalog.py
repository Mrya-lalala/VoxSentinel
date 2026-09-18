"""Source-access catalog for the bounded preparation pilot.

Statuses were verified at execution time (2026-09-17) against the official
sources.  A source that is not accessible now is recorded here with the exact
route and the action only the user can take (accepting terms, portal request).
No tokens, no invented URLs, no guessed metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..records import (
    STATUS_MISSING_SOURCE,
)


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    name: str
    official_url: str
    status: str
    license_note: str
    role: str
    user_action: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "official_url": self.official_url,
            "status": self.status,
            "license": self.license_note,
            "role": self.role,
            "user_action": self.user_action,
            "notes": list(self.notes),
        }


def source_catalog() -> list[SourceEntry]:
    """Return the catalog as verified during this preparation run."""
    return [
        SourceEntry(
            source_id="indicsynth",
            name="IndicSynth synthetic Indic speech",
            official_url="https://huggingface.co/datasets/vdivyasharma/IndicSynth",
            status="accessible",
            license_note="CC BY-NC 4.0 (dataset card); non-commercial research use",
            role="core synthetic windows (12 Indic languages)",
            notes=[
                "Per-language configs; only the upstream `train` partition is used.",
                "Rows are selected via the Hugging Face datasets-server metadata API; "
                "only selected rows' audio assets are downloaded.",
                "CC BY-NC 4.0 forbids commercial use; downstream checkpoints inherit "
                "this restriction for this data portion.",
            ],
        ),
        SourceEntry(
            source_id="nisp",
            name="NISP Indian-accented English speech",
            official_url="https://github.com/iiscleap/NISP-Dataset",
            status="accessible",
            license_note="CC BY 4.0 (repository README)",
            role="genuine Indian-English candidate windows (unpaired without synthetic English)",
            notes=[
                "English recordings of the Hindi/Kannada/Malayalam/Tamil/Telugu native-language folders.",
                "Multipart tar.gz archives are streamed in exact part order with a per-language byte budget; "
                "only official train speakers are used, official test speakers are excluded.",
            ],
        ),
        SourceEntry(
            source_id="asvspoof2019",
            name="ASVspoof 2019 LA",
            official_url="https://datashare.ed.ac.uk/handle/10283/3336",
            status="accessible_selective",
            license_note="ASVspoof 2019 challenge data terms (see DataShare item and asvspoof.org)",
            role="separate fallback/baseline pool (20+20 train, 5+5 dev)",
            notes=[
                "The full LA.zip archive is 7.64 GB and is never downloaded whole; selected FLAC members "
                "and protocol files are fetched with HTTP range requests over the remote ZIP.",
                "Official train/dev membership and protocol labels are preserved; this pool is kept separate "
                "from the Indic pilot and is not routed through the Indic encoder automatically.",
            ],
        ),
        SourceEntry(
            source_id="nptel",
            name="NPTEL2020 Indian-English speech (published pure set)",
            official_url="https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset",
            status="accessible_bounded",
            license_note="CC BY (see repository); YouTube NPTEL content under Creative Commons",
            role="supplementary genuine English windows (lecturer identity unresolved)",
            notes=[
                "Only the official 189 MB `nptel-pure-set.tar.gz` release asset is used; the full corpus is "
                "about 1.1 TB and is not downloaded.",
                "If lecturer identities cannot be resolved, these clips stay supplementary/inventory-only and "
                "are not claimed to be speaker-disjoint.",
            ],
        ),
        SourceEntry(
            source_id="kathbath",
            name="Kathbath genuine Indic speech (IndicSUPERB)",
            official_url="https://huggingface.co/datasets/ai4bharat/Kathbath",
            status="accessible",
            license_note="CC BY 4.0 (dataset card)",
            role="core genuine windows (12 Indic languages) - PROCESSED",
            notes=[
                "Gated access was granted to the user's Hub account (auto terms); reads succeed.",
                "Per-language revision SHA resolved at acquisition; recorded in every manifest row.",
                "Only selected recordings are fetched, row group by row group; a per-language byte cap bounds the ledger.",
                "These recordings provide the genuine side of the paired core; synthetic rows referencing held-out shard recordings are excluded.",
            ],
        ),
        SourceEntry(
            source_id="svarah",
            name="Svarah genuine English evaluation",
            official_url="https://huggingface.co/datasets/ai4bharat/Svarah",
            status="accessible",
            license_note="CC BY 4.0 (dataset card)",
            role="external evaluation only (20 genuine English windows PROCESSED)",
            notes=[
                "Gated access was granted to the user's Hub account (auto terms); reads succeed.",
                "Revision SHA and per-row audio path/speaker/state metadata stored in every parent_refs block.",
                "Selected windows stay outside training, validation, threshold tuning and synthetic generation.",
            ],
        ),
        SourceEntry(
            source_id="indicvoices",
            name="IndicVoices genuine Indic evaluation",
            official_url="https://huggingface.co/datasets/ai4bharat/IndicVoices",
            status="accessible_not_processed",
            license_note="CC BY 4.0 (dataset card)",
            role="external evaluation only (genuine Indic languages)",
            user_action=(
                "Optional follow-up: extend the external evaluation pool with IndicVoices recordings; "
                "the parquet/row-group access pattern is already proven by the Svarah stage."
            ),
            notes=[
                "Access changed during this run: the gate was accepted and file reads now succeed "
                "(verified by reading a Hindi shard footer: train-00000-of-00082.parquet, 5,429 rows).",
                "Not processed in this bounded run; the documented scope selected Svarah for new external "
                "evaluation material.",
                "Even upstream `valid` recordings would be reserved for external evaluation, not training.",
            ],
        ),
        SourceEntry(
            source_id="spire_sies",
            name="SPIRE-SIES Indian-accented English",
            official_url="https://spiredatasets.ee.iisc.ac.in/",
            status="awaiting_user_links",
            license_note="See SPIRE portal Terms & Conditions (not assumed from the paper license)",
            role="supplementary genuine English",
            user_action=(
                "The user reported the portal archive as unavailable for this session; provide a local "
                "SPIRE-SIES archive path in a follow-up run to import it."
            ),
            notes=[
                "The portal exposes a Download tab per dataset; access is request-based.",
                "No audio was fetched; no workaround is attempted in this run.",
            ],
        ),
        SourceEntry(
            source_id="indic_timit",
            name="Indic TIMIT read English",
            official_url="https://spiredatasets.ee.iisc.ac.in/indictimitcorpus",
            status="awaiting_user_links",
            license_note="See SPIRE portal Terms & Conditions",
            role="supplementary genuine English",
            user_action=(
                "The user reported the portal archive as unavailable for this session; provide a local "
                "Indic TIMIT release path in a follow-up run to import it."
            ),
            notes=[
                "80 voice artists, ~240 h read English; unmatched access/terms recorded as a shortfall.",
            ],
        ),
        SourceEntry(
            source_id="synthetic_english",
            name="Synthetic Indian-English coverage",
            official_url="https://huggingface.co/datasets?search=synthetic+speech+indic",
            status=STATUS_MISSING_SOURCE,
            license_note="n/a - no traceable source selected",
            role="required pairing for NISP genuine English - NOT PREPARED",
            user_action=(
                "Provide user-authorized synthetic assets (generator + reference material whose terms permit "
                "generation) or point the importer at a documented dataset; see "
                "configs/synthetic_english_template.json and `fetch --source synthetic_english --import-dir ...`."
            ),
            notes=[
                "No traceable, documented synthetic Indian-English dataset was found; undocumented third-party "
                "TTS dumps without generation provenance or license were rejected on purpose.",
                "The importer stays schema-compatible; NISP genuine windows are delivered as `unpaired_candidate` "
                "and are excluded from the default balanced training manifest.",
                "Accent verification of any future synthetic source is unreviewed until actually reviewed; an `en` "
                "TTS setting alone does not establish an Indian accent.",
            ],
        ),
    ]


__all__ = ["SourceEntry", "source_catalog"]
