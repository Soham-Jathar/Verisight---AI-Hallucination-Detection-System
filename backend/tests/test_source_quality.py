from app.schemas import EvidenceSource
from app.services.source_quality import assess_source, enrich_source


def test_official_source_scores_higher_than_community_wiki() -> None:
    official_score, official_label = assess_source("https://www.python.org/doc/")
    community_score, community_label = assess_source("https://marvel.fandom.com/wiki/Doctor_Doom")

    assert official_score > community_score
    assert official_label == "Official source"
    assert community_label == "Community wiki"


def test_uploaded_document_is_labelled_as_user_evidence() -> None:
    source = enrich_source(
        EvidenceSource(title="Report", url="document://report.pdf", snippet="A short report excerpt.")
    )

    assert source.credibility == 0.90
    assert source.source_quality == "Uploaded document"
