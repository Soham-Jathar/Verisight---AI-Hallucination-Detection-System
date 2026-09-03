from __future__ import annotations

from urllib.parse import urlparse

from app.schemas import EvidenceSource


OFFICIAL_DOMAINS = {
    "python.org", "w3.org", "nasa.gov", "esa.int", "who.int", "un.org",
    "microsoft.com", "openai.com", "apple.com", "google.com", "meta.com",
    "oracle.com", "ibm.com", "mozilla.org", "fifa.com", "olympics.com",
}

REPUTABLE_NEWS_DOMAINS = {
    "reuters.com", "bbc.com", "apnews.com", "nytimes.com", "theguardian.com",
    "economist.com", "nature.com", "scientificamerican.com", "britannica.com",
}


def assess_source(url: str) -> tuple[float, str]:
    """Estimate source quality from its origin, not from its claim text.

    This is a transparent ranking heuristic. It helps evidence selection but
    never turns a source into factual proof by itself.
    """
    if url.startswith("document://"):
        return 0.90, "Uploaded document"

    host = urlparse(url).netloc.lower().removeprefix("www.")
    if not host:
        return 0.45, "Unknown source"
    if host.endswith((".gov", ".gov.in")):
        return 0.96, "Government source"
    if host.endswith((".edu", ".edu.in", ".ac.in")):
        return 0.90, "Academic source"
    if host in OFFICIAL_DOMAINS or any(host.endswith(f".{domain}") for domain in OFFICIAL_DOMAINS):
        return 0.95, "Official source"
    if host.endswith("wikipedia.org"):
        return 0.72, "Encyclopedia"
    if host in REPUTABLE_NEWS_DOMAINS or any(host.endswith(f".{domain}") for domain in REPUTABLE_NEWS_DOMAINS):
        return 0.84, "Reputable publication"
    if host.endswith(("fandom.com", "wikia.com")):
        return 0.35, "Community wiki"
    if host.endswith((".org", ".int")):
        return 0.70, "Organisation source"
    return 0.55, "Web source"


def enrich_source(source: EvidenceSource) -> EvidenceSource:
    credibility, source_quality = assess_source(source.url)
    return source.model_copy(
        update={"credibility": credibility, "source_quality": source_quality}
    )
