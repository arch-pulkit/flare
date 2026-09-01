# BOUNDARY: config/schema_builder.py → imports from astr_o.schema and stdlib only
from __future__ import annotations

import logging
from pathlib import Path

from astr_o.schema import SchemaExtractor

logger = logging.getLogger(__name__)

_TIER_MAP: dict[str, str] = {
    "voltage_spec.txt":         "CRITICAL",
    "thermal_spec.txt":         "CRITICAL",
    "pressure_spec.txt":        "SUPPORTING",
    "adcs_spec.txt":            "CRITICAL",
    "comms_spec.txt":           "CRITICAL",
    "eps_detailed_spec.txt":    "CRITICAL",
}


def build_domain_schema(docs_folder: Path) -> dict:
    """Load all .txt files from docs_folder and build an ASTR-O domain schema.

    Returns schema dict ready for ASTROPipeline(domain_schema=...).
    Does not call ASTROPipeline. Does not write any files.
    """
    docs: list[dict] = []
    for txt_file in sorted(docs_folder.glob("*.txt")):
        tier = _TIER_MAP.get(txt_file.name, "REFERENCE")
        docs.append({
            "text":   txt_file.read_text(encoding="utf-8"),
            "doc_id": txt_file.stem,
            "tier":   tier,
        })
        logger.info("Loaded %s (tier=%s, %d chars)", txt_file.name, tier, len(docs[-1]["text"]))

    if not docs:
        raise ValueError(f"No .txt documents found in {docs_folder}")

    extractor = SchemaExtractor(
        custom_patterns={
            "rpm":       r"(\d+\.?\d*)\s*(?:rpm\b|RPM\b)",
            "frequency": r"(\d+\.?\d*)\s*(?:Hz\b|MHz\b|GHz\b)",
        }
    )
    schema = extractor.extract_and_convert(docs)
    logger.info("Domain schema built: %d entity types", len(schema))
    return schema
