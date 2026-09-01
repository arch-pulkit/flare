from __future__ import annotations

from flare.llm.prompt import parse_llm_response


def test_parse_llm_response_well_formed() -> None:
    raw = (
        "ASSESSMENT: Bus voltage nominal.\n"
        "LIKELY_CAUSE: Transient spike.\n"
        "RECOMMENDED_ACTION: Monitor for 5 minutes.\n"
        "URGENCY: NOMINAL"
    )
    result = parse_llm_response(raw)

    assert set(result.keys()) == {"ASSESSMENT", "LIKELY_CAUSE", "RECOMMENDED_ACTION", "URGENCY"}
    assert result["ASSESSMENT"] == "Bus voltage nominal."
    assert result["LIKELY_CAUSE"] == "Transient spike."
    assert result["RECOMMENDED_ACTION"] == "Monitor for 5 minutes."
    assert result["URGENCY"] == "NOMINAL"


def test_parse_llm_response_missing_field() -> None:
    raw = (
        "ASSESSMENT: Bus voltage nominal.\n"
        "LIKELY_CAUSE: Transient spike.\n"
        "RECOMMENDED_ACTION: Monitor for 5 minutes."
    )
    result = parse_llm_response(raw)

    assert result["ASSESSMENT"] == "Bus voltage nominal."
    assert result["LIKELY_CAUSE"] == "Transient spike."
    assert result["RECOMMENDED_ACTION"] == "Monitor for 5 minutes."
    assert result["URGENCY"] == "NOT PROVIDED"


def test_parse_llm_response_free_prose() -> None:
    raw = "The bus voltage appears to be within limits."
    result = parse_llm_response(raw)

    assert set(result.keys()) == {"ASSESSMENT", "LIKELY_CAUSE", "RECOMMENDED_ACTION", "URGENCY"}
    assert result["ASSESSMENT"] == raw
    assert result["LIKELY_CAUSE"] == "NOT PROVIDED"
    assert result["RECOMMENDED_ACTION"] == "NOT PROVIDED"
    assert result["URGENCY"] == "NOT PROVIDED"
