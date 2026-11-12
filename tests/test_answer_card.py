"""Verifier and question decomposition. No live model."""

from statistikbanken_mcp.answer_card import (
    allocate_budget,
    build_card,
    decompose,
    render_card,
    verify_card,
)

_ESBJERG = {
    "table": "FOLK1A",
    "period": "2026K4",
    "unit": "Antal",
    "value": 114_766,
    "detail": "OMRÅDE: Esbjerg, ALDER: Alder i alt",
}


def _card(**overrides: object) -> dict:
    card = {
        "premise_check": "na",
        "premise_figure": None,
        "table_id": "FOLK1A",
        "periods": ["2026K4"],
        "unit": "Antal",
        "values": [114_766],
        "sub_parts": [{"part": "Esbjerg", "value": 114_766, "missing": None}],
        "share": None,
    }
    card.update(overrides)
    return card


def test_a_card_of_fetched_figures_verifies() -> None:
    card = build_card("Hvor mange indbyggere har Esbjerg?", [_ESBJERG])
    assert verify_card(card, [_ESBJERG]) == []
    assert "114.766" in render_card(card)
    assert "Antal" in render_card(card)


def test_a_value_missing_from_the_rows_is_rejected() -> None:
    errors = verify_card(_card(values=[1]), [_ESBJERG])
    assert errors
    assert any("1" in error for error in errors)


def test_an_unknown_period_is_rejected() -> None:
    errors = verify_card(_card(periods=["1999K1"]), [_ESBJERG])
    assert any("1999K1" in error for error in errors)


def test_a_unit_from_another_table_is_rejected() -> None:
    errors = verify_card(_card(unit="Indeks"), [_ESBJERG])
    assert any("Indeks" in error for error in errors)


def test_a_share_must_use_the_total_as_denominator() -> None:
    women = {**_ESBJERG, "value": 60, "detail": "KØN: Kvinder"}
    men = {**_ESBJERG, "value": 40, "detail": "KØN: Mænd"}
    parts = [
        {"part": "Kvinder", "value": 60, "missing": None},
        {"part": "Mænd", "value": 40, "missing": None},
    ]
    sibling = {
        "numerator": "Kvinder",
        "denominator": "Mænd",
        "numerator_value": 60,
        "denominator_value": 40,
        "result": 150,
        "period": "2026K4",
        "table_id": "FOLK1A",
    }
    assert verify_card(_card(values=[60, 40], sub_parts=parts, share=sibling), [women, men])
    whole = {**sibling, "denominator": "I alt", "denominator_value": 100, "result": 60}
    assert (
        verify_card(_card(values=[60, 40, 100], sub_parts=parts, share=whole), [women, men]) == []
    )


def test_a_false_premise_is_rendered_with_nej() -> None:
    card = _card(premise_check="false", premise_figure=114_766, values=[114_766])
    text = render_card(card)
    assert text.startswith("Nej")
    assert "114.766" in text
    assert verify_card(card, [_ESBJERG]) == []
    invented = _card(premise_check="false", premise_figure=1)
    assert verify_card(invented, [_ESBJERG])


def test_a_sub_part_without_a_value_or_a_reason_is_rejected() -> None:
    card = _card(sub_parts=[{"part": "Ringkøbing-Skjern", "value": None, "missing": None}])
    assert verify_card(card, [_ESBJERG])
    told = _card(sub_parts=[{"part": "Ringkøbing-Skjern", "value": None, "missing": "ikke hentet"}])
    assert verify_card(told, [_ESBJERG]) == []


def test_equivalent_spellings_match_and_a_wrong_number_does_not() -> None:
    for spelling in ("114.766", "114,766", "114766", "114.766,0", "114.766 personer"):
        card = _card(
            values=[spelling],
            sub_parts=[{"part": "Esbjerg", "value": spelling, "missing": None}],
        )
        assert verify_card(card, [_ESBJERG]) == [], spelling
    rate = {**_ESBJERG, "value": 12.3, "unit": "pct"}
    for spelling in ("12,3 %", "12.3%", "0.123"):
        card = _card(
            unit="pct",
            values=[spelling],
            sub_parts=[{"part": "ændring", "value": spelling, "missing": None}],
        )
        assert verify_card(card, [rate]) == [], spelling
    wrong = _card(values=["99.999"])
    assert verify_card(wrong, [_ESBJERG])


def test_a_verified_card_never_renders_blank() -> None:
    card = _card(premise_check="true", sub_parts=[], share=None)
    assert verify_card(card, [_ESBJERG]) == []
    rendered = render_card(card).strip()
    assert rendered
    assert "114.766" in rendered


def test_a_realistic_card_verifies_on_the_first_try() -> None:
    """Dict values, labels, a string period, and a percent derived from counts."""
    facts = [
        {
            "table": "FOLK1A",
            "period": "2026K4",
            "unit": "Antal",
            "value": 200,
            "detail": "OMRÅDE: Aarhus",
        },
        {
            "table": "FOLK1A",
            "period": "2026K4",
            "unit": "Antal",
            "value": 80,
            "detail": "OMRÅDE: Odense",
        },
    ]
    card = {
        "premise_check": "na",
        "premise_figure": None,
        "table_id": "FOLK1A",
        "periods": "2026K4",
        "unit": "pct.",
        "values": [
            {"label": "Aarhus", "value": 200, "unit": "Antal"},
            {"label": "Odense", "value": "80"},
            "Mænd",
            "0-17",
            {"label": "ændring", "value": "150 %"},
            250,
        ],
        "sub_parts": [
            {"part": "Aarhus", "value": {"label": "Aarhus", "value": 200}},
            {"part": "Odense", "value": 80},
            {"part": "Aalborg", "value": None, "missing": "ikke hentet"},
        ],
        "share": None,
    }
    assert verify_card(card, facts) == []
    assert render_card(card).strip()


def test_labels_are_not_figures_and_a_wrong_number_still_fails() -> None:
    card = _card(values=["Mænd", "0-17", {"label": "Esbjerg", "value": 114_766}])
    assert verify_card(card, [_ESBJERG]) == []
    wrong = _card(values=["Mænd", {"label": "Esbjerg", "value": 1}])
    assert verify_card(wrong, [_ESBJERG])


def test_percent_difference_and_ratio_are_derived_from_counts() -> None:
    facts = [
        {**_ESBJERG, "value": 200, "detail": "A"},
        {**_ESBJERG, "value": 80, "detail": "B"},
    ]
    difference = _card(values=[150], unit="pct.", sub_parts=[])
    ratio = _card(values=[250], unit="%", sub_parts=[])
    assert verify_card(difference, facts) == []
    assert verify_card(ratio, facts) == []
    assert verify_card(_card(values=[999], unit="pct.", sub_parts=[]), facts)


def test_decomposition_splits_places_and_the_tool_cap() -> None:
    parts = decompose("Hvor mange indbyggere har Horsens Kommune og Ringkøbing-Skjern Kommune?")
    assert len(parts) == 2
    assert "Horsens" in parts[0]
    assert "Ringkøbing-Skjern" in parts[1]
    assert allocate_budget(2, 6) == [3, 3]
    assert allocate_budget(7, 6) == [1, 1, 1, 1, 1, 1, 0]
    assert decompose("Hvor mange bor i Esbjerg?") == ["Hvor mange bor i Esbjerg?"]
