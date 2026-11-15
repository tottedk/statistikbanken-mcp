"""The model's explicit request wins, and a gap is explained.

These tests lock the two general rules. A server heuristic may fill a
variable the model left unset. It may not replace a code the model set.
A turn with no fetched figures keeps a figure-free explanation.
"""

from __future__ import annotations

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from statistikbanken_mcp.server import find_tabel, hent_data, tabelinfo
from statistikbanken_mcp.statbank import compute_figure, select_named_groups
from statistikbanken_mcp.web import (
    NO_DATA_ANSWER,
    NO_FETCHED_NUMBERS,
    SYSTEM_PROMPT,
    apply_age_intent,
    apply_fetch_defaults,
    guard_answer,
)


def _ages() -> dict:
    return {
        "variables": [
            {
                "id": "ALDER",
                "values": [
                    {"id": "IALT", "text": "Alder i alt"},
                    {"id": "5", "text": "5 år"},
                    {"id": "65", "text": "65 år"},
                    {"id": "65+", "text": "65 år eller derover"},
                ],
            },
            {
                "id": "KØN",
                "values": [
                    {"id": "TOT", "text": "I alt"},
                    {"id": "1", "text": "Mænd"},
                    {"id": "2", "text": "Kvinder"},
                ],
            },
        ]
    }


_COUNTY_NOTE = "De danske amter blev nedlagt 1. januar 2007 og erstattet af regioner og kommuner."


def _fact(value: int, period: str, detail: str, table: str = "FOLK1A") -> dict:
    return {
        "table": table,
        "period": period,
        "unit": "Antal",
        "value": value,
        "detail": detail,
        "unpublished": False,
    }


def test_an_explicit_total_is_not_replaced_by_a_word_in_the_question() -> None:
    chosen = select_named_groups(
        "Hvor mange mænd og kvinder bor der i alt?",
        _ages(),
        {"KØN": ["TOT"]},
    )
    assert chosen["KØN"] == ["TOT"]


def test_a_named_group_still_fills_a_variable_the_model_left_unset() -> None:
    chosen = select_named_groups("Hvor mange kvinder bor i Esbjerg?", _ages(), {})
    assert chosen["KØN"] == ["2"]


def test_relative_time_is_not_read_as_an_age(
    calls: list[dict[str, object]],
) -> None:
    question = "Hvor mange indbyggere havde Esbjerg Kommune for 5 år siden?"
    try:
        hent_data(
            "FOLK1A",
            {"OMRÅDE": ["561"], "KØN": ["TOT"], "Tid": ["2026K3"]},
            question=question,
        )
    except ToolError:
        pass
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    ages = next(item["values"] for item in sent["variables"] if item["code"] == "ALDER")
    assert ages != ["5"]
    kept = apply_age_intent(question, {"table": "FOLK1A", "values": {"OMRÅDE": ["561"]}})
    assert "sum_from" not in kept
    assert "ALDER" not in kept.get("values", {})


def test_an_open_ended_age_is_not_collapsed_to_the_single_year() -> None:
    question = "Hvor stor en andel er 65 år eller derover?"
    chosen = select_named_groups(question, _ages(), {})
    assert "ALDER" not in chosen
    explicit = select_named_groups(question, _ages(), {"ALDER": ["IALT"]})
    assert explicit["ALDER"] == ["IALT"]
    filled = apply_age_intent(
        "Hvor mange er 65 år eller derover i Esbjerg Kommune?",
        {"table": "FOLK1A", "values": {"OMRÅDE": ["561"]}},
    )
    assert "sum_from" not in filled
    assert "ALDER" not in filled.get("values", {})


def test_one_age_intent_does_not_overwrite_every_call() -> None:
    question = "Hvor mange er 0-17 år og hvor mange er over 80?"
    older = apply_age_intent(
        question,
        {"table": "FOLK1A", "values": {"OMRÅDE": ["561"]}, "sum_from": {"ALDER": "80"}},
    )
    assert older["sum_from"] == {"ALDER": "80"}
    assert "sum_range" not in older
    unset = apply_age_intent(question, {"table": "FOLK1A", "values": {"OMRÅDE": ["561"]}})
    assert "sum_from" not in unset
    assert "sum_range" not in unset
    explicit = apply_age_intent(
        "Hvor mange er 80 år og derover?",
        {"table": "FOLK1A", "values": {"OMRÅDE": ["561"], "ALDER": ["80"]}},
    )
    assert explicit["values"]["ALDER"] == ["80"]
    assert "sum_from" not in explicit


def test_two_named_dates_are_not_rewritten_to_the_first() -> None:
    question = "Hvor mange indbyggere havde Viborg Kommune 1. januar 2010 og 1. januar 2026?"
    fetched = apply_fetch_defaults(
        question,
        {"table": "FOLK1A", "values": {"OMRÅDE": ["791"], "Tid": ["2010K1", "2026K1"]}},
    )
    assert fetched["values"]["Tid"] == ["2010K1", "2026K1"]
    facts = [
        _fact(94_000, "2010K1", "OMRÅDE: Viborg, ALDER: Alder i alt"),
        _fact(98_000, "2026K1", "OMRÅDE: Viborg, ALDER: Alder i alt"),
    ]
    answer = guard_answer(question, "kun 2010", facts)
    assert "94.000" in answer
    assert "98.000" in answer
    assert "ikke offentliggjort" not in answer.casefold()


def test_procent_flere_is_a_percent_difference_not_a_share() -> None:
    payload = compute_figure("percent_difference", 703, 400)
    assert payload["result"] == pytest.approx(75.75)
    facts = [
        _fact(703, "2026K1", "OMRÅDE: Aarhus, ALDER: Alder i alt"),
        _fact(400, "2026K1", "OMRÅDE: Esbjerg, ALDER: Alder i alt"),
    ]
    answer = guard_answer(
        "Hvor mange procent flere indbyggere har Aarhus end Esbjerg?",
        "kun tekst",
        facts,
    )
    assert "75,75" in answer
    assert "175,75" not in answer


def test_a_share_uses_the_total_row_as_the_denominator() -> None:
    facts = [
        _fact(400, "2026K1", "ALDER: Alder i alt"),
        _fact(100, "2026K1", "ALDER: 65 år"),
    ]
    answer = guard_answer("Hvor stor er andelen?", "kun tekst", facts)
    assert "Andelen er 25 procent" in answer
    assert "Andelen er 400 procent" not in answer
    named = [
        _fact(1_000, "2026K1", "ALDER: Alder i alt"),
        _fact(100, "2026K1", "ALDER: 65 år"),
        _fact(250, "2026K1", "ALDER: 80 år"),
    ]
    picked = guard_answer("Hvor stor en andel er 80 år?", "kun tekst", named)
    assert "Andelen er 25 procent" in picked
    assert "Andelen er 10 procent" not in picked


def test_a_density_question_keeps_the_table_the_model_chose() -> None:
    fetched = apply_fetch_defaults(
        "Hvor mange indbyggere er der pr. km2 i Esbjerg Kommune?",
        {"table": "ARE207", "values": {"OMRÅDE": ["561"], "Tid": ["2026"]}},
    )
    assert fetched["table"] == "ARE207"
    share = apply_fetch_defaults(
        "Hvor stor en andel af indbyggerne i Esbjerg Kommune er over 80?",
        {"table": "HFUDD11", "values": {"OMRÅDE": ["561"]}},
    )
    assert share["table"] == "HFUDD11"
    population = apply_fetch_defaults(
        "Hvor mange indbyggere har Esbjerg Kommune?",
        {"table": "NRHB", "values": {"Tid": ["2026"]}},
    )
    assert population["table"] == "FOLK1A"


def test_a_figure_free_explanation_is_kept_when_nothing_was_fetched() -> None:
    explanation = "Amterne blev nedlagt 1. januar 2007 og erstattet af regioner og kommuner."
    assert guard_answer("Hvor mange boede der i Ribe Amt i 2005?", explanation, []) == explanation
    generic = guard_answer("Hvor mange boede der i Ribe Amt i 2005?", "Jeg fandt ingen tal.", [])
    assert generic == _COUNTY_NOTE
    assert generic != NO_DATA_ANSWER
    invented = guard_answer("Hvor mange boede der i Ribe Amt?", "Der boede 12.000.", [])
    assert invented == NO_FETCHED_NUMBERS


def test_abolished_counties_are_discoverable_from_search_and_metadata() -> None:
    assert "1 January 2007" in SYSTEM_PROMPT
    assert "amter" in SYSTEM_PROMPT
    hits = find_tabel("befolkning amter", limit=5)
    assert hits
    assert all(hit["geography_note"] == _COUNTY_NOTE for hit in hits)
    info = tabelinfo("FOLK1A", values_filter="amt")
    assert info["geography_note"] == _COUNTY_NOTE
    plain = tabelinfo("FOLK1A")
    assert "geography_note" not in plain


def test_a_fetched_change_states_the_actual_direction() -> None:
    question = "Hvorfor faldt befolkningen i Esbjerg fra 2025 til 2026?"
    facts = [
        _fact(100_000, "2025K4", "OMRÅDE: Esbjerg, ALDER: Alder i alt"),
        _fact(110_000, "2026K4", "OMRÅDE: Esbjerg, ALDER: Alder i alt"),
    ]
    answer = guard_answer(
        question,
        "Den faldt fra 100.000 til 110.000. Forskellen er 10.000.",
        facts,
    )
    assert "ikke faldet" in answer.casefold()
    assert "steget" in answer.casefold()
    assert "100.000" in answer
    assert "110.000" in answer
    ranking = guard_answer(
        "Hvilke fem byer har flest indbyggere, i faldende rækkefølge?",
        "Esbjerg",
        facts,
    )
    assert "ikke faldet" not in ranking.casefold()


def test_an_age_group_plus_the_total_is_accepted() -> None:
    question = "Hvor stor en andel er 65 år eller derover?"
    facts = [
        _fact(100, "2026K1", "ALDER: 65+"),
        _fact(400, "2026K1", "ALDER: Alder i alt"),
    ]
    answer = guard_answer(question, "kun tekst", facts)
    assert "ikke for" not in answer.casefold()
    assert "100" in answer
    assert "400" in answer
    wrong = guard_answer(
        "Hvor mange er 65 år eller derover?",
        "80 år",
        [_fact(12, "2026K1", "ALDER: 80 år")],
    )
    assert "12" in wrong
    assert "ikke for" not in wrong.casefold()


def test_age_bands_written_with_aar_are_accepted() -> None:
    question = "Hvor mange er 0-17 år og 65 år og derover?"
    facts = [
        _fact(30, "2026K1", "ALDER: 0-17 år"),
        _fact(100, "2026K1", "ALDER: 65 år og derover"),
        _fact(400, "2026K1", "ALDER: Alder i alt"),
    ]
    answer = guard_answer(question, "kun tekst", facts)
    assert "ikke for" not in answer.casefold()
    assert "30" in answer
    assert "100" in answer
    assert "400" in answer


def test_a_bare_year_in_a_time_range_uses_the_table_periods(
    calls: list[dict[str, object]],
) -> None:
    hent_data(
        "FOLK1A",
        {"OMRÅDE": ["791"], "ALDER": ["IALT"], "KØN": ["TOT"], "Tid": [">=2008"]},
    )
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    tid = next(item["values"] for item in sent["variables"] if item["code"] == "Tid")
    assert tid == [">=2008K1"]
    calls.clear()
    hent_data(
        "FOLK1A",
        {"OMRÅDE": ["791"], "ALDER": ["IALT"], "KØN": ["TOT"], "Tid": [">=2008<=2026"]},
    )
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    tid = next(item["values"] for item in sent["variables"] if item["code"] == "Tid")
    assert tid == [">=2008K1<=2026K3"]
    calls.clear()
    hent_data(
        "FOLK1A",
        {"OMRÅDE": ["791"], "ALDER": ["IALT"], "KØN": ["TOT"], "Tid": [">=2015K1"]},
    )
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    tid = next(item["values"] for item in sent["variables"] if item["code"] == "Tid")
    assert tid == [">=2015K1"]


def test_every_asked_age_group_is_accepted() -> None:
    question = "Hvor mange er 0-17 år og hvor mange er 65 år eller derover?"
    facts = [
        _fact(30, "2026K1", "ALDER: 0-17"),
        _fact(100, "2026K1", "ALDER: 65+"),
        _fact(400, "2026K1", "ALDER: Alder i alt"),
    ]
    answer = guard_answer(question, "kun tekst", facts)
    assert "ikke for" not in answer.casefold()
    assert "30" in answer
    assert "100" in answer
    open_label = guard_answer(
        "Hvor mange er 65+?",
        "kun tekst",
        [_fact(100, "2026K1", "ALDER: 65 år eller derover")],
    )
    assert "100" in open_label
    assert "ikke for" not in open_label.casefold()


def test_partial_series_keeps_the_available_figures() -> None:
    question = "Hvor meget er befolkningen i Horsens Kommune vokset siden 2000?"
    facts = [
        _fact(50_000, "2008K1", "OMRÅDE: Horsens, ALDER: Alder i alt"),
        _fact(60_000, "2026K3", "OMRÅDE: Horsens, ALDER: Alder i alt"),
        _fact(24_000, "2008K1", "OMRÅDE: Horsens, KØN: Mænd"),
        _fact(29_000, "2026K3", "OMRÅDE: Horsens, KØN: Mænd"),
    ]
    answer = guard_answer(question, "Der findes ikke tal siden 2000.", facts)
    assert "50.000" in answer
    assert "60.000" in answer
    assert "2007" in answer
    assert "ikke med" in answer.casefold()
    span = "Hvor meget er befolkningen i Horsens Kommune vokset fra 2000 til 2026?"
    both = guard_answer(span, "2000 mangler.", facts[:2])
    assert "50.000" in both
    assert "60.000" in both
    assert "2000" in both
    assert "ikke med" in both.casefold()


def test_a_partial_year_keeps_the_unpublished_note() -> None:
    question = "Hvor mange levendefødte var der i Aarhus i 2026?"
    facts = [
        _fact(40, "2026M01", "OMRÅDE: Aarhus", table="BEV22"),
        _fact(55, "2026M09", "OMRÅDE: Aarhus", table="BEV22"),
    ]
    answer = guard_answer(question, "Kun september er med.", facts)
    assert "55" in answer
    assert "ikke offentliggjort" in answer.casefold()
    assert "2026" in answer
    complete = [
        _fact(10, f"2025K{quarter}", "OMRÅDE: Aarhus", table="BEV22") for quarter in range(1, 5)
    ]
    full = guard_answer("Hvor mange levendefødte var der i Aarhus i 2025?", "kun tekst", complete)
    assert "ikke offentliggjort" not in full.casefold()


def test_fetched_ages_are_shown_without_parsing_the_question() -> None:
    """The wording '65-årige' and tiled bands are not unasked ages."""
    recognized = guard_answer(
        "Hvor mange 65-årige eller derover bor i Esbjerg?",
        "kun tekst",
        [
            _fact(100, "2026K1", "OMRÅDE: Esbjerg, ALDER: 65+"),
            _fact(400, "2026K1", "OMRÅDE: Esbjerg, ALDER: Alder i alt"),
        ],
    )
    assert "ikke for" not in recognized.casefold()
    assert "100" in recognized
    assert "400" in recognized
    bands = [_fact(10, "2026K1", f"ALDER: {start}-{start + 4} år") for start in range(15, 70, 5)]
    covered = guard_answer("Hvor mange 15-69-årige er der?", "kun tekst", bands)
    assert "ikke for" not in covered.casefold()
    assert covered.count("10") >= 1


def test_question_text_does_not_inject_a_single_age(
    calls: list[dict[str, object]],
) -> None:
    question = "Hvor mange er der mellem 0 og 17 år i Esbjerg?"
    try:
        hent_data(
            "FOLK1A",
            {"OMRÅDE": ["561"], "KØN": ["TOT"], "Tid": ["2026K3"]},
            question=question,
        )
    except ToolError:
        pass
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    ages = next(item["values"] for item in sent["variables"] if item["code"] == "ALDER")
    assert ages == ["IALT"]
    kept = apply_age_intent(
        "Hvor mange er over 80 i Esbjerg?",
        {"table": "FOLK1A", "values": {"OMRÅDE": ["561"]}},
    )
    assert "sum_from" not in kept


def test_parts_of_a_count_are_summed_when_no_total_row_was_fetched() -> None:
    question = "Hvor mange levendefødte var der i Aalborg Kommune i 3. kvartal 2026?"
    facts = [
        _fact(329, "2026K3", "OMRÅDE: Aalborg, KØN: Mænd", table="BEV22"),
        _fact(324, "2026K3", "OMRÅDE: Aalborg, KØN: Kvinder", table="BEV22"),
    ]
    answer = guard_answer(question, "kun tekst", facts)
    assert "329" in answer
    assert "324" in answer
    assert "653" in answer
    rates = [
        _fact(101, "2026K3", "OMRÅDE: Aalborg, KØN: Mænd", table="PRIS111"),
        _fact(103, "2026K3", "OMRÅDE: Aalborg, KØN: Kvinder", table="PRIS111"),
    ]
    rates[0]["unit"] = "Indeks"
    rates[1]["unit"] = "Indeks"
    index = guard_answer("Hvad er indekset?", "kun tekst", rates)
    assert "204" not in index
