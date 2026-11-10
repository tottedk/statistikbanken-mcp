"""Scorer and verifier fixtures. No xAI call and no Statistikbanken call."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals.score import Truth, classify, cost_usd, parse_numbers, prices_from_env, render_report
from evals.verify import VerifierError, evaluate_check

from statistikbanken_mcp.web import NO_DATA_ANSWER, NO_FETCHED_NUMBERS, UNVERIFIED_NUMBERS

QUESTIONS = Path(__file__).resolve().parents[1] / "evals" / "questions.json"


def _result(answer: str, facts=None, calls=None, **extra):
    payload = {
        "answer": answer,
        "facts": facts or [],
        "calls": calls or [],
        "show_numbers": True,
        "suggestions": [],
        "error": None,
    }
    payload.update(extra)
    return payload


def _fact(value, table="FOLK1A", period="2026K3"):
    return {"table": table, "period": period, "value": value}


def test_parses_danish_and_english_numbers() -> None:
    assert parse_numbers("Der bor 58.381 personer.") == [58381]
    assert parse_numbers("7,7 procent og 102.6") == [7.7, 102.6]
    assert parse_numbers("12.190 tusind, English 1,234.5") == [12190, 1234.5]
    assert parse_numbers("faldet er -10,2") == [-10.2]
    assert parse_numbers("0-6 år i 2026K3") == [0, 6]
    assert 58381 in parse_numbers("5 045 og 58.381")


def test_figures_taxonomy() -> None:
    question = {"question": "Hvor mange?", "expect": {"kind": "figures"}}
    truth = [Truth("FOLK1A", [58381], ["2026K3"])]
    assert classify(question, _result("58.381 indbyggere."), truth) == "correct"
    hidden = _result(UNVERIFIED_NUMBERS, [_fact(58381)])
    assert classify(question, hidden, truth) == "guard_hid_correct_answer"
    assert classify(question, _result(NO_DATA_ANSWER), truth) == "no_answer"
    wrong_table = _result(
        "6.977",
        [_fact(6977, "FOLK1C", "2026K3")],
        [{"name": "hent_data", "arguments": json.dumps({"table": "FOLK1C"})}],
    )
    assert classify(question, wrong_table, truth) == "wrong_table"
    wrong_period = _result("20.064", [_fact(20064, "BIL55", "2026M09")])
    year = [Truth("BIL55", [185394], ["2025"])]
    assert classify(question, wrong_period, year) == "wrong_period"
    wrong_number = _result("1.000", [_fact(1000, "FOLK1A", "2026K3")])
    assert classify(question, wrong_number, truth) == "wrong_number"
    assert classify(question, _result("", error="timed out"), truth) == "timeout_error"


def test_edge_behaviours() -> None:
    gap = {"question": "Hvor mange bor der i Narnia?", "expect": {"kind": "no_table"}}
    assert classify(gap, _result(NO_DATA_ANSWER), []) == "correct"
    assert classify(gap, _result("Der bor 58.381."), []) == "edge_behaviour_wrong"
    future = {
        "question": "Hvor mange indbyggere havde Gentofte Kommune i 1. kvartal 2030?",
        "expect": {"kind": "not_published"},
    }
    assert classify(future, _result(NO_FETCHED_NUMBERS), []) == "correct"
    approximate = {
        "question": "Hvor mange børn bor der i København der lever med mere end 2 voksne?",
        "expect": {"kind": "approximate"},
    }
    bound = [Truth("FAM133N", [7655], ["2026"])]
    assert classify(approximate, _result(NO_DATA_ANSWER), bound) == "edge_behaviour_wrong"
    future = {
        "question": "Hvor mange indbyggere havde Gentofte Kommune i 1. kvartal 2030?",
        "expect": {"kind": "not_published"},
    }
    assert (
        classify(future, _result("1. kvartal 2030 er ikke offentliggjort endnu."), []) == "correct"
    )
    assert classify(future, _result("75.241"), []) == "edge_behaviour_wrong"
    broad = {"question": "Hvordan går det med Danmark?", "expect": {"kind": "too_broad"}}
    assert classify(broad, _result("Spørgsmålet er for bredt."), []) == "correct"
    assert classify(broad, _result("", facts=[_fact(6000000)]), []) == "edge_behaviour_wrong"
    dogs = {"question": "Hvor mange hunde bor i Lynge?", "expect": {"kind": "not_available"}}
    assert classify(dogs, _result("Statistikbanken har ikke tal for hunde."), []) == "correct"
    assert classify(dogs, _result("Der bor 400 hunde."), []) == "edge_behaviour_wrong"


def test_owner_partial_and_lower_bound() -> None:
    partial = {
        "question": "Median indkomst i Danmark vs Sverige.",
        "expect": {"kind": "partial"},
    }
    truth = [Truth("INDKP201", [351100], ["2025"])]
    good = "Medianen i Danmark er 351.100 kr. Sverige er ikke i Statistikbanken."
    assert classify(partial, _result(good), truth) == "correct"
    assert classify(partial, _result("Medianen er 351.100 kr."), truth) == "edge_behaviour_wrong"
    invented = "Danmark 351.100 og Sverige 400.000."
    assert classify(partial, _result(invented), truth) == "edge_behaviour_wrong"
    approximate = {
        "question": "Hvor mange børn bor der i København der lever med mere end 2 voksne?",
        "expect": {"kind": "approximate"},
    }
    bound = [Truth("FAM133N", [7655], ["2026"])]
    stated = "Mindst 7.655. Optællingen er ufuldstændig, fordi 6 personer og derover er topkodet."
    assert classify(approximate, _result(stated), bound) == "correct"
    assert classify(approximate, _result("Der er 7.655 børn."), bound) == "edge_behaviour_wrong"
    assert (
        classify(approximate, _result("Det kan ikke opgøres ud fra Statistikbanken."), bound)
        == "correct"
    )


def test_verifier_sums_only_the_named_cells() -> None:
    check = {
        "table": "FAM133N",
        "variables": {"OMRÅDE": ["101"], "ALDER": {"from": 0, "to": 1}},
        "time": "latest",
        "aggregate": "sum",
        "sum_where": [
            {"ANTPERSH": "4", "ANTBORNH": "1"},
            {"ANTPERSH": "6-", "ANTBORNH": "3"},
        ],
    }

    def fetch_rows(table, variables):
        assert table == "FAM133N"
        assert variables["Tid"] == ["2026"]
        assert "4" in variables["ANTPERSH"] and "6-" in variables["ANTPERSH"]
        return [
            {"ANTPERSH": "4", "ANTBORNH": "1", "ALDER": "0", "TID": "2026", "value": 10},
            {"ANTPERSH": "4", "ANTBORNH": "2", "ALDER": "0", "TID": "2026", "value": 99},
            {"ANTPERSH": "6-", "ANTBORNH": "3", "ALDER": "1", "TID": "2026", "value": 7},
        ]

    def time_codes(table):
        return "Tid", ["2025", "2026"]

    truth = evaluate_check(check, fetch_rows=fetch_rows, time_codes=time_codes)
    assert truth.values == [17]
    assert truth.periods == ["2026"]


def test_a_year_total_is_not_the_latest_month() -> None:
    check = {
        "table": "BIL55",
        "variables": {"REGBIL": ["4000101002"]},
        "time": {"year": "2025", "grain": "month"},
        "aggregate": "sum",
    }

    def fetch_rows(table, variables):
        assert variables["Tid"] == [f"2025M{month:02d}" for month in range(1, 13)]
        return [{"TID": code, "value": 1} for code in variables["Tid"]]

    def time_codes(table):
        months = [f"2025M{month:02d}" for month in range(1, 13)]
        return "Tid", months + ["2026M09"]

    truth = evaluate_check(check, fetch_rows=fetch_rows, time_codes=time_codes)
    assert truth.values == [12]
    assert truth.periods == ["2025"]


def test_cost_uses_list_prices_and_counts_reasoning_as_output() -> None:
    prices = prices_from_env()
    # 10 fresh input tokens, 90 cached, 4 completion, 6 reasoning.
    spent = cost_usd(prompt=100, cached=90, completion=4, reasoning=6, prices=prices)
    expected = 10 * 1.25 / 1_000_000 + 90 * 0.20 / 1_000_000 + 10 * 2.50 / 1_000_000
    assert spent == expected
    assert "docs.x.ai" in prices.source


def test_report_headline_counts_only_correct() -> None:
    prices = prices_from_env()
    records = [
        {
            "id": 1,
            "category": "municipality",
            "outcome": "correct",
            "answer": "58.381",
            "tables": ["FOLK1A"],
            "latency_s": 10,
            "prompt_tokens": 100,
            "cached_tokens": 40,
            "completion_tokens": 5,
            "reasoning_tokens": 5,
            "source": "regression",
        },
        {
            "id": 63,
            "category": "annual_aggregate",
            "outcome": "wrong_period",
            "answer": "20.064 i september",
            "tables": ["BIL55"],
            "latency_s": 30,
            "prompt_tokens": 100,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "source": "regression",
        },
    ]
    report = render_report(
        records,
        {
            "date": "2026-11-03",
            "model": "grok-4",
            "served_model": "grok-4.3",
            "set_size": 105,
            "prices": prices,
        },
    )
    assert "**1/2 correct**" in report
    assert "| timeout/error | 0 |" in report
    assert "| wrong_period | 1 |" in report
    assert "Q63" in report


def test_newest_projection_vintage_is_the_highest_matching_id() -> None:
    from evals.verify import newest_table_id

    tables = [
        {"id": "FRDK125", "text": "Befolkningsfremskrivning 2025 for hele landet"},
        {"id": "FRDK126", "text": "Befolkningsfremskrivning 2026 for hele landet"},
        {"id": "FRDK226", "text": "Nøgletal 2026: Fremskrivning af befolkningens bevægelser"},
        {"id": "FRKM126", "text": "Befolkningsfremskrivning 2026"},
    ]
    spec = {
        "prefix": "FRDK",
        "title": ["befolkningsfremskrivning", "hele landet"],
        "not_title": ["nøgletal"],
    }
    assert newest_table_id(tables, spec) == "FRDK126"


def test_a_contradiction_about_a_fetched_period_is_not_accepted() -> None:
    question = {
        "question": "Hvor meget er befolkningen i Viborg Kommune vokset siden 2000?",
        "expect": {
            "kind": "coverage_gap",
            "require_any": ["2008", "2007", "starter", "reform"],
        },
    }
    truths = [Truth("FOLK1A", [92084.0, 98104.0], ["2008K1", "2026K3"])]
    denied = (
        "92.084 (2008K1) og 98.104 (2026K3). Forskellen er 6.020. "
        "Serien starter i 2008. Seneste tal (2026K3) er ikke med."
    )
    assert classify(question, _result(denied), truths) == "edge_behaviour_wrong"
    consistent = (
        "92.084 (2008K1) og 98.104 (2026K3). Forskellen er 6.020 (6,54 procent). "
        "Viborg i FOLK1A starter i 2008K1, kommunegrænsen efter kommunalreformen i 2007. "
        "2000 er ikke med i denne serie."
    )
    assert classify(question, _result(consistent), truths) == "correct"
    shown = (
        "92.084 (2008 1. kvt.) og 98.104 (2026 3. kvt.). "
        "Forskellen er 6.020 (6,54 procent). "
        "Viborg i FOLK1A starter 2008 1. kvt., kommunegrænsen efter kommunalreformen i 2007. "
        "2000 er ikke med i denne serie."
    )
    assert classify(question, _result(shown), truths) == "correct"
    hidden = (
        "I 2008K1 var befolkningen 92.084. "
        "Seneste tal (2026K3) er ikke vist i resultatet. "
        "Viborg i FOLK1A starter 2008 1. kvt., kommunegrænsen efter kommunalreformen i 2007. "
        "2000 er ikke med i denne serie."
    )
    assert classify(question, _result(hidden), truths) == "edge_behaviour_wrong"
    bare = "92.084 (2008 1. kvt.) og 98.104 (2026 3. kvt.). Forskellen er 6.020 (6,54 procent)."
    strict = {
        "question": question["question"],
        "expect": {
            "kind": "coverage_gap",
            "require_any": ["2008", "2007", "starter", "reform"],
            "require_all": [
                ["2007", "kommunalreform", "kommunegræn"],
                ["ikke med", "før 2008", "foer 2008"],
            ],
        },
    }
    assert classify(strict, _result(bare), truths) == "edge_behaviour_wrong"
    assert classify(strict, _result(shown), truths) == "correct"


def test_coverage_gap_requires_an_explanation() -> None:
    question = {
        "question": "Hvor mange boede der i Danmark ifølge folketællingen i 1934?",
        "expect": {
            "kind": "coverage_gap",
            "require_any": ["1930", "1935", "nærmeste", "folketælling"],
        },
    }
    assert classify(question, _result(NO_DATA_ANSWER), []) == "edge_behaviour_wrong"
    explained = "FT har ikke 1934. Nærmeste perioder i tabellen er 1930 og 1935."
    assert classify(question, _result(explained), []) == "correct"


def test_question_file_is_the_fixed_set() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    assert [item["id"] for item in questions] == list(range(1, 131)) + list(range(201, 301))
    by_id = {item["id"]: item for item in questions}
    growth = {
        106: ("Hvor meget er befolkningen i danmark vokset siden 1989", "1989"),
        107: ("Hvor meget er befolkningen i danmark vokset siden 1934", "1934"),
    }
    for question_id, (text, year) in growth.items():
        item = by_id[question_id]
        assert item["question"] == text
        assert item["category"] == "time_series"
        assert item.get("source") is None
        expect = item["expect"]
        assert expect["kind"] == "figures"
        assert len(expect["checks"]) == 1
        check = expect["checks"][0]
        assert check["table"] == "BEFOLK2"
        assert check["variables"] == {"KØN": ["TOT"], "ALDER": ["TOT"]}
        assert check["time"] == [year, "latest"]
        assert check["aggregate"] == "values"
    probes = {
        108: "Hvor mange indbyggere havde Danmark 1. januar 1925?",
        109: "Hvor mange boede der i Danmark ifølge folketællingen i 1934?",
        110: "Hvor meget er befolkningen i Viborg Kommune vokset siden 2000?",
        111: (
            "Hvor mange flere indbyggere har Esbjerg Kommune i seneste kvartal end 1. januar 2008?"
        ),
        112: "Hvor mange nettoledige var der i Danmark i august 2026, sæsonkorrigeret?",
        113: (
            "Hvor mange indbyggere forventes Danmark at have i 2040 "
            "ifølge befolkningsfremskrivningen?"
        ),
        114: "Hvad var Danmarks BNP i løbende priser i 2024?",
        115: "Hvad var den reale vækst i Danmarks BNP i 2024?",
        116: "Hvor mange fuldtidsledige var der i gennemsnit i Esbjerg Kommune i 2025?",
        117: "Hvor mange indbyggere havde Viborg Kommune 1. januar 2026?",
        118: "Hvor mange kvinder bor der i Viborg Kommune?",
        119: "Hvor mange personer på 65 år og derover bor der i Esbjerg Kommune?",
    }
    for question_id, text in probes.items():
        assert by_id[question_id]["question"] == text
        scratch_id = question_id - 108 + 201
        assert f"scratch {scratch_id}" in by_id[question_id]["note"].casefold()
    assert by_id[109]["expect"]["kind"] == "coverage_gap"
    assert by_id[110]["expect"]["kind"] == "coverage_gap"
    assert by_id[110]["expect"]["require_all"][0][0] == "2007"
    assert "ikke med" in by_id[110]["expect"]["require_all"][1]
    assert by_id[112]["expect"]["checks"][0]["table"] == "AUS07"
    assert by_id[112]["expect"]["checks"][0]["variables"]["YD"] == ["NET"]
    assert by_id[112]["expect"]["checks"][0]["variables"]["SAESONFAK"] == ["10"]
    assert by_id[113]["expect"]["checks"][0]["table"]["prefix"] == "FRDK"
    assert by_id[116]["expect"]["checks"][0]["table"] == "AUL01"
    assert any(check["table"] == "AUL01" for check in by_id[50]["expect"]["checks"])
    assert by_id[63]["expect"]["checks"][0]["aggregate"] == "sum"
    assert by_id[63]["expect"]["checks"][0]["time"]["year"] == "2025"
    owner = [item for item in questions if item.get("source") == "owner"]
    assert [item["id"] for item in owner] == [101, 102, 103, 104, 105]
    assert {item["category"] for item in owner} == {"hard/ambiguous"}
    assert by_id[101]["expect"]["kind"] == "approximate"
    assert by_id[101]["expect"]["checks"][0]["table"] == "FAM133N"
    assert by_id[102]["expect"]["checks"][0]["variables"]["FAMTYP"] == ["EKMB"]
    assert by_id[103]["expect"]["kind"] == "not_available"
    assert by_id[103]["expect"].get("checks") is None
    assert by_id[104]["expect"]["kind"] == "partial"
    assert by_id[104]["expect"]["checks"][0]["table"] == "INDKP201"
    ages = by_id[105]["expect"]["checks"][0]["variables"]["ALDER"]
    assert ages == {"from": 23, "to": 45}
    assert by_id[80]["expect"]["kind"] == "no_table"
    assert by_id[96]["expect"]["kind"] == "too_broad"
    assert any(
        check["variables"].get("SAESONFAK") == ["10"] for check in by_id[95]["expect"]["checks"]
    )
    assert any(
        check["variables"].get("SAESONFAK") == ["10"] for check in by_id[50]["expect"]["checks"]
    )
    assert any(check["table"] == "INDOPH1" for check in by_id[33]["expect"]["checks"])
    change_1995 = {
        120: "Hvor meget steg befolkningen i danmark i 1995 til 2001",
        121: "Hvor meget voksede Danmarks befolkning fra 1995 til 2001?",
        122: "Hvor mange flere indbyggere var der i Danmark i 2001 end i 1995?",
        123: (
            "Hvad var befolkningstilvæksten i Danmark mellem 1995 og 2001, "
            "både i personer og procent?"
        ),
        124: "Hvor meget steg befolkningen i Danmark i perioden 1995–2001?",
        125: "Hvor stor var forskellen på Danmarks befolkning i 1995 og 2001?",
    }
    for question_id, text in change_1995.items():
        item = by_id[question_id]
        assert item["question"] == text
        assert item["expect"]["kind"] == "figures"
        assert len(item["expect"]["checks"]) == 2
        wants_percent = "procent" in text.casefold()
        for check in item["expect"]["checks"]:
            assert check["time"] == ["1995", "2001"]
            assert check["also"] == (["difference", "percent"] if wants_percent else ["difference"])
            assert check["variables"]["KØN"] == ["TOT"]
            assert check["variables"]["ALDER"] == ["TOT"]
        assert item["expect"]["checks"][0]["table"] == "BEFOLK1"
        assert item["expect"]["checks"][1]["table"] == "BEFOLK2"
    esbjerg = by_id[126]
    assert esbjerg["question"] == "Hvor meget mange mennesker bor der i esbjerg"
    assert esbjerg["expect"]["checks"][0]["table"] == "FOLK1A"
    assert esbjerg["expect"]["checks"][0]["variables"]["OMRÅDE"] == ["561"]
    assert "befolkning" in esbjerg["expect"]["require_all"][0]
    assert "kommune" in esbjerg["expect"]["require_all"][1]
    cities = {
        127: "Hvilken by er danmarks tiende største",
        128: "Hvilken by ligger på tiendepladsen i Danmark målt på indbyggertal?",
        129: "Hvad hedder Danmarks tiendestørste by, og hvor mange indbyggere har den?",
        130: "Hvilke ti byer i Danmark har flest indbyggere, i faldende rækkefølge?",
    }
    for question_id, text in cities.items():
        item = by_id[question_id]
        assert item["question"] == text
        check = item["expect"]["checks"][0]
        assert check["table"] == "BY3"
        assert check["aggregate"] == "urban_rank"
        assert check["variables"]["FOLKARTAET"] == ["FOLKETAL"]
    assert by_id[127]["expect"]["checks"][0]["rank"] == 10
    assert by_id[130]["expect"]["checks"][0]["rank"] == [1, 10]
    added = [item for item in questions if 201 <= item["id"] <= 300]
    assert [item["id"] for item in added] == list(range(201, 301))
    assert by_id[231]["expect"]["checks"][0]["table"] == "INDKP105"
    assert by_id[233]["expect"]["checks"][0]["table"] == "PRIS111"
    assert by_id[252]["expect"]["checks"][0]["rank"] == 2
    assert by_id[243]["expect"]["checks"][0]["table"] == "NAN1"
    assert by_id[292]["expect"]["kind"] == "coverage_gap"


def test_a_transient_group_is_retried_once_and_the_leaf_is_recorded(monkeypatch) -> None:
    from evals.run import _run_one

    from statistikbanken_mcp.web import LLM_TIMEOUT_TEXT, LLMError, Settings, TurnResult

    calls = {"n": 0}

    def once(_settings, _question):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ExceptionGroup(
                "unhandled errors in a TaskGroup",
                [LLMError(LLM_TIMEOUT_TEXT)],
            )
        return TurnResult(
            answer="58.381",
            facts=[{"table": "FOLK1A", "period": "2026K3", "value": 58381}],
            calls=[],
        )

    monkeypatch.setattr("statistikbanken_mcp.web._answer_once", once)
    monkeypatch.setattr(
        "evals.run.truths_for",
        lambda question, client=None: [Truth("FOLK1A", [58381.0], ["2026K3"])],
    )
    settings = Settings(
        llm="xai",
        base_url="https://api.x.ai/v1",
        model="grok-4",
        api_key="test-key",
        host="127.0.0.1",
        port=8080,
        daily_cap=1,
        per_minute=1,
        max_question_chars=300,
        max_tool_calls=8,
        llm_timeout=45,
        tool_timeout=30,
    )
    question = {
        "id": 1,
        "question": "Hvor mange?",
        "category": "municipality",
        "expect": {"kind": "figures", "checks": [{"table": "FOLK1A"}]},
    }
    record = _run_one(settings, question)
    assert calls["n"] == 2
    assert record["outcome"] == "correct"
    assert record["error"] is None

    calls["n"] = 0

    def always_fail(_settings, _question):
        calls["n"] += 1
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [LLMError(LLM_TIMEOUT_TEXT)],
        )

    monkeypatch.setattr("statistikbanken_mcp.web._answer_once", always_fail)
    failed = _run_one(settings, question)
    assert calls["n"] == 2
    assert failed["outcome"] == "timeout_error"
    assert "ExceptionGroup" not in (failed["error"] or "")
    assert "LLMError" in (failed["error"] or "")

    calls["n"] = 0

    def not_transient(_settings, _question):
        calls["n"] += 1
        raise ValueError("nope")

    monkeypatch.setattr("statistikbanken_mcp.web._answer_once", not_transient)
    plain = _run_one(settings, question)
    assert calls["n"] == 1
    assert "ValueError" in (plain["error"] or "")


def test_million_and_billion_kroner_are_the_same_figure() -> None:
    """NAHL2 states millions. NAN1 states billions. 3,118,538 mio. is 3,118.5 mia."""
    question = {
        "question": "Hvad var Danmarks BNP i løbende priser i 2025?",
        "expect": {"kind": "figures"},
    }
    truth = [Truth("NAN1", [3118.5], ["2025"])]
    stated = _result("3.118.538 mio. kr. (NAHL2, BNP, løbende priser)")
    assert classify(question, stated, truth) == "correct"
    billion = _result("3.118,5 mia. kr.")
    assert classify(question, billion, truth) == "correct"


def test_change_check_adds_the_difference_and_percent() -> None:
    """1995 and 2001 are not enough. The requested change has to be in the truth."""

    def time_codes(_table: str) -> tuple[str, list[str]]:
        return "Tid", ["1995", "2001"]

    def fetch_rows(_table: str, _variables: dict) -> list[dict]:
        return [
            {"Tid": "1995", "value": 5_215_718},
            {"Tid": "2001", "value": 5_349_212},
        ]

    truth = evaluate_check(
        {
            "table": "BEFOLK1",
            "variables": {"KØN": ["TOT"], "ALDER": ["TOT"]},
            "time": ["1995", "2001"],
            "aggregate": "values",
            "also": ["difference", "percent"],
        },
        fetch_rows=fetch_rows,
        time_codes=time_codes,
    )
    assert truth.values[:2] == [5_215_718, 5_349_212]
    assert truth.values[2] == 133_494
    assert abs(truth.values[3] - (133_494 / 5_215_718 * 100)) < 1e-9
    question = {
        "question": "Hvor meget steg befolkningen i danmark i 1995 til 2001",
        "expect": {"kind": "figures"},
    }
    only_end = _result("5.349.212 (BEFOLK1, 2001, Antal, Alder i alt)")
    assert classify(question, only_end, [truth]) != "correct"
    full = _result("5.215.718 (1995) og 5.349.212 (2001). Forskellen er 133.494 (2,56 procent).")
    assert classify(question, full, [truth]) == "correct"


def test_a_city_number_without_the_urban_area_is_not_correct() -> None:
    question = {
        "question": "Hvilken by er danmarks tiende største",
        "expect": {
            "kind": "figures",
            "require_all": [["Roskilde"], ["byområde", "byomraade"]],
        },
    }
    truth = [Truth("BY3", [53897.0], ["2026"])]
    dumped = _result("6.031.699 (FOLK1A, Hele landet). 53.897")
    assert classify(question, dumped, truth) != "correct"
    named = _result("Roskilde er den 10. største by med 53.897 indbyggere (byområde).")
    assert classify(question, named, truth) == "correct"


def test_rank_names_have_to_appear_in_order() -> None:
    question = {"question": "Hvilke ti byer", "expect": {"kind": "figures"}}
    try:
        truth = [Truth("BY3", [150.0, 140.0], ["2026"], ("Hovedstadsområdet", "Aarhus"))]
    except TypeError:
        raise AssertionError("Truth does not carry the ranked names") from None
    swapped = _result("Aarhus 140 og Hovedstadsområdet 150 (byområde).")
    assert classify(question, swapped, truth) != "correct"
    ordered = _result("1. Hovedstadsområdet: 150\n2. Aarhus: 140\nDet er byområder.")
    assert classify(question, ordered, truth) == "correct"


def test_urban_rank_check_is_the_tenth_city_not_every_row() -> None:
    rows = [
        {"BYER": "101-01100 København (del af Hovedstadsområdet)", "Tid": "2026", "value": 100},
        {"BYER": "147-01100 Frederiksberg (del af Hovedstadsområdet)", "Tid": "2026", "value": 50},
        {"BYER": "000 Hovedstadsområdet", "Tid": "2026", "value": 0},
        {"BYER": "000 Landdistrikter", "Tid": "2026", "value": 999},
        {"BYER": "751-11045 Aarhus", "Tid": "2026", "value": 90},
        {"BYER": "461-10677 Odense", "Tid": "2026", "value": 80},
        {"BYER": "851-10938 Aalborg", "Tid": "2026", "value": 70},
        {"BYER": "561-11196 Esbjerg", "Tid": "2026", "value": 60},
        {"BYER": "615-10615 Horsens", "Tid": "2026", "value": 50},
        {"BYER": "730-10691 Randers", "Tid": "2026", "value": 40},
        {"BYER": "621-10588 Kolding", "Tid": "2026", "value": 30},
        {"BYER": "630-10722 Vejle", "Tid": "2026", "value": 20},
        {"BYER": "265-10850 Roskilde", "Tid": "2026", "value": 15},
        {"BYER": "740-10820 Silkeborg", "Tid": "2026", "value": 10},
    ]

    def time_codes(_table: str) -> tuple[str, list[str]]:
        return "Tid", ["2026"]

    def fetch_rows(_table: str, _variables: dict) -> list[dict]:
        return rows

    truth = evaluate_check(
        {
            "table": "BY3",
            "variables": {"FOLKARTAET": ["FOLKETAL"], "BYER": ["*"]},
            "time": ["2026"],
            "aggregate": "urban_rank",
            "rank": 10,
        },
        fetch_rows=fetch_rows,
        time_codes=time_codes,
    )
    assert truth.names == ("Roskilde",)
    assert truth.values == [15]
    listed = evaluate_check(
        {
            "table": "BY3",
            "variables": {"FOLKARTAET": ["FOLKETAL"], "BYER": ["*"]},
            "time": ["2026"],
            "aggregate": "urban_rank",
            "rank": [1, 10],
        },
        fetch_rows=fetch_rows,
        time_codes=time_codes,
    )
    assert listed.names[0] == "Hovedstadsområdet"
    assert listed.names[9] == "Roskilde"
    assert listed.values[0] == 150
    assert "Landdistrikter" not in listed.names
    assert all("del af" not in name for name in listed.names)
    with pytest.raises(VerifierError):
        evaluate_check(
            {
                "table": "BY3",
                "variables": {"FOLKARTAET": ["FOLKETAL"], "BYER": ["*"]},
                "time": ["2026"],
                "aggregate": "urban_rank",
                "rank": 99,
            },
            fetch_rows=fetch_rows,
            time_codes=time_codes,
        )
