"""The three questions this server exists to answer, on recorded responses.

Numbers are the FOLK1A release of 2026-08-10 (latest period 2026K3).
"""

from statistikbanken_mcp.server import find_tabel, hent_data, tabelinfo

# Same quarter, ten years apart, plus the quarters in between. 2023K3 dips.
RUDERSDAL_K3 = {
    "2016K3": 55915,
    "2017K3": 56060,
    "2018K3": 56134,
    "2019K3": 56556,
    "2020K3": 56623,
    "2021K3": 57195,
    "2022K3": 57424,
    "2023K3": 57055,
    "2024K3": 57308,
    "2025K3": 57386,
    "2026K3": 58381,
}


def _folk1a() -> None:
    hits = find_tabel("befolkning")
    assert hits[0]["id"] == "FOLK1A"
    info = tabelinfo("FOLK1A")
    ages = next(item for item in info["variables"] if item["id"] == "ALDER")
    assert ages["numeric_codes"] == {"from": "0", "to": "125"}


def test_over_80_in_rudersdal(calls: list[dict[str, object]]) -> None:
    _folk1a()
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["230"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["(1)"],
        },
        sum_from={"ALDER": "80"},
    )
    assert result["unit"] == "Antal"
    assert result["rows"] == [
        {
            "OMRÅDE": "Rudersdal",
            "KØN": "I alt",
            "ALDER": "80+",
            "CIVILSTAND": "I alt",
            "Tid": "2026K3",
            "value": 5045,
        }
    ]
    sent = _data_body(calls)
    alder = _values(sent, "ALDER")
    assert len(alder) == 1
    assert alder[0].startswith("sum(80+|80+=80;81;")
    assert alder[0].endswith(";125)")
    assert ">=" not in alder[0]


def test_rudersdal_population_over_ten_years() -> None:
    _folk1a()
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["230"],
            "KØN": ["TOT"],
            "ALDER": ["IALT"],
            "CIVILSTAND": ["TOT"],
            "Tid": [">=2016K3"],
        },
    )
    rows = result["rows"]
    assert rows[0]["Tid"] == "2016K3"
    assert rows[-1]["Tid"] == "2026K3"
    assert len(rows) == 41
    assert all(row["OMRÅDE"] == "Rudersdal" for row in rows)
    assert all(row["ALDER"] == "Alder i alt" for row in rows)
    by_period = {row["Tid"]: row["value"] for row in rows}
    for period, value in RUDERSDAL_K3.items():
        assert by_period[period] == value
    assert by_period["2026K3"] - by_period["2016K3"] == 2466
    assert by_period["2023K3"] < by_period["2022K3"]


def test_which_municipality_has_the_most_people_over_80() -> None:
    _folk1a()
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["230", "223", "157"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["(1)"],
        },
        sum_from={"ALDER": "80"},
    )
    assert len(result["rows"]) == 3
    assert all(row["ALDER"] == "80+" and row["Tid"] == "2026K3" for row in result["rows"])
    by_name = {row["OMRÅDE"]: row["value"] for row in result["rows"]}
    assert by_name == {"Rudersdal": 5045, "Hørsholm": 2790, "Gentofte": 5486}
    assert max(by_name, key=by_name.get) == "Gentofte"


def _data_body(calls: list[dict[str, object]]) -> dict[str, object]:
    bodies = [call["body"] for call in calls if str(call["path"]).endswith("/data")]
    assert bodies
    body = bodies[-1]
    assert isinstance(body, dict)
    return body


def _values(body: dict[str, object], code: str) -> list[str]:
    variables = body["variables"]
    assert isinstance(variables, list)
    for item in variables:
        assert isinstance(item, dict)
        if item["code"] == code:
            values = item["values"]
            assert isinstance(values, list)
            return [str(value) for value in values]
    raise AssertionError(code)
