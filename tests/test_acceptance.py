"""The questions this server exists to answer, on recorded responses.

Population figures are the FOLK1A release of 2026-08-10 (latest period 2026K3).
Income is INDKP105 for 2025, updated 2026-10-02. Sports clubs are IDRFOR01 for 2025.
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


def test_children_0_to_17_in_rudersdal_and_horsholm(calls: list[dict[str, object]]) -> None:
    _folk1a()
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["230", "223"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["2026K3"],
        },
        sum_range={"ALDER": ["0", "17"]},
    )
    assert result["unit"] == "Antal"
    assert result["periods"] == ["2026K3"]
    assert result["rows"] == [
        {
            "OMRÅDE": "Rudersdal",
            "KØN": "I alt",
            "ALDER": "0-17",
            "CIVILSTAND": "I alt",
            "Tid": "2026K3",
            "value": 12770,
        },
        {
            "OMRÅDE": "Hørsholm",
            "KØN": "I alt",
            "ALDER": "0-17",
            "CIVILSTAND": "I alt",
            "Tid": "2026K3",
            "value": 4916,
        },
    ]
    alder = _values(_data_body(calls), "ALDER")
    assert len(alder) == 1
    assert alder[0].startswith("sum(0-17|0-17=0;1;")
    assert alder[0].endswith(";17)")


def test_average_income_in_rudersdal_and_horsholm() -> None:
    hits = find_tabel("indkomst", limit=15)
    match = next(hit for hit in hits if hit["id"] == "INDKP105")
    assert match["title"] == "Indkomst i alt"
    assert match["unit"] == "-"
    assert match["latest_period"] == "2025"
    info = tabelinfo("INDKP105")
    enhed = next(item for item in info["variables"] if item["id"] == "ENHED")
    labels = {item["id"]: item["text"] for item in enhed["values"]}
    assert labels["118"] == "Gennemsnit for personer i gruppen (kr.)"
    result = hent_data(
        "INDKP105",
        values={
            "OMRÅDE": ["230", "223"],
            "ENHED": ["118"],
            "KOEN": ["MOK"],
            "ALDER1": ["00"],
            "INDKINTB": ["000"],
            "Tid": ["2025"],
        },
    )
    assert result["title"] == "Indkomst i alt"
    assert result["unit"] == "-"
    assert result["periods"] == ["2025"]
    by_name = {row["OMRÅDE"]: row for row in result["rows"]}
    assert set(by_name) == {"Rudersdal", "Hørsholm"}
    for row in result["rows"]:
        assert row["ENHED"] == "Gennemsnit for personer i gruppen (kr.)"
        assert row["Tid"] == "2025"
    assert by_name["Rudersdal"]["value"] == 757969
    assert by_name["Hørsholm"]["value"] == 679635


def test_association_membership_nearest_table_is_sports_clubs() -> None:
    hits = find_tabel("forening")
    by_id = {hit["id"]: hit for hit in hits}
    assert by_id["IDRFOR01"]["title"] == "Idrætsforeninger og medlemmer"
    assert by_id["IDRFOR01"]["unit"] == "Antal"
    assert by_id["IDRFOR01"]["latest_period"] == "2025"
    assert by_id["KV2FR3"]["title"] == "Foreningsmedlemskab"
    info = tabelinfo("IDRFOR01")
    assert info["title"] == "Idrætsforeninger og medlemmer"
    assert "Idrætsforeninger" in info["description"]
    key_figures = next(item for item in info["variables"] if item["id"] == "AKTP")
    labels = {item["id"]: item["text"] for item in key_figures["values"]}
    assert labels == {"104": "Foreninger", "105": "Medlemmer"}
    result = hent_data(
        "IDRFOR01",
        values={
            "BLSTKOM": ["230"],
            "ORGANISATION": ["100"],
            "AKTP": ["104", "105"],
            "Tid": ["2025"],
        },
    )
    assert "idræt" in result["title"].casefold()
    assert "idræt" in result["description"].casefold()
    assert result["unit"] == "Antal"
    assert result["periods"] == ["2025"]
    by_measure = {row["AKTP"]: row["value"] for row in result["rows"]}
    assert by_measure == {"Foreninger": 83, "Medlemmer": 29632}
    assert all(row["BLSTKOM"] == "Rudersdal" and row["Tid"] == "2025" for row in result["rows"])


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
