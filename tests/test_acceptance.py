"""The questions this server exists to answer, on recorded responses.

Population figures are the FOLK1A release of 2026-08-10 (latest period 2026K3).
Income is INDKP105 for 2025, updated 2026-10-02. Sports clubs are IDRFOR01 for 2025.
The same figures were checked against the live API on 2026-11-01, including
Viborg in IDRFOR01 (208 clubs, 49821 memberships), which is not in the
Esbjerg recording below.
"""

from statistikbanken_mcp.server import find_tabel, hent_data, tabelinfo

# Official wording. The tests below keep their own asserts; this list is the
# question a person actually asks, with the figure that question should return.
ACCEPTANCE_QUESTIONS: tuple[tuple[str, str], ...] = (
    (
        "Hvor mange over 80 bor i Esbjerg?",
        "FOLK1A 2026K3: 7556",
    ),
    (
        "Hvordan har befolkningstallet i Viborg udviklet sig de seneste 10 år?",
        "FOLK1A: 96166 i 2016K3, 98104 i 2026K3",
    ),
    (
        "Hvor mange over 80 bor i Gentofte, Esbjerg og Viborg?",
        "FOLK1A 2026K3: Gentofte 5486, Esbjerg 7556, Viborg 6420",
    ),
    (
        "Hvor mange foreningsmedlemmer er der i Esbjerg?",
        "IDRFOR01 2025: Esbjerg 212 foreninger og 54205 medlemmer; Viborg 208 og 49821",
    ),
    (
        "Hvor mange børn mellem 0 og 17 år bor i Esbjerg og Viborg?",
        "FOLK1A 2026K3: Esbjerg 21616, Viborg 19171",
    ),
    (
        "Hvad er den gennemsnitlige indkomst i alt før skat i Esbjerg og Viborg?",
        "INDKP105 2025: Esbjerg 407775, Viborg 402605",
    ),
)

# Same quarter, ten years apart, plus the quarters in between. 2020K3 dips.
VIBORG_K3 = {
    "2016K3": 96166,
    "2017K3": 96884,
    "2018K3": 97061,
    "2019K3": 97245,
    "2020K3": 96706,
    "2021K3": 96696,
    "2022K3": 97455,
    "2023K3": 97564,
    "2024K3": 97659,
    "2025K3": 97867,
    "2026K3": 98104,
}


def _folk1a() -> None:
    hits = find_tabel("befolkning")
    assert hits[0]["id"] == "FOLK1A"
    info = tabelinfo("FOLK1A")
    ages = next(item for item in info["variables"] if item["id"] == "ALDER")
    assert ages["numeric_codes"] == {"from": "0", "to": "125"}


def test_over_80_in_esbjerg(calls: list[dict[str, object]]) -> None:
    _folk1a()
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["561"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["(1)"],
        },
        sum_from={"ALDER": "80"},
    )
    assert result["unit"] == "Antal"
    assert result["rows"] == [
        {
            "OMRÅDE": "Esbjerg",
            "KØN": "I alt",
            "ALDER": "80+",
            "CIVILSTAND": "I alt",
            "Tid": "2026K3",
            "value": 7556,
        }
    ]
    sent = _data_body(calls)
    alder = _values(sent, "ALDER")
    assert len(alder) == 1
    assert alder[0].startswith("sum(80+|80+=80;81;")
    assert alder[0].endswith(";125)")
    assert ">=" not in alder[0]


def test_viborg_population_over_ten_years() -> None:
    _folk1a()
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["791"],
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
    assert all(row["OMRÅDE"] == "Viborg" for row in rows)
    assert all(row["ALDER"] == "Alder i alt" for row in rows)
    by_period = {row["Tid"]: row["value"] for row in rows}
    for period, value in VIBORG_K3.items():
        assert by_period[period] == value
    assert by_period["2026K3"] - by_period["2016K3"] == 1938
    assert by_period["2020K3"] < by_period["2019K3"]


def test_which_municipality_has_the_most_people_over_80() -> None:
    _folk1a()
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["561", "791", "157"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["(1)"],
        },
        sum_from={"ALDER": "80"},
    )
    assert len(result["rows"]) == 3
    assert all(row["ALDER"] == "80+" and row["Tid"] == "2026K3" for row in result["rows"])
    by_name = {row["OMRÅDE"]: row["value"] for row in result["rows"]}
    assert by_name == {"Esbjerg": 7556, "Viborg": 6420, "Gentofte": 5486}
    assert max(by_name, key=by_name.get) == "Esbjerg"


def test_children_0_to_17_in_esbjerg_and_viborg(calls: list[dict[str, object]]) -> None:
    _folk1a()
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["561", "791"],
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
            "OMRÅDE": "Esbjerg",
            "KØN": "I alt",
            "ALDER": "0-17",
            "CIVILSTAND": "I alt",
            "Tid": "2026K3",
            "value": 21616,
        },
        {
            "OMRÅDE": "Viborg",
            "KØN": "I alt",
            "ALDER": "0-17",
            "CIVILSTAND": "I alt",
            "Tid": "2026K3",
            "value": 19171,
        },
    ]
    alder = _values(_data_body(calls), "ALDER")
    assert len(alder) == 1
    assert alder[0].startswith("sum(0-17|0-17=0;1;")
    assert alder[0].endswith(";17)")


def test_average_income_in_esbjerg_and_viborg() -> None:
    question = ACCEPTANCE_QUESTIONS[5][0]
    assert question.startswith("Hvad er den gennemsnitlige indkomst")
    ids = [hit["id"] for hit in find_tabel(question, limit=3)]
    assert "INDKP105" in ids
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
            "OMRÅDE": ["561", "791"],
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
    assert set(by_name) == {"Esbjerg", "Viborg"}
    for row in result["rows"]:
        assert row["ENHED"] == "Gennemsnit for personer i gruppen (kr.)"
        assert row["Tid"] == "2025"
    assert by_name["Esbjerg"]["value"] == 407775
    assert by_name["Viborg"]["value"] == 402605


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
            "BLSTKOM": ["561"],
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
    assert by_measure == {"Foreninger": 212, "Medlemmer": 54205}
    assert all(row["BLSTKOM"] == "Esbjerg" and row["Tid"] == "2025" for row in result["rows"])


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
