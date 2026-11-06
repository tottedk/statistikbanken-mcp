"""The first-answer example calls the live API and prints a cited figure."""

from __future__ import annotations

import pytest

from statistikbanken_mcp.statbank import StatbankClient

pytestmark = pytest.mark.live


def test_esbjerg_80plus_example_prints_the_live_figure(capsys: pytest.CaptureFixture[str]) -> None:
    from examples.esbjerg_80plus import main

    client = StatbankClient()
    try:
        result = client.fetch(
            "FOLK1A",
            values={
                "OMRÅDE": ["Esbjerg"],
                "KØN": ["TOT"],
                "CIVILSTAND": ["TOT"],
                "Tid": ["(1)"],
            },
            sum_from={"ALDER": "80"},
        )
    finally:
        client.close()

    assert result["table"] == "FOLK1A"
    assert result["rows"], "FOLK1A Esbjerg 80+ is published"
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["OMRÅDE"] == "Esbjerg"
    assert row["ALDER"] == "80+"
    value = row["value"]
    period = row["Tid"]
    assert isinstance(value, int) and value > 0
    assert isinstance(period, str) and period
    # 7,556 is the published 2026K3 figure. A later quarter is the live figure.
    if period == "2026K3":
        assert value == 7556
    else:
        assert period > "2026K3"

    assert main() == 0
    out = capsys.readouterr().out
    assert "FOLK1A" in out
    assert period in out
    assert str(value) in out
    assert "Danmarks Statistik" in out
    assert "Statistikbanken" in out
