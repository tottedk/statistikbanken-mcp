"""Print people aged 80 and over in Esbjerg from FOLK1A.

No language model and no MCP client. The figure, period, and source come from
the public Statistikbanken API. A newer published quarter is printed as-is.
"""

from __future__ import annotations

import sys

from statistikbanken_mcp.statbank import StatbankClient, StatbankError


def main() -> int:
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
    except StatbankError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        client.close()

    rows = result["rows"]
    if len(rows) != 1:
        print(f"Expected one FOLK1A row, got {len(rows)}.", file=sys.stderr)
        return 1
    row = rows[0]
    table = result["table"]
    period = row["Tid"]
    value = row["value"]
    print(
        f"{table} {period}: {value} people aged {row['ALDER']} in {row['OMRÅDE']}.\n"
        f"Source: Danmarks Statistik, Statistikbanken, table {table}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
