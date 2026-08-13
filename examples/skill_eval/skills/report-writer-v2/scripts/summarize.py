#!/usr/bin/env python3
"""Per-period totals for a two-column CSV, as JSON.

Bundled with the v2 skill. The point of it in this example is not the
arithmetic — it is that ``skill_trigger``'s ``required_resources`` can tell
"the agent ran the script we shipped" apart from "the agent wrote its own
loop again", which is the difference between a skill that saves work and a
skill that only looks like it does.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize.py <csv>", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8")))
    if not rows:
        print("empty csv", file=sys.stderr)
        return 1

    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[row["period"]] += float(row["amount"])

    periods = sorted(totals)
    latest, previous = periods[-1], (periods[-2] if len(periods) > 1 else None)
    delta = totals[latest] - totals[previous] if previous else 0.0

    print(
        json.dumps(
            {
                "totals": {p: round(totals[p], 2) for p in periods},
                "latest": latest,
                "delta": round(delta, 2),
                "delta_pct": (
                    round(delta / totals[previous] * 100, 1)
                    if previous and totals[previous]
                    else None
                ),
                "top": max(rows, key=lambda r: float(r["amount"]))["region"],
                "bottom": min(rows, key=lambda r: float(r["amount"]))["region"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
