---
name: report-writer
description: Turn a CSV of raw numbers into a written report — totals, period-over-period movement, and a short narrative a manager can read. Use this whenever someone asks for a report, summary, write-up, recap or 复盘 built from a data file (a CSV, an export, a spreadsheet dump), even when they never say the word "report" and even when they only describe the audience ("something I can send my boss").
---

<!-- compass eval fixture: v2 -->

# Report Writer

## Steps

1. **Summarise with the bundled script, not by hand.**

   ```bash
   python .claude/skills/report-writer/scripts/summarize.py <path-to-csv>
   ```

   It prints per-period totals, the quarter-over-quarter delta and the top and
   bottom rows as JSON. Doing this by hand costs a dozen tool calls and gets the
   arithmetic wrong on the rounding, which is the failure this script exists to
   remove.

2. **Write `report.md`** in the project root, in this order:

   ```markdown
   # <period> report
   ## Headline          <- one sentence: the number that moved and by how much
   ## Numbers           <- a table, straight from the script's output
   ## What changed      <- two or three sentences on why
   ```

3. **Say what you could not tell.** If the data cannot support a claim the
   reader will want (a cause, a forecast), write that down instead of guessing.
   A report that flags its own gap is more useful than one that fills it in.

## When this is the wrong tool

If the request is to *edit* an existing report, convert a file between formats,
or draw a chart, this skill has nothing to add — do it directly.
