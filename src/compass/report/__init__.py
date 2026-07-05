"""Report generation and analysis module."""

from compass.report.analyzer import AnalysisReport, EvalResultAnalyzer, TaskEvalResult
from compass.report.compare import (
    CaseFlip,
    CaseRecord,
    ComparisonReport,
    PairedStats,
    compare_paths,
    compare_results,
    load_case_records,
    paired_stats,
)
from compass.report.console import ConsoleReporter
from compass.report.html import HTMLReporter

__all__ = [
    "HTMLReporter",
    "ConsoleReporter",
    "EvalResultAnalyzer",
    "TaskEvalResult",
    "AnalysisReport",
    # Paired comparison
    "CaseFlip",
    "CaseRecord",
    "ComparisonReport",
    "PairedStats",
    "compare_paths",
    "compare_results",
    "load_case_records",
    "paired_stats",
]
