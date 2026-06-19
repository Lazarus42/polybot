#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
REPORT_DIR="${REPORT_DIR:-reports/underdog_optimization_kalshi}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/underdog_tests}"
COMMON=(--report-dir "$REPORT_DIR" --output-dir "$OUTPUT_DIR" --capital 5000)

"$PYTHON" scripts/underdog_test_suite.py weeks "${COMMON[@]}" --samples 10000
"$PYTHON" scripts/underdog_test_suite.py periods "${COMMON[@]}"
"$PYTHON" scripts/underdog_test_suite.py tail "${COMMON[@]}"
"$PYTHON" scripts/underdog_test_suite.py bootstrap "${COMMON[@]}" --cluster day --samples 10000
"$PYTHON" scripts/underdog_test_suite.py bootstrap "${COMMON[@]}" --cluster week --samples 10000
"$PYTHON" scripts/underdog_test_suite.py stability "${COMMON[@]}"
"$PYTHON" scripts/underdog_test_suite.py stress "${COMMON[@]}"
"$PYTHON" scripts/underdog_test_suite.py liquidity "${COMMON[@]}"
"$PYTHON" scripts/underdog_test_suite.py bankroll "${COMMON[@]}" --participation 0.10
"$PYTHON" scripts/underdog_test_suite.py baselines "${COMMON[@]}"
"$PYTHON" scripts/underdog_test_suite.py walk-forward "${COMMON[@]}"
