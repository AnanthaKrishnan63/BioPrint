#!/usr/bin/env bash
set -euo pipefail
report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$report_dir/../../.." && pwd)"
bash "$repo_root/scripts/research.sh" "$report_dir/analyze_comparisons.py" > "$report_dir/analysis-build.log"
bash "$repo_root/scripts/research.sh" "$report_dir/prepare.py"
cd "$report_dir"
pdflatex -interaction=nonstopmode -halt-on-error bioprint-report.tex > build.log
pdflatex -interaction=nonstopmode -halt-on-error bioprint-report.tex >> build.log
pdfinfo bioprint-report.pdf | head -18
