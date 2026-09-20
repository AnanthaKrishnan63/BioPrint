#!/usr/bin/env bash
set -euo pipefail
report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$report_dir/../../.." && pwd)"
bash "$repo_root/scripts/research.sh" "$report_dir/analyze_comparisons.py" > "$report_dir/analysis-build.log"
bash "$repo_root/scripts/research.sh" "$report_dir/prepare.py"
cd "$report_dir"
pdflatex -interaction=nonstopmode -halt-on-error bioprint-report.tex > build.log
pdflatex -interaction=nonstopmode -halt-on-error bioprint-report.tex >> build.log
page_count="$(pdfinfo bioprint-report.pdf | awk '/^Pages:/ {print $2}')"
if [[ "$page_count" != 6 ]]; then
  echo "Report must be exactly six pages; got $page_count. Final publication copy was not updated." >&2
  exit 1
fi
cp bioprint-report.pdf ../BioPrint_Report.pdf
pdfinfo bioprint-report.pdf | head -18
