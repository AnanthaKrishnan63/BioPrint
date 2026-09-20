#!/usr/bin/env bash
set -euo pipefail
demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkout_dir="$(cd -- "$demo_dir/../.." && pwd)"
common_git="$(git -C "$checkout_dir" rev-parse --path-format=absolute --git-common-dir)"
research_root="${BIOPRINT_RESEARCH_ROOT:-$(dirname -- "$common_git")}"
export BIOPRINT_RESEARCH_ROOT="$research_root"
exec bash "$research_root/scripts/research.sh" "$demo_dir/demo.py" "$@"
