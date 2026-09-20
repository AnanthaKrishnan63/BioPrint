#!/usr/bin/env bash
# The main-branch entry point always serves only localhost:8000.
set -euo pipefail
checkout_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${BIGIDEA_PYTHON:-python}" "$checkout_dir/launch.py" "$@"
