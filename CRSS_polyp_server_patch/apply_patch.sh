#!/usr/bin/env bash
set -euo pipefail
patch_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/polyp1450.patch"
project_root="${1:-.}"
cd "$project_root"
git rev-parse --show-toplevel >/dev/null
if git apply --reverse --check "$patch_file" 2>/dev/null; then
    echo 'Same patch is already applied.'
    exit 0
fi
git apply --check "$patch_file"
git apply "$patch_file"
echo 'Applied. Run: GPU=0 bash scripts/run_polyp1450.sh prepare'
