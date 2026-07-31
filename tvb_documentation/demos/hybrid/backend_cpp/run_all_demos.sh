#!/usr/bin/env bash
# Run every Python script and Jupyter notebook in this directory and report
# pass / fail for each.  Skips __pycache__ and the outputs directory.
#
# Usage:
#   ./run_all_demos.sh            # run all
#   ./run_all_demos.sh --timeout 120   # custom per-demo timeout (seconds)
#   SKIP="benchmark_*" ./run_all_demos.sh   # skip glob pattern
#
# Exit code: 0 if everything passed, 1 if any demo failed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMEOUT=${TIMEOUT:-180}    # seconds per demo
SKIP_PATTERN=${SKIP:-""}   # optional glob to skip, e.g. "benchmark_*"

PASS=()
FAIL=()
SKIP_LIST=()

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
RESET='\033[0m'

run_script() {
    local file="$1"
    local name
    name="$(basename "$file")"

    if [[ -n "$SKIP_PATTERN" ]] && [[ "$name" == $SKIP_PATTERN ]]; then
        SKIP_LIST+=("$name")
        echo -e "${YELLOW}SKIP${RESET}  $name"
        return
    fi

    printf "%-60s " "$name"
    local log
    log="$(mktemp /tmp/tvb_demo_XXXXXX.log)"

    if timeout "$TIMEOUT" python "$file" > "$log" 2>&1; then
        echo -e "${GREEN}PASS${RESET}"
        PASS+=("$name")
    else
        local exit_code=$?
        echo -e "${RED}FAIL${RESET}  (exit $exit_code)"
        echo "    >>> Last 10 lines of output:"
        tail -10 "$log" | sed 's/^/    /'
        FAIL+=("$name")
    fi
    rm -f "$log"
}

run_notebook() {
    local file="$1"
    local name
    name="$(basename "$file")"

    if [[ -n "$SKIP_PATTERN" ]] && [[ "$name" == $SKIP_PATTERN ]]; then
        SKIP_LIST+=("$name")
        echo -e "${YELLOW}SKIP${RESET}  $name"
        return
    fi

    printf "%-60s " "$name"
    local log
    log="$(mktemp /tmp/tvb_demo_XXXXXX.log)"
    local out_nb
    out_nb="$(mktemp /tmp/tvb_demo_XXXXXX.ipynb)"

    if timeout "$TIMEOUT" /home/ziaee/envs/hybrid/bin/jupyter nbconvert \
            --to notebook \
            --execute \
            --ExecutePreprocessor.timeout="$TIMEOUT" \
            --output "$out_nb" \
            "$file" > "$log" 2>&1; then
        echo -e "${GREEN}PASS${RESET}"
        PASS+=("$name")
    else
        local exit_code=$?
        echo -e "${RED}FAIL${RESET}  (exit $exit_code)"
        echo "    >>> Last 10 lines of output:"
        tail -10 "$log" | sed 's/^/    /'
        FAIL+=("$name")
    fi
    rm -f "$log" "$out_nb"
}

echo "========================================================"
echo " TVB C++ backend demos  (timeout=${TIMEOUT}s each)"
echo " Directory: $SCRIPT_DIR"
echo "========================================================"
echo ""

cd "$SCRIPT_DIR"

# Run Python scripts (sorted, skip this script itself and helper dirs)
while IFS= read -r -d '' f; do
    [[ "$(basename "$f")" == "run_all_demos.sh" ]] && continue
    run_script "$f"
done < <(find "$SCRIPT_DIR" -maxdepth 1 -name "*.py" -not -name "__*" -print0 | sort -z)

# Run notebooks
while IFS= read -r -d '' f; do
    run_notebook "$f"
done < <(find "$SCRIPT_DIR" -maxdepth 1 -name "*.ipynb" -print0 | sort -z)

echo ""
echo "========================================================"
printf " PASS: %d   FAIL: %d   SKIP: %d\n" \
    "${#PASS[@]}" "${#FAIL[@]}" "${#SKIP_LIST[@]}"
echo "========================================================"

if [[ ${#FAIL[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed demos:${RESET}"
    for f in "${FAIL[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

exit 0
