#!/usr/bin/env bash
# Stop benchmark runners and child inference processes started by this project.

set -u

LOCKFILE="/tmp/benchmark_exhaustive.lock"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ $# -gt 0 ]]; then
    echo "Usage: scripts/stop_benchmarks.sh [--dry-run]"
    exit 2
fi

PATTERN='run_exhaustive_benchmarks|bench_llamacpp|bench_mlx|llama-bench|llama-server|mlx_lm\.server'

find_matches() {
    local ps_out
    if ! ps_out="$(ps -axo pid=,command= 2>&1)"; then
        echo "process scan failed: $ps_out" >&2
        return 1
    fi
    printf '%s\n' "$ps_out" | awk -v self="$$" -v parent="$PPID" -v pat="$PATTERN" '
        {
            pid = $1
            cmd = $0
            sub(/^[[:space:]]*[0-9]+[[:space:]]+/, "", cmd)
            if (pid == self || pid == parent) next
            if (cmd ~ pat && cmd !~ /awk -v self=/) print pid " " cmd
        }
    '
}

MATCHES=()
if ! MATCH_OUTPUT="$(find_matches)"; then
    echo "ERROR: Could not scan active processes. Run this command outside the sandbox."
    exit 1
fi
while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    MATCHES+=("$line")
done <<< "$MATCH_OUTPUT"

if [[ "${#MATCHES[@]}" -eq 0 ]]; then
    echo "No benchmark/model-server processes found."
else
    echo "Matched processes:"
    printf '  %s\n' "${MATCHES[@]}"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run: no processes killed, lock left untouched."
    exit 0
fi

if [[ "${#MATCHES[@]}" -gt 0 ]]; then
    for line in "${MATCHES[@]}"; do
        pid="${line%% *}"
        if [[ "$pid" =~ ^[0-9]+$ ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    SURVIVORS=()
    if ! SURVIVOR_OUTPUT="$(find_matches)"; then
        echo "WARNING: Could not rescan survivors after kill."
        SURVIVOR_OUTPUT=""
    fi
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        SURVIVORS+=("$line")
    done <<< "$SURVIVOR_OUTPUT"
    for line in "${SURVIVORS[@]}"; do
        pid="${line%% *}"
        if [[ "$pid" =~ ^[0-9]+$ ]]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
fi

rm -f "$LOCKFILE"
echo "Stopped benchmark processes and removed $LOCKFILE."
