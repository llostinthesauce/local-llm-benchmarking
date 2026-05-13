#!/usr/bin/env bash
# Run every available model/backend variant across speed passes.
# This is the single orchestrator. No other benchmark coordination scripts exist.
#
# Passes default to all 4 (micro normal high max). Pass a subset with --passes.

set -u

cd "$(dirname "$0")/.." || exit 1

DRY_RUN=0
EXECUTE=0
ALLOW_DIRTY=0
ALLOW_ACTIVE=0
RUN_DIR=""
PASSES=(micro normal high max)

usage() {
    cat <<'EOF'
Usage:
  scripts/run_exhaustive_benchmarks.sh --dry-run [--passes micro,normal]
  scripts/run_exhaustive_benchmarks.sh --execute [--allow-dirty] [--passes micro,normal]

Real benchmark runs require --execute. A dirty git worktree is refused unless
--allow-dirty is passed. --passes accepts a comma-separated subset of:
micro, normal, high, max. Default: all four.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=1; shift ;;
        --execute)       EXECUTE=1; shift ;;
        --allow-dirty)   ALLOW_DIRTY=1; shift ;;
        --allow-active)  ALLOW_ACTIVE=1; shift ;;
        --passes)
            IFS=',' read -ra PASSES <<< "${2:-}"
            shift 2
            ;;
        --run-dir)
            RUN_DIR="${2:-}"
            [[ -z "$RUN_DIR" ]] && { echo "ERROR: --run-dir requires a value"; exit 2; }
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1"; usage; exit 2 ;;
    esac
done

if [[ "$DRY_RUN" == "1" && "$EXECUTE" == "1" ]]; then
    echo "ERROR: choose only one of --dry-run or --execute"
    exit 2
fi

if [[ "$DRY_RUN" != "1" && "$EXECUTE" != "1" ]]; then
    usage
    exit 2
fi

LOCKFILE="/tmp/benchmark_exhaustive.lock"
SERVER_PID=""

cleanup() {
    cleanup_server
    if [[ "$DRY_RUN" != "1" ]]; then
        rm -f "$LOCKFILE"
    fi
}

cleanup_and_exit() {
    local rc=$?
    cleanup
    exit "$rc"
}

abort_run() {
    local signal="$1"
    if [[ "$(type -t log 2>/dev/null || true)" == "function" ]]; then
        log "Interrupted by $signal; cleaning up"
    else
        echo "Interrupted by $signal; cleaning up" >&2
    fi
    cleanup
    exit 130
}

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

active_processes() {
    local ps_out
    local current_pgid
    if ! current_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"; then
        current_pgid=""
    fi
    if ! ps_out="$(ps -axo pid=,pgid=,command= 2>&1)"; then
        echo "process scan failed: $ps_out" >&2
        return 1
    fi
    printf '%s\n' "$ps_out" | awk -v self="$$" -v parent="$PPID" -v current_pgid="$current_pgid" '
        BEGIN { pat = "run_exhaustive_benchmarks|bench_llamacpp|bench_mlx|llama-bench|llama-server|mlx_lm\\.server" }
        {
            pid = $1
            pgid = $2
            cmd = $0
            sub(/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+/, "", cmd)
            if (pid == self || pid == parent) next
            if (current_pgid != "" && pgid == current_pgid) next
            if (cmd ~ pat && cmd !~ /awk -v self=/) print pid " " cmd
        }
    '
}

if [[ "$DRY_RUN" != "1" ]]; then
    if [[ "$ALLOW_ACTIVE" != "1" ]]; then
        if ! ACTIVE_PROCS="$(active_processes)"; then
            echo "ERROR: Could not scan active processes. Run outside the sandbox or pass --allow-active intentionally."
            exit 4
        fi
        if [[ -n "$ACTIVE_PROCS" ]]; then
            echo "ERROR: Existing benchmark/model-server process(es) detected:"
            echo "$ACTIVE_PROCS"
            echo "Run scripts/stop_benchmarks.sh first, or pass --allow-active if this is intentional."
            exit 4
        fi
    fi
    if [[ -f "$LOCKFILE" ]]; then
        LOCK_PID=$(cat "$LOCKFILE" 2>/dev/null)
        if kill -0 "$LOCK_PID" 2>/dev/null; then
            echo "ERROR: Another exhaustive benchmark is already running (PID $LOCK_PID)."
            echo "If this is stale, remove $LOCKFILE and retry."
            exit 3
        fi
        rm -f "$LOCKFILE"
    fi
    echo $$ > "$LOCKFILE"
fi

trap cleanup_and_exit EXIT
trap 'abort_run INT' INT
trap 'abort_run TERM' TERM

if [[ -x ".venv/bin/python3" ]]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: python3 not found"
    exit 1
fi

export PYTHONUNBUFFERED=1
TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="results/exhaustive_${TS}"
fi
LOG="$RUN_DIR/master.log"
COOLDOWN="${BENCH_COOLDOWN:-30}"
MEM_GUARD="${BENCH_MEM_GUARD:-95}"

mkdir -p "$RUN_DIR"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"
}

run_step() {
    local label="$1"
    shift
    log "START $label"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[%s] DRY RUN %s:' "$(date '+%H:%M:%S')" "$label" | tee -a "$LOG"
        printf ' %q' "$@" | tee -a "$LOG"
        printf '\n' | tee -a "$LOG"
        log "END   $label rc=0"
        return 0
    fi
    "$@" >>"$LOG" 2>&1
    local rc=$?
    log "END   $label rc=$rc"
    return "$rc"
}

wait_for_health() {
    local url="$1"
    local label="$2"
    local i
    for i in $(seq 1 180); do
        if curl -sS --max-time 2 "$url" >/dev/null 2>&1; then
            log "$label ready after ${i}s"
            return 0
        fi
        if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "$label died before becoming ready"
            return 1
        fi
        sleep 1
    done
    log "$label did not become ready"
    return 1
}

write_registry_tsv() {
    "$PYTHON" - <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
import model_registry

for row in model_registry.iter_models(model_registry.DEFAULT_CONFIG):
    if row["backend"] == "llamacpp" and not row["exists"]:
        continue
    fields = [
        row["backend"],
        row["family_id"],
        row["path"],
        str(row["ctx_cap"]),
        row["quant"] or "-",
        row["name"],
        str(row.get("mtp_supported", False)),
        row.get("draft_model") or "-",
        row.get("architecture", "dense"),
    ]
    print("\t".join(fields))
PY
}

REGISTRY_TSV="$RUN_DIR/registry.tsv"
write_registry_tsv > "$REGISTRY_TSV"

log "=== Exhaustive benchmark started ==="
log "Run dir: $RUN_DIR"
log "Python: $PYTHON"
log "Passes: ${PASSES[*]}"
log "Cooldown: ${COOLDOWN}s"
log "Mem guard: ${MEM_GUARD}%"
log "Registry snapshot:"
cat "$REGISTRY_TSV" | tee -a "$LOG"
if [[ "$DRY_RUN" == "1" ]]; then
    log "Mode: dry-run, no benchmark commands will execute"
fi

log ""
log "=== llama.cpp raw: all existing GGUF rows ==="
run_step "llamacpp_raw_all" \
    "$PYTHON" scripts/bench_llamacpp_raw.py \
        --config configs/models.local.json \
        --all \
        --passes "${PASSES[@]}" \
        --output-dir "$RUN_DIR/llamacpp_raw" \
        --cooldown "$COOLDOWN" \
        --mem-guard "$MEM_GUARD"

log ""
log "=== llama.cpp API: each existing GGUF row via llama-server ==="
while IFS=$'\t' read -r backend family_id path ctx_cap quant name mtp_supported draft_model architecture; do
    [[ "$backend" == "llamacpp" ]] || continue
    MTP_ARGS=()
    if [[ "$mtp_supported" == "True" || "$mtp_supported" == "true" ]]; then
        MTP_ARGS=(--mtp)
    fi

    log "Starting llama-server for $family_id $quant"
    SERVER_LOG="$RUN_DIR/server_${family_id}_${quant}.log"
    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY RUN start server: bash scripts/serve_local.sh $path --backend llamacpp --host 127.0.0.1"
        run_step "llamacpp_api_${family_id}_${quant}" \
            "$PYTHON" scripts/bench_llamacpp_api.py \
                --model "$name" \
                --passes "${PASSES[@]}" \
                --ctx-cap "$ctx_cap" \
                --quant "$quant" \
                --output-dir "$RUN_DIR/llamacpp_api" \
                --cooldown "$COOLDOWN" \
                --mem-guard "$MEM_GUARD" \
                ${MTP_ARGS:+"${MTP_ARGS[@]}"}
        continue
    fi

    bash scripts/serve_local.sh "$path" --backend llamacpp --host 127.0.0.1 >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    if wait_for_health "http://127.0.0.1:8080/v1/models" "llama-server $family_id"; then
        run_step "llamacpp_api_${family_id}_${quant}" \
            "$PYTHON" scripts/bench_llamacpp_api.py \
                --model "$name" \
                --passes "${PASSES[@]}" \
                --ctx-cap "$ctx_cap" \
                --quant "$quant" \
                --output-dir "$RUN_DIR/llamacpp_api" \
                --cooldown "$COOLDOWN" \
                --mem-guard "$MEM_GUARD" \
                ${MTP_ARGS:+"${MTP_ARGS[@]}"}
    else
        log "SKIP llama.cpp API for $family_id $quant; server did not become healthy"
    fi

    cleanup_server
    sleep 5
done < "$REGISTRY_TSV"

log ""
log "=== MLX Direct: all MLX rows ==="
while IFS=$'\t' read -r backend family_id path ctx_cap quant name mtp_supported draft_model architecture; do
    [[ "$backend" == "mlx" ]] || continue
    DRAFT_ARGS=()
    if [[ -n "${draft_model:-}" && "${draft_model:-}" != "-" ]]; then
        DRAFT_ARGS=(--draft-model "$draft_model")
    fi
    SIZE_GB=0
    if [[ -d "$path" ]]; then
        SIZE_GB=$(du -sg "$path" 2>/dev/null | cut -f1 || echo 0)
    fi
    run_step "mlx_direct_${family_id}_${quant}" \
        "$PYTHON" scripts/bench_mlx_direct.py \
            --model "$path" \
            --passes "${PASSES[@]}" \
            --ctx-cap "$ctx_cap" \
            --quant "$quant" \
            --architecture "${architecture:-dense}" \
            --model-size-gb "$SIZE_GB" \
            --output-dir "$RUN_DIR/mlx_direct" \
            --cooldown "$COOLDOWN" \
            --mem-guard "$MEM_GUARD" \
            ${DRAFT_ARGS:+"${DRAFT_ARGS[@]}"}
done < "$REGISTRY_TSV"

log ""
log "=== MLX API: all MLX rows ==="
while IFS=$'\t' read -r backend family_id path ctx_cap quant name mtp_supported draft_model architecture; do
    [[ "$backend" == "mlx" ]] || continue
    DRAFT_ARGS=()
    if [[ -n "${draft_model:-}" && "${draft_model:-}" != "-" ]]; then
        DRAFT_ARGS=(--draft-model "$draft_model")
    fi
    run_step "mlx_api_${family_id}_${quant}" \
        "$PYTHON" scripts/bench_mlx_api.py \
            --model "$path" \
            --passes "${PASSES[@]}" \
            --ctx-cap "$ctx_cap" \
            --quant "$quant" \
            --output-dir "$RUN_DIR/mlx_api" \
            --cooldown "$COOLDOWN" \
            --mem-guard "$MEM_GUARD" \
            ${DRAFT_ARGS:+"${DRAFT_ARGS[@]}"}
done < "$REGISTRY_TSV"

log ""
log "=== Exhaustive benchmark finished ==="
log "Results: $RUN_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY RUN skip aggregate report"
else
    run_step "aggregate_results" \
        "$PYTHON" scripts/aggregate_results.py "$RUN_DIR"
fi
