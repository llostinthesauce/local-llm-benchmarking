#!/usr/bin/env bash
set -e

# ==========================================
# Local Model Server Launcher
# Supports: llama.cpp (GGUF), MLX
# ==========================================

cd "$(dirname "$0")/.."

if [[ -x ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
else
    PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: python3 not found"
    exit 1
fi

PORT=""
HOST=""
BACKEND="llamacpp"
DRY_RUN=0
MODEL_ARG="$1"

usage() {
    echo "Usage: $0 <model_alias_or_path> [port] [--backend <llamacpp|mlx|mlx-vlm>] [--host <host>] [--dry-run]"
    echo ""
    echo "Backends:"
    echo " llamacpp — llama-server (GGUF files, OpenAI-compatible API)"
    echo " mlx — mlx_lm.server (MLX text directories, OpenAI-compatible API)"
    echo " mlx-vlm — mlx_vlm.server (MLX vision directories, OpenAI-compatible API)"
    echo ""
    echo "Auto-detection rules (when --backend not specified):"
    echo " *.gguf files → llamacpp"
    echo " MLX directories with config.json → mlx (text). Most local omni models"
    echo "   (Qwen3.5/3.6, Gemma 4) carry a vision_config but are served as text;"
    echo "   pass --backend mlx-vlm explicitly to load the vision tower."
    echo " Otherwise → llamacpp (default)"
    echo ""
    echo "Aliases come from configs/models.local.json."
    echo "Create it with:"
    echo "  python3 scripts/discover_models.py --roots ~/.lmstudio/models --write configs/models.local.json"
    exit 1
}

if [[ -z "$MODEL_ARG" ]]; then
    usage
fi

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        ''|*[!0-9]*)
            echo "ERROR: Unknown argument: $1"
            usage
            ;;
        *)
            PORT="$1"
            shift
            ;;
    esac
done

# Auto-detect backend if not specified
if [[ "$MODEL_ARG" == *.gguf ]]; then
    BACKEND="llamacpp"
elif [[ -d "$MODEL_ARG" && -f "$MODEL_ARG/config.json" ]]; then
    BACKEND="mlx"
fi

if [[ -z "$HOST" ]]; then
    HOST="127.0.0.1"
fi

if [[ -z "$PORT" ]]; then
    if [[ "$BACKEND" == "mlx" || "$BACKEND" == "mlx-vlm" ]]; then
        PORT="8085"
    else
        PORT="8080"
    fi
fi

# The registry catalogs vision models under backend "mlx"; ask it that way.
RESOLVE_BACKEND="$BACKEND"
[[ "$RESOLVE_BACKEND" == "mlx-vlm" ]] && RESOLVE_BACKEND="mlx"

if [[ -f "scripts/model_registry.py" ]]; then
    if RESOLVED_ENV=$("$PYTHON_BIN" scripts/model_registry.py resolve "$MODEL_ARG" --backend "$RESOLVE_BACKEND" --format shell 2>/tmp/serve_local_resolve.err); then
        eval "$RESOLVED_ENV"
        MODEL_ARG="$MODEL_PATH"
    elif [[ "$MODEL_ARG" != *.gguf && ! -e "$(eval echo "$MODEL_ARG")" ]]; then
        cat /tmp/serve_local_resolve.err
        exit 1
    fi
fi

echo "=========================================="
echo " Local Inference Server"
echo " Backend: $BACKEND"
echo " Host: $HOST"
echo " Port: $PORT"
echo " Model: $MODEL_ARG"
echo "=========================================="

case "$BACKEND" in
    llamacpp)
        MODEL_PATH=$(eval echo "$MODEL_ARG")
        if [[ ! -e "$MODEL_PATH" ]]; then
            echo "ERROR: Model path does not exist: $MODEL_PATH"
            exit 1
        fi

        CTX_SIZE=""
        [[ -n "${MODEL_CTX_CAP:-}" ]] && CTX_SIZE="$MODEL_CTX_CAP"
        [[ -z "$CTX_SIZE" ]] && CTX_SIZE="262144"

        echo " Context: $CTX_SIZE"
        echo ""

        LLAMA_SERVER_BIN="${MODEL_SERVER_BINARY:-}"
        if [[ -z "$LLAMA_SERVER_BIN" ]]; then
            LLAMA_SERVER_BIN="$(command -v llama-server || true)"
        fi

        if [[ -z "$LLAMA_SERVER_BIN" || ! -x "$LLAMA_SERVER_BIN" ]]; then
            echo "ERROR: llama-server not found. Install via: brew install llama.cpp"
            exit 1
        fi

        EXTRA_ARGS=()
        if echo "$MODEL_PATH" | grep -qi "gemma.*4\|gemma-4"; then
            EXTRA_ARGS=("--chat-template" "gemma2")
        fi
        if [[ -n "${MODEL_CHAT_TEMPLATE:-}" ]]; then
            TEMPLATE_LC="$(printf '%s' "$MODEL_CHAT_TEMPLATE" | tr '[:upper:]' '[:lower:]')"
            if [[ "$TEMPLATE_LC" == "chatml" ]]; then
                echo "ERROR: Refusing to force --chat-template chatml. Qwen/Granite GGUF models carry embedded templates." >&2
                exit 1
            fi
            EXTRA_ARGS=("--chat-template" "$MODEL_CHAT_TEMPLATE")
        fi

        MTP_ARGS=()
        MODEL_MTP_LC="$(printf '%s' "${MODEL_MTP_SUPPORTED:-}" | tr '[:upper:]' '[:lower:]')"
        if [[ "$MODEL_MTP_LC" == "true" ]]; then
            if ! "$LLAMA_SERVER_BIN" --help 2>&1 | grep -Eq -- '--spec-type .*\bmtp\b'; then
                echo "ERROR: $MODEL_NAME requires a llama-server build with --spec-type mtp support." >&2
                echo "Configured binary does not expose MTP: $LLAMA_SERVER_BIN" >&2
                exit 1
            fi
            MTP_ARGS=(-np 1 --spec-type "${MODEL_SPEC_TYPE:-mtp}" --spec-draft-n-max "${MODEL_SPEC_DRAFT_N_MAX:-3}")
        fi

        MMPROJ_ARGS=()
        MMPROJ_PATH="${MODEL_MMPROJ_PATH:-}"
        if [[ -z "$MMPROJ_PATH" ]]; then
            MODEL_DIR="$(dirname "$MODEL_PATH")"
            shopt -s nullglob
            MMPROJ_CANDIDATES=("$MODEL_DIR"/*mmproj*.gguf "$MODEL_DIR"/mmproj*.gguf)
            shopt -u nullglob
            if [[ ${#MMPROJ_CANDIDATES[@]} -gt 0 ]]; then
                MMPROJ_PATH="${MMPROJ_CANDIDATES[0]}"
            fi
        fi
        if [[ -n "$MMPROJ_PATH" ]]; then
            if [[ ! -e "$MMPROJ_PATH" ]]; then
                echo "ERROR: mmproj path does not exist: $MMPROJ_PATH"
                exit 1
            fi
            echo " Multimodal projector: $MMPROJ_PATH"
            MMPROJ_ARGS=(--mmproj "$MMPROJ_PATH")
        fi

        CMD=("$LLAMA_SERVER_BIN" -m "$MODEL_PATH" -c "$CTX_SIZE" --port "$PORT" --host "$HOST" -ngl all -ctk q8_0 -ctv q8_0 -fa on --mlock --no-mmap -t 8 -tb 16 "${EXTRA_ARGS[@]}" "${MTP_ARGS[@]}" "${MMPROJ_ARGS[@]}")
        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'DRY RUN:'
            printf ' %q' "${CMD[@]}"
            printf '\n'
            exit 0
        fi

        # Replace the wrapper shell so callers can signal the actual server.
        exec "${CMD[@]}"
        ;;

    mlx)
        MODEL_PATH=$(eval echo "$MODEL_ARG")
        if [[ "$MODEL_PATH" == */* && ! -d "$MODEL_PATH" ]]; then
            :
        elif [[ ! -d "$MODEL_PATH" ]]; then
            echo "ERROR: MLX model directory does not exist: $MODEL_PATH"
            exit 1
        fi

        echo ""
        if command -v mlx_lm.server >/dev/null 2>&1; then
            CMD=(mlx_lm.server --model "$MODEL_PATH" --host "$HOST" --port "$PORT")
        else
            CMD=("$PYTHON_BIN" -m mlx_lm server --model "$MODEL_PATH" --host "$HOST" --port "$PORT")
        fi
        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'DRY RUN:'
            printf ' %q' "${CMD[@]}"
            printf '\n'
            exit 0
        fi
        exec "${CMD[@]}"
        ;;

    mlx-vlm)
        MODEL_PATH=$(eval echo "$MODEL_ARG")
        if [[ "$MODEL_PATH" == */* && ! -d "$MODEL_PATH" ]]; then
            :
        elif [[ ! -d "$MODEL_PATH" ]]; then
            echo "ERROR: MLX-VLM model directory does not exist: $MODEL_PATH"
            exit 1
        fi

        echo ""
        if command -v mlx_vlm.server >/dev/null 2>&1; then
            CMD=(mlx_vlm.server --model "$MODEL_PATH" --host "$HOST" --port "$PORT")
        else
            CMD=("$PYTHON_BIN" -m mlx_vlm server --model "$MODEL_PATH" --host "$HOST" --port "$PORT")
        fi
        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'DRY RUN:'
            printf ' %q' "${CMD[@]}"
            printf '\n'
            exit 0
        fi
        exec "${CMD[@]}"
        ;;

    *)
        echo "ERROR: Unknown backend '$BACKEND'. Use: llamacpp, mlx, or mlx-vlm"
        exit 1
        ;;
esac
