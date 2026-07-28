#!/usr/bin/env bash
set -e

# ==========================================
# Local Model Server Launcher
#
# Registry-driven (configs/models.local.json via scripts/model_registry.py).
# Paired with scripts/llama_serve_menu.py (the interactive "python llama TUI").
#
# Backends:
#   llamacpp  llama-server, GGUF. Auto-enables MTP speculative decoding when the
#             resolved model has mtp_supported=true (see MTP block below).
#   mlx       mlx_lm.server     — MLX text, no KV quant.
#   mlx-kv    mlx_vlm.server    — MLX + KV cache q8 turboquant.
#   mlx-vlm   mlx_vlm.server    — MLX vision tower.
#
# MTP (Multi-Token Prediction) has two on-disk shapes, both auto-detected:
#   self-speculative / embedded head  (Qwen3.6-35B-A3B-MTP): bare --spec-type draft-mtp
#   separate draft head               (Gemma 4 26B mtp-*.gguf): adds --spec-draft-model
# llama.cpp cannot combine MTP with --mmproj or -np > 1, so MTP mode forces
# -np 1 and suppresses mmproj.
# Gemma 4 MTP needs llama.cpp >= b9610 (Gemma4 MTP landed in PR #23398, 2026-06-07);
# older builds fail with "unknown model architecture: 'gemma4-assistant'".
# KV cache: default -ctk/-ctv q8_0 is fine WITH MTP (verified: Qwen ~0.80,
# Gemma ~0.58 draft acceptance). This is unrelated to the MLX mlx-kv backend.
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
FORCE_NO_MTP=0
MODEL_ARG="$1"

usage() {
    echo "Usage: $0 <model_alias_or_path> [port] [--backend <llamacpp|mlx|mlx-kv|mlx-vlm>] [--host <host>] [--dry-run] [--no-mtp]"
    echo ""
    echo "Backends:"
    echo " llamacpp — llama-server (GGUF files, OpenAI-compatible API)"
    echo " mlx     — mlx_lm.server (MLX text directories, OpenAI-compatible API, no KV quant)"
    echo " mlx-kv  — mlx_vlm.server + --kv-bits 8 --kv-quant-scheme turboquant (KV cache q8)"
    echo " mlx-vlm — mlx_vlm.server (MLX vision directories, OpenAI-compatible API)"
    echo ""
    echo "Auto-detection rules (when --backend not specified):"
    echo " *.gguf files → llamacpp"
    echo " MLX directories with config.json → mlx (text). Most local omni models"
    echo "   (Qwen3.5/3.6, Gemma 4) carry a vision_config but are served as text;"
    echo "   pass --backend mlx-kv for KV cache quantization, --backend mlx-vlm for vision."
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
        --no-mtp)
            FORCE_NO_MTP=1
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
    if [[ "$BACKEND" == "mlx" || "$BACKEND" == "mlx-vlm" || "$BACKEND" == "mlx-kv" ]]; then
        PORT="8085"
    else
        PORT="8080"
    fi
fi

# The registry catalogs all MLX models under backend "mlx"; ask it that way.
RESOLVE_BACKEND="$BACKEND"
[[ "$RESOLVE_BACKEND" == "mlx-vlm" || "$RESOLVE_BACKEND" == "mlx-kv" ]] && RESOLVE_BACKEND="mlx"

if [[ -f "scripts/model_registry.py" ]]; then
    if RESOLVED_ENV=$("$PYTHON_BIN" scripts/model_registry.py resolve "$MODEL_ARG" --backend "$RESOLVE_BACKEND" --format shell 2>/tmp/serve_local_resolve.err); then
        eval "$RESOLVED_ENV"
        MODEL_ARG="$MODEL_PATH"
    elif [[ "$MODEL_ARG" != *.gguf && ! -e "$(eval echo "$MODEL_ARG")" ]]; then
        cat /tmp/serve_local_resolve.err
        exit 1
    fi
fi

# Registry-provided sampling defaults (per-family, e.g. Gemma 4's
# temp=1.0/top_p=0.95/top_k=64 model-card config). Only set if resolution
# populated them; per-request overrides from clients still take precedence.
SAMPLING_ARGS=()
if [[ -n "${MODEL_TEMPERATURE:-}" ]]; then
    SAMPLING_ARGS=(--temp "$MODEL_TEMPERATURE" --top-p "${MODEL_TOP_P:-1.0}" --top-k "${MODEL_TOP_K:-0}")
    echo " Sampling: temp=$MODEL_TEMPERATURE top_p=${MODEL_TOP_P:-1.0} top_k=${MODEL_TOP_K:-0}"
fi

echo "=========================================="
echo " Local Inference Server"
echo " Backend: $BACKEND"
echo " Host: $HOST"
echo " Port: $PORT"
echo " Model: $MODEL_ARG"
[[ -n "${MODEL_USE_CASE:-}" ]] && echo " Use case: $MODEL_USE_CASE"
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

        # Do NOT blanket-force a chat template. GGUFs that carry their own
        # embedded template (the gemma-4-26B QAT GGUF uses a <|turn>/<|channel>
        # thinking format) are corrupted by an override — forcing gemma2 made the
        # model emit garbage ("9b 9b 9b…"). Only honour an explicit registry
        # template (MODEL_CHAT_TEMPLATE) below; otherwise let llama.cpp use the
        # GGUF's embedded template.
        EXTRA_ARGS=()
        if [[ -n "${MODEL_CHAT_TEMPLATE:-}" ]]; then
            TEMPLATE_LC="$(printf '%s' "$MODEL_CHAT_TEMPLATE" | tr '[:upper:]' '[:lower:]')"
            if [[ "$TEMPLATE_LC" == "chatml" ]]; then
                echo "ERROR: Refusing to force --chat-template chatml. Qwen/Granite GGUF models carry embedded templates." >&2
                exit 1
            fi
            EXTRA_ARGS=("--chat-template" "$MODEL_CHAT_TEMPLATE")
        fi

        # --- MTP (Multi-Token Prediction) speculative decoding ---
        # Gated on registry mtp_supported=true. Handles both on-disk shapes:
        #   * separate draft head  (Gemma: mtp-*.gguf beside the GGUF) → --spec-draft-model
        #   * self-speculative      (Qwen3.6 MTP: head embedded in the GGUF) → bare --spec-type
        # MTP cannot coexist with --mmproj or -np > 1 in llama.cpp, so when active we
        # pin -np 1 here and suppress mmproj in the block below (MTP_ACTIVE flag).
        MTP_ARGS=()
        MTP_ACTIVE=0
        MODEL_MTP_LC="$(printf '%s' "${MODEL_MTP_SUPPORTED:-}" | tr '[:upper:]' '[:lower:]')"
        if [[ "$MODEL_MTP_LC" == "true" && "$FORCE_NO_MTP" == "1" ]]; then
            echo " MTP: forced OFF (--no-mtp) — serving as plain GGUF baseline."
        fi
        if [[ "$MODEL_MTP_LC" == "true" && "$FORCE_NO_MTP" != "1" ]]; then
            if ! "$LLAMA_SERVER_BIN" --help 2>&1 | grep -Eq -- '--spec-type.*draft-mtp|--spec-type.*\bmtp\b'; then
                echo "ERROR: $MODEL_NAME requires a llama-server build with --spec-type draft-mtp support." >&2
                echo "Configured binary does not expose MTP: $LLAMA_SERVER_BIN" >&2
                echo "Fix: brew upgrade llama.cpp  (Gemma 4 MTP needs >= b9610 / PR #23398)." >&2
                exit 1
            fi
            # NOTE: the --help probe passes on builds that expose the flag but cannot
            # load the gemma4-assistant draft arch (e.g. b9410), which then crash at
            # model load with "unknown model architecture". If that happens, upgrade:
            #   brew upgrade llama.cpp
            MTP_ACTIVE=1
            # Optional separate draft head: explicit registry value, else auto-detect
            # mtp-*.gguf beside the main GGUF. Absent for self-speculative models.
            DRAFT_MODEL_PATH="${MODEL_DRAFT_MODEL:-}"
            if [[ -z "$DRAFT_MODEL_PATH" ]]; then
                _MODEL_DIR="$(dirname "$MODEL_PATH")"
                shopt -s nullglob
                _DRAFT_CANDIDATES=("$_MODEL_DIR"/mtp-*.gguf "$_MODEL_DIR"/MTP/*.gguf)
                shopt -u nullglob
                [[ ${#_DRAFT_CANDIDATES[@]} -gt 0 ]] && DRAFT_MODEL_PATH="${_DRAFT_CANDIDATES[0]}"
            fi
            MTP_ARGS=(--spec-type "${MODEL_SPEC_TYPE:-draft-mtp}" --spec-draft-n-max "${MODEL_SPEC_DRAFT_N_MAX:-2}" -np 1)
            if [[ -n "$DRAFT_MODEL_PATH" ]]; then
                echo " MTP: ON — separate draft head $(basename "$DRAFT_MODEL_PATH") (n-max=${MODEL_SPEC_DRAFT_N_MAX:-2})"
                MTP_ARGS+=(--spec-draft-model "$DRAFT_MODEL_PATH")
            else
                echo " MTP: ON — self-speculative / embedded head (n-max=${MODEL_SPEC_DRAFT_N_MAX:-2})"
            fi
        fi

        MMPROJ_ARGS=()
        if [[ "$MTP_ACTIVE" == "1" ]]; then
            # llama.cpp does not support --mmproj together with MTP.
            if [[ -n "${MODEL_MMPROJ_PATH:-}" ]]; then
                echo " Note: mmproj suppressed — llama.cpp cannot combine --mmproj with MTP."
            fi
        else
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
        fi

        CMD=("$LLAMA_SERVER_BIN" -m "$MODEL_PATH" -c "$CTX_SIZE" --port "$PORT" --host "$HOST" -ngl all -ctk q8_0 -ctv q8_0 -fa on --mlock --no-mmap -t 8 -tb 16 "${SAMPLING_ARGS[@]}" "${EXTRA_ARGS[@]}" "${MTP_ARGS[@]}" "${MMPROJ_ARGS[@]}")
        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'DRY RUN:'
            printf ' %q' "${CMD[@]}"
            printf '\n'
            exit 0
        fi

        # Replace the wrapper shell so callers can signal the actual server.
        exec "${CMD[@]}"
        ;;

    mlx|mlx-vlm|mlx-kv)
        MODEL_PATH=$(eval echo "$MODEL_ARG")
        if [[ ! -d "$MODEL_PATH" ]]; then
            echo "ERROR: MLX model directory not found on disk: $MODEL_PATH" >&2
            echo "Refusing to launch: a non-local --model would make the MLX server download from Hugging Face." >&2
            exit 1
        fi

        # Pick the runner. mlx_lm serves text MLX dirs; mlx_vlm serves vision/omni
        # towers. mlx_lm CANNOT load the gemma4_unified omni arch, so route those to
        # mlx_vlm even when the caller asked for plain --backend mlx. (Plain
        # "has vision_config" is intentionally NOT enough — most omni models serve
        # fine and faster as text under mlx_lm.)
        MLX_SERVER="mlx_lm"
        [[ "$BACKEND" == "mlx-vlm" || "$BACKEND" == "mlx-kv" ]] && MLX_SERVER="mlx_vlm"
        # Registry can pin a model to a specific MLX server (models.local.json
        # "mlx_server") for archs/weights mlx_lm cannot load: gemma4_unified (12B)
        # and the gemma4 E4B elastic checkpoint. This is the reliable signal.
        if [[ "${MODEL_MLX_SERVER:-}" == "mlx_vlm" && "$MLX_SERVER" == "mlx_lm" ]]; then
            echo " Note: registry pins this model to mlx_vlm.server (mlx_lm cannot load it)."
            MLX_SERVER="mlx_vlm"
        fi
        # Fallback heuristic if a gemma4_unified model was not pinned in the registry.
        if [[ "$MLX_SERVER" == "mlx_lm" && -f "$MODEL_PATH/config.json" ]] \
            && grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"gemma4_unified"' "$MODEL_PATH/config.json"; then
            echo " Note: model_type gemma4_unified is unsupported by mlx_lm — routing to mlx_vlm.server."
            MLX_SERVER="mlx_vlm"
        fi

        # Serve strictly from disk. HF_HUB_OFFLINE stops the server from silently
        # downloading a model when a client request names a repo-id that is not the
        # loaded local path (mlx_lm/mlx_vlm reload per-request "model").
        MLX_ENV=(env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)

        # mlx_vlm.server has no --temp/--top-p/--top-k flags (sampling is
        # per-request only); only mlx_lm.server accepts registry sampling defaults.
        MLX_SAMPLING_ARGS=()
        if [[ "$MLX_SERVER" == "mlx_lm" ]]; then
            MLX_SAMPLING_ARGS=("${SAMPLING_ARGS[@]}")
        elif [[ ${#SAMPLING_ARGS[@]} -gt 0 ]]; then
            echo " Note: mlx_vlm.server has no --temp/--top-p/--top-k flags; sampling defaults from the registry won't apply server-side."
        fi

        # KV cache quantization — only available via mlx_vlm.server (mlx_lm has no --kv-bits).
        KV_ARGS=()
        if [[ "$BACKEND" == "mlx-kv" ]]; then
            KV_ARGS=(--kv-bits 8 --kv-quant-scheme turboquant)
            echo " KV cache: q8 turboquant (mlx_vlm.server)"
        fi

        echo ""
        if command -v "${MLX_SERVER}.server" >/dev/null 2>&1; then
            CMD=("${MLX_ENV[@]}" "${MLX_SERVER}.server" --model "$MODEL_PATH" --host "$HOST" --port "$PORT" "${MLX_SAMPLING_ARGS[@]}" "${KV_ARGS[@]}")
        else
            CMD=("${MLX_ENV[@]}" "$PYTHON_BIN" -m "$MLX_SERVER" server --model "$MODEL_PATH" --host "$HOST" --port "$PORT" "${MLX_SAMPLING_ARGS[@]}" "${KV_ARGS[@]}")
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
        echo "ERROR: Unknown backend '$BACKEND'. Use: llamacpp, mlx, mlx-kv, or mlx-vlm"
        exit 1
        ;;
esac
