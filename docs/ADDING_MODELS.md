# Adding a Model

Two config files, and most of the time you only touch one.

| File | Committed? | Contains |
|---|---|---|
| `configs/model_catalog.json` | yes | Public matching rules: family names, context caps, sampling defaults, filename patterns |
| `configs/models.local.json` | **no** (git-ignored) | Generated. Real filesystem paths for your machine |

The split exists so the repo can describe *what a Qwen3.6-35B is* without ever
committing where yours happens to live.

## The fast path

```bash
# 1. Download weights under a root you already scan
hf download poolside/Laguna-XS-2.1-GGUF \
  --local-dir ~/.lmstudio/models/poolside/Laguna-XS-2.1-GGUF

# 2. Re-scan
python3 scripts/discover_models.py --roots ~/.lmstudio/models --write configs/models.local.json

# 3. Confirm
python3 scripts/model_registry.py list
```

That's it. Unmatched models are registered as `custom` families using catalog
defaults, so a brand-new model is benchmarkable immediately — it just won't have
a friendly alias or a correct context cap yet.

Scan more than one root:

```bash
python3 scripts/discover_models.py --roots ~/.lmstudio/models ~/models --write configs/models.local.json

# or set it once
export BENCH_MODEL_ROOTS="$HOME/.lmstudio/models:$HOME/models"
```

## Registering it properly

To get a real alias, the right context cap, and correct sampling, add a family to
`configs/model_catalog.json`:

```json
{
  "id": "laguna_xs_33b_moe",
  "name": "Poolside Laguna XS 2.1 (33B-A3B MoE)",
  "family": "laguna",
  "architecture": "moe",
  "ctx_cap": 262144,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "gguf_patterns": ["Laguna-XS-2\\.1.*\\.gguf$"],
  "mlx_patterns": ["Laguna-XS-2\\.1.*(?:MLX|mlx|4bit|5bit|6bit|8bit)"]
}
```

### Fields

| Field | Why it matters |
|---|---|
| `id` | Stable identifier. Results group by it, so changing it splits your history |
| `family` | Drives per-family behaviour — e.g. `gemma4` gets no system prompt, because its template does not take one |
| `architecture` | `moe` or `dense`. Reported in results; MoE and dense are not comparable on tokens/sec alone |
| `ctx_cap` | The `max` pass fills this, and `niah` probes up to it. **This is a claim the `niah` eval will test** |
| `temperature` / `top_p` / `top_k` | Model-card values. Wrong sampling makes a good model look broken |
| `gguf_patterns` / `mlx_patterns` | Python regex, matched case-insensitively against both the full path and the bare filename |

Add short aliases in `LEGACY_ALIASES` in `scripts/model_registry.py`:

```python
"laguna": "laguna_xs_33b_moe",
"laguna-xs": "laguna_xs_33b_moe",
```

Then re-run `discover_models.py` and check your work:

```bash
python3 scripts/model_registry.py resolve laguna --backend llamacpp --format json
bash scripts/serve_local.sh laguna --backend llamacpp --dry-run
```

`--dry-run` prints the exact command without launching, which is the quickest way
to catch a wrong path or a bad engine choice.

## Engine routing

`serve_local.sh` picks the engine from the file type and the registry:

| Weights | Engine | Port |
|---|---|---|
| `.gguf` | `llama-server` | 8080 |
| MLX directory | `mlx_lm.server` | 8085 |
| MLX directory pinned by the registry | `mlx_vlm.server` | 8085 |

Some architectures cannot load under `mlx_lm` at all — Gemma 4 E4B's elastic
weights and the `gemma4_unified` 12B are the current examples. Pin those with
`mlx_server` on the entry in `models.local.json`:

```json
{ "quant": "4bit", "repo": "/path/to/model", "mlx_server": "mlx_vlm" }
```

`serve_local.sh` also sniffs `config.json` for `gemma4_unified` and routes
automatically, so the pin is a belt-and-braces measure.

### Client-side model IDs differ by server

This trips everyone up once:

```bash
# mlx_lm.server — the id is literally "default_model"
curl http://127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"default_model","messages":[{"role":"user","content":"hi"}]}'

# mlx_vlm.server — the id must be the full filesystem path
curl http://127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/full/path/to/model-dir","messages":[{"role":"user","content":"hi"}]}'
```

The full path satisfies both, which is why the web GUI and the eval runner always
send it.

## Chat templates

The registry deliberately does **not** force `--chat-template`. GGUF files carry
their own, and overriding it corrupts the prompt in ways that look like the model
being broken rather than the harness being wrong.

Forcing `gemma2` on a Gemma 4 QAT GGUF produced `"9b 9b 9b…"` where the embedded
template produced `"Two plus two equals four."` Same class of failure as forcing
`chatml` on Qwen, which silently breaks tool calling. `_validate_chat_template()`
hard-refuses the known-bad combinations.

## Known architecture gotchas

| Model | Issue |
|---|---|
| Cohere North Mini Code 1.0 | GGUF uses `cohere2moe`; stock llama.cpp cannot load it until PR #24260 lands. Build from that branch |
| Gemma 4 (all sizes) | Google republished under the **same names** on 2026-07-15/16 with chat-template and tool-calling fixes. Weights pulled before that date are materially different — re-download |
| Gemma 4 E4B | Elastic weights; `mlx_lm` cannot load it. Needs `mlx_vlm` |
| Gemma 4 12B | `gemma4_unified`, encoder-free. Needs `mlx_vlm >= 0.6.1` |

## Operating conventions

1. **One model root.** Everything under `~/.lmstudio/models` as `<org>/<repo>`.
   The HuggingFace hub cache is intentionally left empty.
2. **Serving never downloads.** MLX servers launch with `HF_HUB_OFFLINE=1`. A
   benchmark that pauses to fetch weights has already invalidated its own timing.
3. **Never commit `models.local.json`.** It is git-ignored, and `smoke_test.py`
   fails the build if a tracked file contains a local path.
