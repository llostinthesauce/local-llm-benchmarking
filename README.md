# Local LLM Benchmarking

Simple benchmark harness for comparing local inference backends on Apple Silicon.

The active suite has five backends:

| Backend | Engine | CSV tag |
| --- | --- | --- |
| llama.cpp raw | `llama-bench` | `LLAMACPP_RAW` |
| llama.cpp server | `llama-server` OpenAI-compatible API | `LLAMACPP_API` |
| MLX direct | `mlx_lm.stream_generate` | `MLX_DIRECT` |
| MLX server | `mlx_lm.server` OpenAI-compatible API | `MLX_API` |
| MLX-VLM server | `mlx_vlm.server` OpenAI-compatible API | `MLX_VLM_API` |

The TUI is the main entry point. Standalone scripts are kept for automation and debugging.

## Requirements

- macOS on Apple Silicon
- Python 3.9+
- Python packages: `psutil`, `questionary`, `rich`, `requests`
- llama.cpp tools on `PATH`: `llama-bench`, `llama-server`
- Optional for MLX backends: `mlx`, `mlx-lm`

```bash
pip install -r requirements.txt
```

Install llama.cpp however you prefer. With Homebrew:

```bash
brew install llama.cpp
```

## Model Discovery

Committed files do not contain local model paths.

Create a local registry by pointing the scanner at the folders where your models live:

```bash
python3 scripts/discover_models.py --roots ~/.lmstudio/models ~/models --write configs/models.local.json
```

`configs/models.local.json` is ignored by git. It is the machine-specific registry used by the TUI, server launcher, and benchmark scripts.

Check what was found:

```bash
python3 scripts/model_registry.py list
python3 scripts/check_model_availability.py
```

The scanner recognizes known Qwen, Gemma, and Granite families from `configs/model_catalog.json`. Unmatched GGUF files and MLX directories are still added as custom models with conservative defaults.

## Run

Interactive TUI:

```bash
python3 bench_tui.py
```

Dry-run the full orchestrator:

```bash
bash scripts/run_exhaustive_benchmarks.sh --dry-run
```

Run a small benchmark:

```bash
bash scripts/run_exhaustive_benchmarks.sh --execute --passes micro --allow-dirty
```

Run the full configured suite:

```bash
bash scripts/run_exhaustive_benchmarks.sh --execute --allow-dirty
```

Aggregate an existing run:

```bash
python3 scripts/aggregate_results.py results/exhaustive_<timestamp>/
```

## Verify Setup

Run these before a real benchmark or before publishing changes:

```bash
python3 scripts/smoke_test.py
python3 scripts/discover_models.py --roots ~/.lmstudio/models ~/models --print
python3 scripts/model_registry.py list
bash scripts/run_exhaustive_benchmarks.sh --dry-run
```

The smoke test compiles the Python files, checks the backend-to-runner wiring, validates CSV columns, and scans public files for local development paths.

To verify the server launch commands without starting a model:

```bash
bash scripts/serve_local.sh qwen35-uncensored --backend llamacpp --dry-run
bash scripts/serve_local.sh gemma26 --backend mlx --dry-run
```

Use aliases that exist in your generated `configs/models.local.json`.

## Serve A Model

Aliases come from `configs/models.local.json`.

```bash
bash scripts/serve_local.sh qwen35-uncensored --backend llamacpp --host 127.0.0.1
bash scripts/serve_local.sh gemma26 --backend mlx --host 127.0.0.1
```

Both server launch paths bind to loopback by default.

You can also pass a direct model path:

```bash
bash scripts/serve_local.sh /path/to/model.gguf --backend llamacpp
bash scripts/serve_local.sh /path/to/mlx-model-dir --backend mlx
```

## Passes

| Pass | Intent |
| --- | --- |
| `micro` | Small prompt and short output for smoke checks |
| `normal` | Practical coding prompt |
| `high` | Hard architecture prompt with a 64K context budget and 2K output |
| `max` | Long-context stress: fills near the model context cap and asks for 4K output |

For text-generation backends, `max` builds deterministic synthetic context close to `ctx_cap - gen_tokens - margin`. The CSV `prompt_tokens` field is the proof that the backend actually consumed the long context.

## Repository Hygiene

Generated and local-only files are ignored:

- `configs/models.local.json`
- `results/`
- `tools/`
- `.venv/`
- `.claude/`
- `.remember/`
- `archive/`
- `__pycache__/`

Run the static guardrail before publishing changes:

```bash
python3 scripts/smoke_test.py
```

## Layout

```text
.
├── bench_tui.py
├── configs/
│   ├── model_catalog.json
│   └── models.local.example.json
├── scripts/
│   ├── aggregate_results.py
│   ├── bench_llamacpp_api.py
│   ├── bench_llamacpp_raw.py
│   ├── bench_mlx_api.py
│   ├── bench_mlx_direct.py
│   ├── check_model_availability.py
│   ├── discover_models.py
│   ├── llama_serve_menu.py
│   ├── model_registry.py
│   ├── prompts.py
│   ├── run_exhaustive_benchmarks.sh
│   ├── serve_local.sh
│   ├── smoke_test.py
│   └── stop_benchmarks.sh
└── requirements.txt
```

## License

MIT
