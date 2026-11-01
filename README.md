# olares-mode-switch

Private Olares Desktop app that flips the RTX GPU between:

- **Production** — stop Unsloth, resume Qwen3.6-27B (llama.cpp) with MemorySlice **23Gi**, then ensure Router + EmbeddingGemma + kbmcp are Running.
- **Training** — stop Qwen, resume Unsloth with **~22Gi**, then ensure Router + EmbeddingGemma + kbmcp (and Unsloth) are Running.

EmbeddingGemma (Intel) is resumed **without** `--compute-binding` so its binding is never touched.

## Requirements

- Olares OS **≥ 1.12.7**
- Admin must add `modeswitch` to `LOGIN_OLARES_CLI_ALLOW_LIST` (keep `lares`) before install — see install notes below
- Configure `PROD_APP`, `TRAIN_APP`, `GPU_NODE`, `GPU_DEVICE` to match your Market names / Accelerator device

## Image

`ghcr.io/wooj-commits/olares-mode-switch:0.1.0`

## Chart package

Build locally:

```bash
helm package chart/modeswitch -d .
# -> modeswitch-0.1.0.tgz
```

## Env vars

| Env | Default | Purpose |
|-----|---------|---------|
| `PROD_APP` | `qwen3627b` | Qwen llama.cpp clone Market name |
| `TRAIN_APP` | `unsloth` | Unsloth Market name |
| `GPU_NODE` | _(empty)_ | Node from Accelerator |
| `GPU_DEVICE` | _(empty)_ | Device id from Accelerator |
| `PROD_VRAM_GI` | `23` | Production MemorySlice |
| `TRAIN_VRAM_GI` | `22` | Training MemorySlice |
| `PRODUCTION_ENSURE_RUNNING` | `qwen3627b,router,embeddinggemma,kbmcp` | Must be Running after Production switch |
| `TRAINING_ENSURE_RUNNING` | `unsloth,router,embeddinggemma,kbmcp` | Must be Running after Training switch |

Update the ensure lists if your Market names differ (e.g. a custom EmbeddingGemma clone name).

## Allowlist (required once)

ConfigMap `login-olares-cli-allow-list` in namespace `os-framework`:

```yaml
LOGIN_OLARES_CLI_ALLOW_LIST: "lares,modeswitch"
```

Then restart the `app-service` StatefulSet in `os-framework`.

## License

Apache-2.0
