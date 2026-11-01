# Admin UI steps (no CLI)

## 1. LOGIN_OLARES_CLI_ALLOW_LIST

1. Open **Control Hub** from the Olares Desktop.
2. In the left navigation, click **Browse**.
3. Expand namespace **`os-framework`**.
4. Expand **ConfigMaps**, select **`login-olares-cli-allow-list`**.
5. Click the **edit** (edit_square) button top-right to open the YAML editor.
6. Under `data`, set exactly:

```yaml
LOGIN_OLARES_CLI_ALLOW_LIST: "lares,modeswitch"
```

(Keep `lares` if Lares is installed; the env **replaces** the default list, it does not merge.)

7. Click **Confirm** to save.
8. Still under **`os-framework`**, open **StatefulSets** (or Workloads) → **`app-service`** → **Restart** (type `app-service` to confirm).
   - Required: ConfigMap values via `envFrom` are only loaded at pod start.
9. Wait until `app-service` is healthy (green).

## 2. Discover Market app names (Settings UI)

1. Open **Settings → Applications** (or **Market → My Olares**).
2. Note the **exact app names** shown for:
   - Your Qwen3.6-27B llama.cpp clone → set as `PROD_APP`
   - Unsloth → set as `TRAIN_APP`
   - Router, EmbeddingGemma / embedding app, Knowledge Base MCP (`kbmcp`)
3. Put those names (comma-separated, no spaces preferred) into
   `PRODUCTION_ENSURE_RUNNING` and `TRAINING_ENSURE_RUNNING` when installing/editing the Mode Switch app envs.

## 3. Discover GPU node + device id (Settings UI)

1. Open **Settings → Accelerator** (formerly GPU).
2. Open the RTX 5090 card details.
3. Copy **Node** → `GPU_NODE` and **Device ID** → `GPU_DEVICE`
   (same values `olares-cli settings compute list` would show as NODE / DEVICE-ID).
4. Confirm the card is in **MemorySlice** (or switch support type if needed) before Production/Training resumes with a Gi quota.

## 4. Install Mode Switch

1. Market → Local / Upload chart → `modeswitch-0.1.0.tgz` (or sideload).
2. Fill envs: `PROD_APP`, `TRAIN_APP`, `GPU_NODE`, `GPU_DEVICE`, ensure lists.
3. Install (CPU-only). Entrance is **private** (Olares login).
