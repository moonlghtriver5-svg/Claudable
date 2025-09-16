# Claude Code Router Setup

This guide walks through using [claude-code-router](https://github.com/musistudio/claude-code-router) with Claudable to proxy
[OpenRouter](https://openrouter.ai/) models such as **Qwen 3 Coder**. The router exposes an Anthropic-compatible Messages API,
so Claudable can reuse its Claude tooling (terminal, MCP, file edits) without any additional plugins.

> **Prerequisites**
>
> - OpenRouter account with API key (create one from the OpenRouter dashboard)
> - Node.js ≥ 18 and Python ≥ 3.10 (already required for Claudable)
> - A running Claudable API instance (`npm run dev` from the repo root)

---

## 1. Install the router CLI

```bash
npm install -g @musistudio/claude-code-router
```

Verify the installation:

```bash
ccr --version
```

---

## 2. Configure OpenRouter as the upstream provider

Create `~/.claude-code-router/config.json` (the CLI creates the folder on first
run). The example below maps the router's default model to OpenRouter's
`qwen/qwen3-coder` target and forwards your API key in the `x-api-key`
header that OpenRouter expects:

```json
{
  "servers": {
    "default": {
      "port": 3456
    }
  },
  "providers": {
    "openrouter": {
      "baseURL": "https://openrouter.ai/api",
      "apiKey": "${OPENROUTER_API_KEY}",
      "defaultHeaders": {
        "HTTP-Referer": "https://claudable.local"
      }
    }
  },
  "routers": {
    "default": {
      "defaultProvider": "openrouter",
      "routes": {
        "openrouter,qwen/qwen3-coder": {
          "default": true
        }
      }
    }
  }
}
```

You can keep the `${OPENROUTER_API_KEY}` placeholder and export the key in your
shell before launching the router:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

---

## 3. Launch the router

```bash
ccr start --router default
```

By default the router listens on port **3456**. You can change this with
`--port` or by editing the config shown above.

Leave the process running; Claudable will stream chat sessions to this HTTP
server.

---

## 4. Connect Claudable to the router

Claudable reads the following environment variables at startup:

| Variable | Required | Description |
| --- | --- | --- |
| `CLAUDE_CODE_ROUTER_URL` | Optional | Base URL to the running router (e.g. `http://127.0.0.1:3456`). If omitted, Claudable falls back to `http://127.0.0.1:${CLAUDE_CODE_ROUTER_PORT}` or the Codespaces forwarding URL. |
| `CLAUDE_CODE_ROUTER_PORT` | Optional | Overrides the fallback port (default `3456`). Useful when the router runs on a different port but you do not want to hardcode the full URL. |
| `CLAUDE_CODE_ROUTER_API_KEY` | Optional | API key forwarded to the router via `x-api-key` and `Authorization` headers. Set this if your router requires authentication. |
| `CLAUDE_CODE_ROUTER_MODEL` | Optional | Overrides the default model mapping (`openrouter,qwen/qwen3-coder`). |

Export the variables and restart the API process so it picks up the changes:

```bash
export CLAUDE_CODE_ROUTER_URL="http://127.0.0.1:3456"
export CLAUDE_CODE_ROUTER_API_KEY="${OPENROUTER_API_KEY}"
export CLAUDE_CODE_ROUTER_MODEL="openrouter,qwen/qwen3-coder"
# restart the API (for example by re-running `npm run dev`)
```

---

## 5. GitHub Codespaces quick start

Codespaces run both the Claudable backend and the router inside the same
container. In most situations you can rely on the default loopback URL, but the
forwarding domain is handy when you want to hit the router from your local
machine or share the endpoint with a teammate.

```bash
# Optional: expose the router through the public forwarding domain
export CLAUDE_CODE_ROUTER_PORT=3456
export CLAUDE_CODE_ROUTER_URL="https://${CLAUDE_CODE_ROUTER_PORT}-${CODESPACE_NAME}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"

# Ensure the Claudable backend can call the router directly inside the Codespace
export CLAUDE_CODE_ROUTER_API_KEY="${OPENROUTER_API_KEY}"  # if needed
```

Then, in two terminals inside the Codespace:

```bash
# Terminal 1 – start the router
OPENROUTER_API_KEY=sk-or-... ccr start --router default --port ${CLAUDE_CODE_ROUTER_PORT:-3456}

# Terminal 2 – start Claudable (automatically picks up the env vars)
npm run dev
```

When the web UI loads, open **Settings → AI Agents** and press **Refresh
Status**. The **Claude Code Router** entry should report as installed, and you
can now select it from the home page assistant dropdown.

---

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Home page option stays greyed out | Confirm the router is reachable: `curl $CLAUDE_CODE_ROUTER_URL/health`. The API needs a `200` response. |
| `Router request failed: ...` in the terminal | Check that the router process is running and that your OpenRouter API key has sufficient quota. |
| Missing tool results or partial edits | The router streams Anthropic Messages events. Ensure you are running the latest `@musistudio/claude-code-router` release so tool invocations are forwarded correctly. |

For additional router configuration options, refer to the
[claude-code-router README](https://github.com/musistudio/claude-code-router#readme).
