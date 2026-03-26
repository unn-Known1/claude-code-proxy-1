# Anthropic API Proxy for Gemini & OpenAI Models 🔄

**Use Anthropic clients (like Claude Code) with Gemini, OpenAI, or direct Anthropic backends.** 🤝

A proxy server that lets you use Anthropic clients with Gemini, OpenAI, or Anthropic models themselves (a transparent proxy of sorts), all via LiteLLM. 🌉

## Quick Start ⚡

### Prerequisites

- OpenAI API key 🔑
- Google AI Studio (Gemini) API key (if using Google provider) 🔑
- Google Cloud Project with Vertex AI API enabled (if using Application Default Credentials for Gemini) ☁️
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

#### From source

1. **Clone this repository**:
   ```bash
   git clone https://github.com/1rgs/claude-code-proxy.git
   cd claude-code-proxy
   ```

2. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Configure Environment Variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your API keys and model configurations.

4. **Run the server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```

#### Docker

```bash
docker run -d --env-file .env -p 8082:8082 ghcr.io/1rgs/claude-code-proxy:latest
```

### Using with Claude Code 🎮

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `GEMINI_API_KEY` | Your Google AI Studio key | Optional |
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Optional |
| `PREFERRED_PROVIDER` | Provider: `openai`, `google`, or `anthropic` | `openai` |
| `BIG_MODEL` | Model for `sonnet` requests | `gpt-4.5` |
| `SMALL_MODEL` | Model for `haiku` requests | `gpt-4o-mini` |

### Model Mapping

- `PREFERRED_PROVIDER=openai` (default): `haiku`/`sonnet` map to `openai/SMALL_MODEL`/`openai/BIG_MODEL`
- `PREFERRED_PROVIDER=google`: `haiku`/`sonnet` map to `gemini/SMALL_MODEL`/`gemini/BIG_MODEL`
- `PREFERRED_PROVIDER=anthropic`: Pass directly to Anthropic (no remapping)

### Supported Models

**OpenAI**: gpt-4.5, gpt-4o, gpt-4o-mini, gpt-4.1  
**Gemini**: gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash

## How It Works 🧩

1. **Receives** requests in Anthropic's API format 📥
2. **Translates** to OpenAI/Gemini format via LiteLLM 🔄
3. **Sends** to backend provider 📤
4. **Converts** response back to Anthropic format 🔄
5. **Returns** to client ✅

Supports both streaming and non-streaming responses. 🌊

## Contributing 🤝

Contributions welcome! Please submit a Pull Request. 🎁