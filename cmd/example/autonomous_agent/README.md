# Autonomous UTCP Agent CLI (OpenClaw-style)

This project provides an OpenClaw-like CLI on top of `github.com/Protocol-Lattice/go-agent`:

- UTCP codemode via `WithCodeModeUtcp(...)`
- Specialist agents registered as UTCP tools via `RegisterAsUTCPProvider(...)`
- Command-driven UX: `agent`, `loop`, `chat`, `tools`, `doctor`

## Commands

- `agent`: single-turn execution (default target is `coordinator`)
- `loop`: autonomous multi-step execution until `AUTONOMOUS_DONE`
- `chat`: interactive REPL with runtime agent switching
- `tools`: list configured tools or live registered UTCP tools
- `doctor`: validate provider/model/env setup

## Specialist UTCP Tools

Default tools are prefixed with `local_` and registered at runtime:

- `local_researcher.run`
- `local_builder.run`
- `local_reviewer.run`

## Common Flags

- `--provider` (default from `LLM_PROVIDER`, fallback `gemini`)
- `--model` (default from `LLM_MODEL`, fallback `gemini-2.5-pro`)
- `--session-id` (default from `AGENT_SESSION`, fallback `autonomous-session`)
- `--context-window` (default `20`)
- `--tool-prefix` (default from `UTCP_TOOL_PREFIX`, fallback `local.`)

## Usage

```bash
# Single turn through coordinator
go run ./cmd/autonomous-agent agent \
  --message "Draft rollout plan for UTCP migration"

# Single turn through specialist
go run ./cmd/autonomous-agent agent \
  --agent reviewer \
  --message "What could fail in this deploy plan?"

# Autonomous loop
go run ./cmd/autonomous-agent loop \
  --goal "Design and verify a UTCP-based repository triage workflow" \
  --max-steps 8

# Interactive mode
go run ./cmd/autonomous-agent chat --goal "Prepare release plan"

# Tools
go run ./cmd/autonomous-agent tools
go run ./cmd/autonomous-agent tools --live

# Environment checks
go run ./cmd/autonomous-agent doctor
```

## Chat Commands

Inside `chat`:

- `/help`
- `/tools`
- `/agent <name>`
- `/exit`

## Notes

- You must provide provider credentials through environment variables expected by your selected provider.
- Runtime now validates provider credentials before bootstrapping agents and reports missing keys explicitly.
- If `ADK_EMBED_PROVIDER` is unset, it is inferred for known providers (`gemini`, `openai`, `ollama`); otherwise set it manually.
- `loop` completes when the model emits `AUTONOMOUS_DONE`; otherwise it exits at `--max-steps`.
- `agent --thinking`, `agent --local`, and `agent --deliver` are included for OpenClaw-like UX compatibility.
