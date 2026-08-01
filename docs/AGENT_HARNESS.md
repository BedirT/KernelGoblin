# Codex Agent Harness

KernelGoblin keeps durable agent configuration in the smallest appropriate
surface:

- `AGENTS.md` contains concise repository rules, commands, verification gates,
  collaboration boundaries, and review rules.
- `.agents/skills/port-gpu-kernel/` contains the reusable, progressively loaded
  porting workflow and backend references.
- `.codex/config.toml` limits project subagent concurrency.
- `.codex/agents/` defines narrow read-only specialists. They are used only when
  a user asks for delegation or parallel work; the primary agent owns edits.
- `./kg validate` mechanically validates manifests and agent configuration.

This follows current Codex guidance to keep `AGENTS.md` small, use repository
skills for richer repeatable workflows, and use project-scoped agent TOML files
for specialized subagents. No lifecycle hook is installed: a Stop hook that
always builds GPU code would be expensive, platform-specific, and surprising.
CI and explicit `kg` commands are the appropriate enforcement layer. Hosted CI
compiles the Metal artifacts but does not claim physical-GPU execution; the
real-backend test runs locally or on a GPU-equipped self-hosted runner.

## Skill discovery

The official [OpenAI skills catalog](https://github.com/openai/skills) was
checked during initial setup. It provides general skills and an installer but
no focused CUDA/Metal kernel-port workflow, so this repository owns a local
`port-gpu-kernel` skill rather than depending on a superficially related skill.

Codex discovers repo-local skills under `.agents/skills`. Invoke this one
explicitly as `$port-gpu-kernel`, or let Codex select it for matching work.

## Trust and verification

Project `.codex/config.toml` and project-local agents load only after the
repository is trusted in Codex. Validate checked-in configuration with:

```sh
./kg validate
```

To inspect loaded instructions in a fresh Codex CLI session:

```sh
codex --ask-for-approval never "Summarize the active repository instructions and available KernelGoblin skills."
```

Do not put credentials, provider selection, personal model preferences, or
telemetry settings in the project config. Those belong in user or managed
configuration.
