# Low-cost agent setup for DROCAT beginners

This repository's direct-analysis skill is designed for a coding agent that
can run local shell commands, inspect focused files, edit scripts, and report
results. A beginner can use Codex CLI, the Codex desktop app, or the Codex VS
Code extension; the DeepSeek provider configuration is shared by those Codex
clients.

## Recommended low-cost path

Use **DeepSeek V4 Flash** for routine DROCAT work: running a prepared script,
checking an output folder, explaining a traceback, or making a small targeted
patch. It is the model documented as currently supporting Codex and the
Responses API. Use a higher reasoning effort or a stronger model only for a
complex backend change, then switch back to Flash for routine work. Pricing and
availability can change, so check the platform before adding funds.

Official references:

- [DeepSeek Platform](https://platform.deepseek.com/)
- [Using the Responses API](https://api-docs.deepseek.com/guides/responses_api/)
- [DeepSeek Codex integration](https://api-docs.deepseek.com/quick_start/agent_integrations/codex/)

## Beginner setup: what an agent is

An agent is a program that can combine your natural-language request with tools
such as a terminal, file editor, and (when enabled) web search. For DROCAT, the
agent should be allowed to work only in the cloned repository and its selected
output folders. It should ask before changing credentials, downloading large
datasets, deleting results, or pushing code.

You do not need to paste the whole repository into a chat. Open the repository
as the agent's project, install the `drocat-usage` skill, and ask focused
requests such as:

```text
Use the DROCAT v4.5.0 direct-analysis skill. Run a cached FindPath analysis
from aMe12 to PPL101 in male-cns:v0.9, with max_interlayer=2, CSV output, and
save everything under local_data/agent_runs/aMe12_to_PPL101. Inspect the
outputs, summarize row counts, and finish by reporting the validated artifacts
and the next command to render the network. Do not stop at a plan.
```

For a code repair:

```text
Use the DROCAT v4.5.0 direct-analysis skill. Reproduce this direct-script
failure with the smallest query, inspect only the relevant script and backend
signature, patch the call, compile it, and run the focused regression test.
Finish by reporting the patch and test result. Do not change tokens or
dependencies.
```

## Configure Codex with DeepSeek

1. Install Codex CLI or launch the Codex desktop/VS Code client once. The first
   launch creates `~/.codex`.
2. Create a DeepSeek API key at the [DeepSeek Platform](https://platform.deepseek.com/).
   Treat it like a password; do not put it in this repository.
3. Use DeepSeek's official setup script. It backs up the existing
   `~/.codex/config.toml`, writes `~/.codex/models.json`, preserves compatible
   MCP/project settings, validates the files, and offers a restore option:

   macOS/Linux:

   ```bash
   bash <(curl -fsSL https://cdn.deepseek.com/api-docs/codex-deepseek-setup-en.sh)
   ```

   Windows PowerShell:

   ```powershell
   irm https://cdn.deepseek.com/api-docs/codex-deepseek-setup-en.ps1 | iex
   ```

   Review remote setup scripts according to your organization's security
   policy. Do not run them in the repository, and keep the generated backup.

4. Choose `deepseek-v4-flash` in the setup menu, then restart the Codex client
   if it does not appear. The current DeepSeek documentation says Codex and the
   Responses API support Flash; Pro support is listed as forthcoming, so do not
   hard-code Pro as the only model.
5. Open the DROCAT repository as the project and verify that the agent can run a
   harmless command such as `git status --short` and the launcher dry run:

   ```bash
   python skills/drocat-usage/scripts/run_direct.py \
     --conda-env drocat-4.5.0 \
     --script scripts/FindPath.py \
     --dry-run
   ```

## What the Codex configuration represents

The official setup writes a provider entry in `~/.codex/config.toml` and a
model catalog in `~/.codex/models.json`. The catalog describes the model slug,
context window, supported reasoning levels, shell/tool mode, and Responses API
compatibility. Keep these files user-level and outside the repository. Do not
copy a complete `models.json` into project source; the setup script is the
source of truth and can update it when DeepSeek changes its model metadata.

DeepSeek's Responses API example uses the OpenAI SDK with:

```python
from openai import OpenAI

client = OpenAI(
    api_key="<DEEPSEEK_API_KEY>",
    base_url="https://api.deepseek.com",
)
response = client.responses.create(
    model="deepseek-v4-flash",
    instructions="You are a careful coding agent.",
    input="Run the focused DROCAT analysis and report artifacts.",
)
print(response.output_text)
```

For this repository, prefer the Codex integration rather than writing a custom
API client: Codex already supplies shell execution, patching, context control,
and the `drocat-usage` skill. The Responses API is stateless for the documented
DeepSeek compatibility profile, so keep the repository path and task details in
the active agent session.

## Cost and reliability guardrails

- Use Flash with low reasoning for status checks, file listing, and prepared
  script execution.
- Use high/max reasoning only when the agent must understand a non-trivial
  backend contract or review a scientific result.
- Ask the agent to use `--dry-run`, small queries, `showfig=False`, and cached
  data before expensive runs.
- Keep API spend and data-download spend separate: the DeepSeek key does not
  replace NeuPrint/CAVE tokens.
- Review every patch that changes thresholds, dataset selection, pathfinding
  algorithm, or mesh transforms; these can change scientific conclusions.
