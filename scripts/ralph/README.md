# Ralph Workflow For BulkFormer

This directory vendors the minimal [Ralph](https://github.com/snarktank/ralph) assets and configures **Cursor CLI as the default** so the diagnostics rollout runs as repeated **fresh-context** agent iterations. Progress persists in git and on-disk state (`prd.json`, `progress.txt`); the loop does **not** trust the LLM to say "done" — it stops only when **external verification** passes.

## Why Ralph

- **Fresh context every iteration** — No context accumulation; each run is cheap and clear.
- **Code is the memory** — Git history and `prd.json` / `progress.txt` carry progress, not tokens.
- **External verification** — The script stops only when quality checks (pytest or compileall) pass *and* every story in `prd.json` has `passes: true`. "It's better to fail predictably than succeed unpredictably."

## What Is Here

- `ralph.sh`: loop runner; **default tool is Cursor** (`agent`). Optional: `--tool amp` or `--tool claude`.
- `CLAUDE.md`: single prompt used for Cursor, Amp, and Claude Code (tool-agnostic).
- `prompt.md`: Amp-specific prompt (optional; CLAUDE.md is used when not overridden).
- `prd.template.json`: BulkFormer rollout template with story boundaries.
- `progress.template.txt`: starter progress log with repo-specific patterns.
- `.gitignore`: ignores Ralph runtime state so runs do not pollute git status.

## Default: Cursor CLI

1. **Install Cursor CLI** (if not already):
   ```bash
   curl https://cursor.com/install -fsSL | bash
   ```
   Or on macOS: `brew install --cask cursor-cli`.

2. **Authenticate**:
   ```bash
   agent login
   ```
   Or set `CURSOR_API_KEY` in your environment.

3. **Run Ralph** (Cursor is the default; no `--tool` needed):
   ```bash
   ./scripts/ralph/ralph.sh 10
   ```
   Or with a specific model (e.g. for cost savings):
   ```bash
   ./scripts/ralph/ralph.sh --model grok 10
   ```

## Other backends

- Amp: `./scripts/ralph/ralph.sh --tool amp 10`
- Claude Code: `./scripts/ralph/ralph.sh --tool claude 10`

## Branch And Commit Workflow

1. Copy `prd.template.json` to `prd.json`.
2. Set `branchName` to a dedicated feature branch from `main` (e.g. `ralph/bulkformer-dx-toolkit`).
3. Copy `progress.template.txt` to `progress.txt` if you want the initial codebase patterns.
4. Run `./scripts/ralph/ralph.sh [max_iterations]` (Cursor by default).
5. Each iteration completes one story and one focused commit; the loop exits only when verification and `prd.json` say all stories are done.

## External Verification

After each iteration the script:

1. Runs **quality checks**: `python -m pytest` if tests exist; otherwise `python -m compileall` on `utils`, `model`, `bulkformer_dx`.
2. Checks **prd.json**: all `userStories[].passes` must be `true`.

Success = both pass. The agent can reply "COMPLETE", but the loop does not exit until verification passes.

## Runtime Files (untracked)

- `prd.json`
- `progress.txt`
- `archive/`
- `.last-branch`

## Repo-Specific Notes

- The prompts tell Ralph not to commit checkpoints (`model/*.pt`) or other large local assets.
- Validation is pragmatic: pytest when present, otherwise compileall for the main packages.
