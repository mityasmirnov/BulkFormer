# Ralph Workflow

BulkFormer uses a **repo-local Ralph** workflow under `scripts/ralph/` so the diagnostics rollout runs as repeated **fresh-context** agent iterations. **Cursor CLI is the default**; the loop relies on **external verification** (quality checks + `prd.json` on disk), not on the LLM saying "done."

## Philosophy

- **Fresh context, persistent code** — Each iteration starts with ~5K tokens of context; progress lives in git and in `scripts/ralph/prd.json` / `progress.txt`. No context accumulation.
- **External verification** — The loop does not stop when the model says "COMPLETE". It stops when:
  1. Quality checks pass (pytest, or compileall on key packages).
  2. Every story in `prd.json` has `passes: true`.
- **"It's better to fail predictably than succeed unpredictably."** — [Geoffrey Huntley](https://ghuntley.com/ralph/); see also [Cursor forum: The Ralph Wiggum Technique](https://forum.cursor.com).

## State Between Iterations

- Git history (commits from previous iterations)
- `scripts/ralph/prd.json` (which stories are done)
- `scripts/ralph/progress.txt` (learnings and patterns)

## Files

- `scripts/ralph/ralph.sh` — Loop runner; default tool is Cursor (`agent`).
- `scripts/ralph/CLAUDE.md` — Single prompt for Cursor, Amp, and Claude Code.
- `scripts/ralph/prd.template.json` — BulkFormer rollout template.
- `scripts/ralph/progress.template.txt` — Starter progress log.

## Branch Discipline

- Use one dedicated feature branch for the rollout; keep `main` untouched.
- Set `branchName` in `scripts/ralph/prd.json` to that branch.
- Run Ralph only from that branch; each iteration produces one focused commit.

## Setup

```bash
# Install Cursor CLI (default backend)
curl https://cursor.com/install -fsSL | bash
agent login

# One-time Ralph setup
cp scripts/ralph/prd.template.json scripts/ralph/prd.json
cp scripts/ralph/progress.template.txt scripts/ralph/progress.txt
chmod +x scripts/ralph/ralph.sh
```

Edit `scripts/ralph/prd.json`: set `branchName` and adjust stories if needed.

## Running (Cursor by default)

```bash
./scripts/ralph/ralph.sh 10
```

With a specific model (e.g. for cost):

```bash
./scripts/ralph/ralph.sh --model grok 10
```

Other backends:

```bash
./scripts/ralph/ralph.sh --tool amp 10
./scripts/ralph/ralph.sh --tool claude 10
```

## Verification Behavior

After each iteration the script:

1. Runs **pytest** if the project has tests; on failure the loop continues (next iteration can fix). If there are no tests, runs **compileall** on `utils`, `model`, `bulkformer_dx`.
2. Reads **prd.json** and checks that every `userStories[].passes` is `true`.

Exit 0 only when both conditions are satisfied. The agent may output "COMPLETE", but the script ignores that for the stop condition.

## Safety

- `scripts/ralph/.gitignore` keeps runtime files untracked.
- Prompts instruct the agent not to commit local checkpoints or external datasets.
