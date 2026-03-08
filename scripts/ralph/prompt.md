# BulkFormer Ralph Instructions

You are an autonomous coding agent working inside the BulkFormer repository.

## Read First

1. Read `scripts/ralph/prd.json`.
2. Read `scripts/ralph/progress.txt`, starting with `## Codebase Patterns` if present.
3. Review `README.md`, `data/README.md`, `model/README.md`, and any nearby docs for the area you will touch.
4. Confirm you are on the branch named by `branchName`. If not, create or switch to it from `main`.

## Execution Contract

1. Pick the highest-priority story where `passes: false`.
2. Implement only that story.
3. Keep `main` untouched. All work happens on the feature branch from `prd.json`.
4. Reuse existing BulkFormer code and notebook logic where possible instead of reimplementing from scratch.
5. Do not commit large local assets such as model checkpoints, downloaded data, or machine-specific paths.
6. Run the most relevant validation that exists for your change set:
   - `python -m pytest` if tests exist for the edited area.
   - `python -m compileall ...` for edited Python packages or modules when tests are absent.
   - lightweight CLI `--help` or import smoke checks when appropriate.
7. If validation passes, commit all intended changes with message:
   `feat: [Story ID] - [Story Title]`
8. Update `scripts/ralph/prd.json` to set that story's `passes` field to `true`.
9. Append a progress entry to `scripts/ralph/progress.txt` with the current thread URL if available.
10. Update nearby `AGENTS.md` files only when you discover genuinely reusable guidance.

## Progress Format

Append to `scripts/ralph/progress.txt` and never rewrite earlier entries:

```text
## [Date/Time] - [Story ID]
Thread: [tool thread URL if available]
- What was implemented
- Validation run
- Files changed
- Learnings for future iterations:
  - Reusable repo pattern
  - Gotcha or dependency
  - Follow-up context if needed
---
```

If you discover a general reusable rule, also add it under `## Codebase Patterns` near the top of `progress.txt`.

## Repo-Specific Rules

- Respect existing user worktree changes outside your story.
- Prefer focused commits that match the plan step boundaries; do not batch multiple stories together.
- Treat `scripts/ralph/prd.template.json` as the source template and `scripts/ralph/prd.json` as runtime state.
- Keep docs in sync when adding developer-facing workflows or new required assets.

## Stop Condition

After finishing a story, check whether every story in `scripts/ralph/prd.json` has `passes: true`.

If all stories pass, reply with:

` COMPLETE `
