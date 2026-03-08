# BulkFormer Agent Notes

## Ralph (default workflow)

- **Ralph is the default** for rollout work: run `./scripts/ralph/ralph.sh [N]` from the repo root. It uses the **Cursor CLI** (`agent`) by default; each iteration gets fresh context; progress lives in git and `scripts/ralph/prd.json` / `progress.txt`.
- The loop **does not trust the LLM** to say "done". It exits only when **external verification** passes: quality checks (pytest or compileall) and all stories in `prd.json` have `passes: true`. See `docs/development/ralph-workflow.md`.

## Repo Conventions

- Keep large model checkpoints, downloaded data, and machine-specific inputs out of commits.
- Prefer extracting and reusing logic from `bulkformer_extract_feature.ipynb`, `utils/`, and `model/` instead of duplicating the same behavior in new modules.
- When adding developer workflows, keep the repo-owned templates in `scripts/ralph/` and treat runtime state there as local-only.

## Validation

- Use targeted checks for the area you changed.
- Run `python -m pytest` when tests exist.
- If no tests exist yet, run lightweight import, CLI, or `python -m compileall` checks for touched Python modules before committing.

## Documentation

- Update `README.md` for top-level discoverability when adding new developer tooling or workflows.
- Keep `data/README.md`, `model/README.md`, and `utils/README.md` focused on assets and module-specific usage.
