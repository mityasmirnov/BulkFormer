#!/bin/bash
# Ralph loop for BulkFormer: fresh-context iterations with external verification.
# Vendored from https://github.com/snarktank/ralph; Cursor CLI is the default.
# Usage: ./scripts/ralph/ralph.sh [--tool cursor|amp|claude] [--model MODEL] [max_iterations]

set -euo pipefail

TOOL="cursor"
CURSOR_MODEL=""
MAX_ITERATIONS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool)
      TOOL="$2"
      shift 2
      ;;
    --tool=*)
      TOOL="${1#*=}"
      shift
      ;;
    --model)
      CURSOR_MODEL="$2"
      shift 2
      ;;
    --model=*)
      CURSOR_MODEL="${1#*=}"
      shift
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      fi
      shift
      ;;
  esac
done

if [[ "$TOOL" != "cursor" && "$TOOL" != "amp" && "$TOOL" != "claude" ]]; then
  echo "Error: invalid tool '$TOOL'. Use 'cursor', 'amp', or 'claude'."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
ARCHIVE_DIR="$SCRIPT_DIR/archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.last-branch"
PROMPT_FILE="$SCRIPT_DIR/CLAUDE.md"

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required. Install it before running Ralph."
  exit 1
fi

if [[ "$TOOL" == "cursor" ]] && ! command -v agent >/dev/null 2>&1; then
  echo "Error: Cursor CLI (agent) not found. Install it: curl https://cursor.com/install -fsSL | bash"
  echo "Then run: agent login   (or set CURSOR_API_KEY)"
  exit 1
fi

if [ ! -f "$PRD_FILE" ]; then
  echo "Error: $PRD_FILE not found."
  echo "Copy scripts/ralph/prd.template.json to scripts/ralph/prd.json and edit it first."
  exit 1
fi

# Archive previous run if branch changed
if [ -f "$PRD_FILE" ] && [ -f "$LAST_BRANCH_FILE" ]; then
  CURRENT_BRANCH="$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")"
  LAST_BRANCH="$(<"$LAST_BRANCH_FILE" 2>/dev/null || echo "")"

  if [ -n "$CURRENT_BRANCH" ] && [ -n "$LAST_BRANCH" ] && [ "$CURRENT_BRANCH" != "$LAST_BRANCH" ]; then
    DATE="$(date +%Y-%m-%d)"
    FOLDER_NAME="$(echo "$LAST_BRANCH" | sed 's|^ralph/||')"
    ARCHIVE_FOLDER="$ARCHIVE_DIR/$DATE-$FOLDER_NAME"

    echo "Archiving previous run for branch: $LAST_BRANCH"
    mkdir -p "$ARCHIVE_FOLDER"
    [ -f "$PRD_FILE" ] && cp "$PRD_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_FOLDER/"
    echo "Archived to: $ARCHIVE_FOLDER"

    echo "# Ralph Progress Log" > "$PROGRESS_FILE"
    echo "Started: $(date)" >> "$PROGRESS_FILE"
    echo "---" >> "$PROGRESS_FILE"
  fi
fi

CURRENT_BRANCH="$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")"
if [ -n "$CURRENT_BRANCH" ]; then
  echo "$CURRENT_BRANCH" > "$LAST_BRANCH_FILE"
fi

if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# Ralph Progress Log" > "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

# External verification: do not trust the LLM to say "done". Use disk state + quality checks.
run_verification() {
  cd "$REPO_ROOT" || return 1

  # 1) Quality checks: pytest if it runs tests, else compileall for key Python paths
  if ! python -m pytest -q 2>/dev/null; then
    local rc=$?
    # 0=pass, 1=fail, 2=no tests collected, 4=no tests run / interrupted
    if [[ $rc -eq 1 ]]; then
      echo "Verification: pytest failed (exit $rc)."
      return 1
    fi
    for d in utils model bulkformer_dx; do
      [[ -d "$d" ]] && python -m compileall -q "$d" 2>/dev/null || true
    done
  fi

  # 2) All stories passed? (source of truth: prd.json on disk)
  local all_pass
  all_pass="$(jq -r '[.userStories[]? | .passes] | all' "$PRD_FILE" 2>/dev/null || echo "false")"
  if [[ "$all_pass" == "true" ]]; then
    echo "Verification: all stories passed in prd.json."
    return 0
  fi
  return 1
}

echo "Starting BulkFormer Ralph (default: Cursor) — tool=$TOOL, max_iterations=$MAX_ITERATIONS"
echo "External verification: quality checks + prd.json (all passes). Loop does not trust LLM 'done'."
echo ""

for i in $(seq 1 "$MAX_ITERATIONS"); do
  echo ""
  echo "==============================================================="
  echo " Ralph Iteration $i of $MAX_ITERATIONS ($TOOL)"
  echo "==============================================================="

  if [[ "$TOOL" == "cursor" ]]; then
    CMD=(agent -p --trust)
    [ -n "$CURSOR_MODEL" ] && CMD+=(--model "$CURSOR_MODEL")
    OUTPUT="$(cd "$REPO_ROOT" && "${CMD[@]}" "$(cat "$PROMPT_FILE")" 2>&1 | tee /dev/stderr)" || true
  elif [[ "$TOOL" == "amp" ]]; then
    OUTPUT="$(cat "$PROMPT_FILE" | amp --dangerously-allow-all 2>&1 | tee /dev/stderr)" || true
  else
    OUTPUT="$(claude --dangerously-skip-permissions --print < "$PROMPT_FILE" 2>&1 | tee /dev/stderr)" || true
  fi

  # Optional: stop if agent explicitly said complete (still verify below)
  if echo "$OUTPUT" | grep -q " COMPLETE "; then
    echo "Agent reported COMPLETE; running external verification..."
  fi

  if run_verification; then
    echo ""
    echo "Ralph completed: all stories passed and verification succeeded."
    echo "Completed at iteration $i of $MAX_ITERATIONS"
    exit 0
  fi

  echo "Iteration $i done. Continuing..."
  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS) without completing all stories."
echo "See $PROGRESS_FILE for the run log."
exit 1
