#!/bin/sh
# Fetch the issues the IssueTriageDemo labels (needs the GitHub CLI, `gh auth login`).
#
#     Sources/IssueTriageDemo/fetch-issues.sh [output.json]
#
# Default output: ~/Library/Caches/FluidUse/issue-triage/issues.json (where the app looks when given no path).
# The file holds real issue authors; the app shows made-up handles in their place. Keep it out of the repository.
set -eu
out="${1:-$HOME/Library/Caches/FluidUse/issue-triage/issues.json}"
mkdir -p "$(dirname "$out")"
gh issue list -R vllm-project/semantic-router --state all --limit 1000 \
    --json number,title,body,labels,state,author,createdAt,comments > "$out"
echo "wrote $out"
