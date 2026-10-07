# IssueTriageDemo

A GitHub-style dark issues page for a fictional repo, **routerlabs / semantic-switch**, listing 1,000 real open-source
issues (vllm-project/semantic-router) with their labels stripped and their authors replaced by made-up handles. Press
**✦ Triage with Decision 2.0** and Decision-2.0-Kai-0.6B on Core ML labels every issue live: type, owning workgroup,
priority, needs-info and good-first-issue, **5 decisions per issue in one model call**.

- Label chips pop in as each issue is labeled; the list auto-scrolls to keep the current row centered.
- The sidebar routes issues to workgroups, re-sorted live by count; the stats bar shows issues triaged, labels decided,
  ms per issue and issues / sec.
- Click an issue to see the model's probabilities per question (type, workgroup, priority with its urgency score,
  flags).
- Priority cuts the Score answer's expected level (0-2): P0 at >= 1.452, P1 at >= 1.251, else P2 (this repo's p90 /
  p50). needs-info and good first issue need yes >= 0.6.

The model runs back to back in a detached task; the UI only consumes results. Each triaged issue prints one line to
stdout (`66.1 ms  #4612  enhancement  wg/router-models-inference-runtime  priority/P2`), plus a summary at the end.

## Run

```sh
Sources/IssueTriageDemo/fetch-issues.sh          # gh CLI → ~/Library/Caches/FluidUse/issue-triage/issues.json
swift run -c release IssueTriageDemo [issues.json] [--auto]
```

The issues file holds real author logins (shown only as hashed handles), so it stays out of the repository. `--auto`
starts triage 1.5 s after the model is ready, for hands-free recording. The model snapshot downloads on first run
(`Decision2ModelStore.ensure(.kai)`, 1.1 GB).

## Measured

M-series Mac, 1,000 issues: 66.8 ms per issue, 15.0 issues / sec, 5,000 decisions in 67 s, with the UI live. Labels
match the Python Core ML reference demo on every issue compared (626 of 626).
