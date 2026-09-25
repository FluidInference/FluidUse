# Sort decisions

Streams Fastino's Fast Decisions development split (17 domains × 100 documents, Apache-2.0) through
GLiNER2.5-Decide on Core ML. Every question a document carries (for an email: category, action, needs reply,
is phishing) is answered in one call, and each answer is checked against the dataset's label.

```bash
swift run -c release SortDecisionsDemo
DECISIONS_AUTOPLAY=6 swift run -c release SortDecisionsDemo   # 6 documents in Show, then Turbo, no clicks
```

Headless: `swift run -c release SortDecisionsCheck --inflight=4` (M5 Pro, macOS 27: 1,700 documents, 2,900
decisions in about 24 s; 62.7% average over the 17 domains). Documents longer than the 256-token package are trimmed
from the end (198 of 1,700), which costs about half a point against untrimmed 512-token calls.

Scoring follows the dataset card (exact match per decision, multi-label heads answered with one label). Fastino's
published 60.2% is on a held-out test split that is not public; these are development-split numbers.
