# Sort anything

Streams 1,000 Wikipedia abstracts (a balanced, seeded sample of the DBpedia-14 test split) and sorts each one into
categories you can edit while it runs. The model is GLiNER2.5-Decide, converted to Core ML and run on-device with
the fp16 128-token package from `FluidInference/gliner2-5-decide-coreml`. Nothing is trained on DBpedia: the category
names are the only hint the model gets.

```bash
swift run -c release SortAnythingDemo
```

- **Show** animates one card at a time into its bucket; **Pace** sets cards per second.
- `SORT_AUTOSTART=show` or `SORT_AUTOSTART=turbo` starts a run on launch; `SORT_AUTOPLAY=12` flies 12 cards in
  Show mode and then switches to Turbo, with no clicks.
- **Turbo** keeps four model calls in flight and sorts as fast as the Mac allows.
- Type a new category to add it; the next item can land there. Removing a category keeps the items already sorted.
- "Matches DBpedia label" counts only items whose DBpedia class is an active category, so custom categories that
  split a class (for example `musician` next to `artist`) lower it even when the sort looks right.

Headless numbers on the same 1,000 items: `swift run -c release SortAnythingCheck --inflight=4`
(M5 Pro, macOS 27: 89.0% match, 1,000 items in about 5.9 s, peak memory about 1 GB; identical
predictions to Fastino's PyTorch release on all 1,000).

Data: DBpedia-14 test split (Zhang et al., 2015), Wikipedia text via DBpedia, CC BY-SA 3.0, fetched from the
Hugging Face dataset viewer on first launch and cached; it is not bundled. Model: fastino/GLiNER2.5-Decide,
Apache-2.0.
