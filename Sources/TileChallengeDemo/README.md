# Picture challenge

"Select all images with a …" photo grids, solved on the Mac by EmbeddingGemma 2 with no training: each photo is
embedded (vision encoder on the GPU, text model on the Neural Engine) and ticked when the asked-for category beats
all 21 (traffic light, fire hydrant, bus, bicycle, motorcycle, boat, bridge, palm tree, zebra, …). Photos and labels
are a Caltech-256 subset in `~/Library/Application Support/FluidUse/Datasets/caltech256-select` (`manifest.json` of
`{file, label}`); the grids are generated here, nothing comes from or is sent to any real CAPTCHA service.

```bash
swift run -c release TileChallengeDemo        # first run downloads ~0.9 GB of models
Sources/TileChallengeDemo/demo.sh             # same, plus one terminal: macmon above the live log
```

It runs on its own: three mixed grids and three lookalike grids (bus vs fire truck, horse vs zebra…) at a watchable pace, then **turbo**
(`--grids=200` by default): the next batch is decoded and embedded while the current grid is shown, and the counters
show photos per second, milliseconds per grid and accuracy against the true labels. Space stops or reruns it;
`--manual` waits for Space.
