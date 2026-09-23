# Flappy Bird · local model comparison

```bash
swift build -c release --product GLiClassFlappyDemo
swift run -c release GLiClassFlappyDemo
```

The native macOS app plays a 60 Hz fixed-step game. Select GLiClass Edge Apps v2 LUT8, Laya multilingual E8, GLiNER 2.5 small/base/multilingual W8, Verdict, or one of the bridged runtimes (Kev 0.5B/0.6B, Decision 1.0 Kai/Lex, LFM2.5-350M-RLCD, Jeff, NanoJev). Each chooses **flap** or **coast** every six frames when its previous reply has arrived. Physics and pipe motion continue while Core ML runs; the optional delay slider exposes how slow replies affect play. The app shows model time, response age, late replies, score, and survival. Manual and heuristic controls let you replay a course by seed.

Model assets load from the FluidUse cache or their published Hugging Face packages. To use existing local assets, set `GLICLASS_MODEL_DIR`, `LAYA_MODEL_DIR`, `GLINER2_SMALL_MODEL_DIR`, `GLINER2_BASE_MODEL_DIR`, `GLINER2_MULTI_MODEL_DIR`, or `VERDICT_MODEL_DIR` before launching. Bridged models go through `PublishedCoreMLManager.load` (pinned snapshot plus a `uv` environment). NanoJev weights are not redistributed, so it needs `NANOJEV_MODEL_DIR` (a local conversion) and `NANOJEV_PYTHON`. The GLiNER directories need `tokenizer/tokenizer.json` and their published `.mlpackage` (or compiled `.mlmodelc`).

The state is a compact text description, and both labels include a 300 ms physics forecast. Option order alternates. The **Safety guard** toggle changes a choice when its forecast collides or clearly leaves the next pipe opening and the other action is safer. The app counts and shows those overrides. Turn the guard off to see the raw model policy. Forecasts are assistance given to every model; guarded survival does not measure unaided model control. Text truncation stops the run with an error.

For a bounded local check, `swift run -c release FlappyBirdCheck gliner2Multilingual` runs seeds 1 and 2 for at most 20 simulated seconds with that model's raw and guarded policies plus the heuristic. Running it without a valid name lists every model. `--raw-only --seeds=3,4` runs only raw choices on the two additional fixed courses. The check measures model calls but advances six game frames per decision without waiting for wall time. The app is the latency-sensitive demonstration; the check isolates action quality at the chosen 10 Hz cadence.

On the local Apple M5 Pro, GLiClass passed zero pipes on both selected seeds, ending at 1.40 and 1.55 simulated seconds. The heuristic passed 10 pipes on each and reached the 20-second cap. GLiClass selected flap on 13/14 and 13/16 decisions respectively, including every decision when coast was listed first. Its median Core ML call was 1.54 ms on both courses; the longest input was 69 tokens, with no truncation. The game outcome therefore does not suggest a useful standalone GLiClass controller with this prompt and forecast design. The measurements are a small exploratory check, not a general model ranking. Raw local output is in FluidAudio's ignored `.mobius/flappy-gliclass/RESULTS.json`.

With the safety guard, GLiClass reached the 20-second cap and passed 10 pipes on both courses. The guard changed **131/200** model decisions on seed 1 and **106/200** on seed 2, so that improvement is largely from deterministic safety logic.

On four fixed courses (seeds 1–4), **GLiNER 2.5 multilingual was the strongest raw policy in this small check**, though it passed only one pipe in total. No other model passed any pipe. Bridged call times include the Python worker round trip.

| Model | Pipes, seeds 1–4 | Combined survival | Median model call, per-course range | L128 package |
| --- | ---: | ---: | ---: | ---: |
| GLiNER 2.5 multilingual W8 | 1 | 10.98 s | 5.38–9.58 ms | 367 MB |
| Laya multilingual E8 | 0 | 5.93 s | 3.92–4.11 ms | 428 MB compiled |
| GLiClass Edge Apps v2 LUT8 | 0 | 5.83 s | 1.54–1.82 ms | 31 MB |
| GLiNER 2.5 base W8 | 0 | 5.40 s | 10.41–13.66 ms | 278 MB |
| Kev 0.6B FP16 (bridge) | 0 | 9.03 s | 11.8–23.8 ms | — |
| Decision 1.0 Lex FP16 (bridge) | 0 | 9.03 s | 5.9–6.0 ms | — |
| Kev 0.5B FP16 (bridge) | 0 | 8.70 s | 9.7–13.4 ms | — |
| Decision 1.0 Kai FP16 (bridge) | 0 | 8.60 s | 5.5–6.1 ms | — |
| GLiNER 2.5 small W8 | 0 | 8.53 s | 11.0–13.7 ms | — |
| LFM2.5-350M-RLCD FP16 (bridge) | 0 | 7.13 s | 147–167 ms | — |
| Jeff FP16 (bridge) | 0 | 6.65 s | 12.8–23.0 ms | — |
| NanoJev FP16 (local conversion, bridge) | 0 | 5.82 s | 75.2–81.5 ms | — |
| Verdict FP16 L128 | 0 | 5.60 s | 7.4–7.6 ms | — |
| Heuristic control | 40 | 80 s (four 20 s caps) | — | — |

LFM chose the option listed second in its context on every call (flap 0/9 when flap was listed first, 9/9 when coast was), which points to position bias rather than reading the forecast. LFM and NanoJev take longer than one 100 ms decision period, so the live app shows late replies for them. The first two seeds also included guarded runs for the original four models: GLiClass, Laya, and GLiNER base each reached 10 pipes on both; GLiNER multilingual reached 10 and 7. Guards changed 106–131 choices for the first three models and 21–32 for GLiNER multilingual. Guarded scores mainly reflect the deterministic safety filter. The small courses do not establish a general winner or measure visual game understanding. Raw local JSON rows are in FluidAudio's ignored `.mobius/flappy-gliclass/` directory.
