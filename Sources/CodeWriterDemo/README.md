# Python writer

A 0.5B model on this Mac writes Python from plain-English requests and the code runs: Qwen2.5-Coder-0.5B-Instruct
on Core ML ([qwen2.5-coder-0.5b-coreml](https://huggingface.co/FluidInference/qwen2.5-coder-0.5b-coreml)), the
prompt on the Neural Engine and the writing on the GPU. Nothing leaves the machine.

```bash
swift run -c release CodeWriterDemo                       # downloads the model (~1 GB) on first launch
Sources/CodeWriterDemo/demo.sh --autostart                # same app, plus one terminal: macmon above the live log
CODE_WRITER_MODEL_DIR=<folder> swift run -c release CodeWriterDemo   # a local copy instead of the download
```

**Play** (⌘↩) works down a list of ten short tasks, one per Python feature (built-ins, list comprehension, slicing,
dict, string methods, conversion, tuples, loops, `sorted`, `all()`). Each one is written live into the editor, then
`python3` runs the task's three asserts and the console shows each result. **Pause** stops after the current task;
**Replay** (⌘R) starts over. Type your own request in the bar at the bottom: it is written the same way and checked
to parse as Python (it has no tests). Running the model's code needs `python3` on the `PATH`.

The tasks come from MBPP (sanitized test split, CC-BY-4.0), with the first assert in the prompt so the model knows
the function name. The list is **hand-picked for the recording**: one original pick (sort a matrix by row sum) failed
and was replaced with one that passes. All ten pass; the unbiased numbers are below.

M5 Pro, macOS 27, greedy decoding:

| | |
|---|---|
| The ten demo tasks | 10/10 pass all asserts, ~50 s in total (95–330 tokens each) |
| Prompt (448 tokens, Neural Engine) | ~52 ms |
| Writing (GPU) | 36–51 tokens/s |
| MBPP sanitized test split (257) | 119/257 = 46.3% |
| HumanEval (164) | 89/164 = 54.3% (PyTorch fp32: 90/164 = 54.9%) |

`CodeWriterCheck` replays HumanEval through the Swift host and compares every token stream with a reference run.
