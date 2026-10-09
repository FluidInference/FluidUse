# Code search

Ask a codebase questions in plain English and get the function that does it, even when the answer shares none of
the question's words. EmbeddingGemma 2 (text model, 100% on the Neural Engine) embeds every Swift function and type;
a question is one more embedding and one matrix-vector product.

```bash
swift run -c release CodeSearchDemo                       # indexes ~/Documents/FluidAudio
swift run -c release CodeSearchDemo --repo=~/Code/MyApp
Sources/CodeSearchDemo/demo.sh                            # same app, plus one terminal: macmon above the live log
```

**Play** (↩) runs the show once: **📥 index** every declaration (`CodeChunker`: one chunk per `func` / `init` / type,
doc comment included, embedded as `title: <file> <symbol> | text: <code>` truncated to 128 tokens), then **🔎 eight
example questions** (each typed; the top result's code opens, then #2 and #3; the header shows how many files an
exact-phrase grep finds), then **⚡ search speed** for 30 s (~70 different questions back to back, 64 at a time,
ranked against every function with one matrix multiply). **Pause** (Space) holds it anywhere; **Replay** (⌘R) starts
over. When it is done, ask your own questions. `--segment=`, `--dwell=`, `--tokens=`, `--autostart` adjust it.

FluidAudio on an M5 Pro (macOS 27):

| | |
|---|---|
| Index 5,617 functions and types (483 files, 125k lines) | 28.5 s, 197 per second |
| The eight example questions | right function first on all eight; exact-phrase grep finds 0 files for seven |
| Search speed | 712 searches/s, ~1.4 ms each |
| Known-answer set (`CodeSearchCheck --eval`, 16 questions) | 11/16 first, 12/16 in the top 5 |

`CodeSearchCheck` scores chunk lengths and title formats: file name + symbol in the title beat symbol alone or file
alone; 64 tokens indexes ~1.7× faster with a lower top-1 (9/16) and a higher top-5 (14/16).
