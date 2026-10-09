"""Write 1000 topic-balanced public tweets (cardiffnlp/tweet_topic_multi) for EvokeSearchDemo.

    uv run --with pandas --with pyarrow --with huggingface_hub python fetch-posts.py posts.json
    swift run -c release EvokeSearchDemo --posts posts.json

Handles and links are masked upstream; the demo shows fictional author names. Check the dataset's terms before
redistributing the output.
"""

import json
import re
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

N = 1000
df = pd.read_parquet(
    hf_hub_download(
        "cardiffnlp/tweet_topic_multi", "tweet_topic_multi/train_all-00000-of-00001.parquet", repo_type="dataset"
    )
)


def clean(text):
    text = re.sub(r"\{\{URL\}\}", "", text)
    text = re.sub(r"\{\{USERNAME\}\}", "@user", text)
    text = re.sub(r"\{@(.+?)@\}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


df["clean"] = df.text.map(clean)
df = df[(df.clean.str.len() >= 40) & (df.label_name.map(len) > 0)].drop_duplicates("clean")
df["topic"] = df.label_name.map(lambda labels: list(labels)[0])
groups = [g.sample(frac=1, random_state=0) for _, g in df.groupby("topic")]
rows, i = [], 0
while len(rows) < N:
    for g in groups:
        if i < len(g) and len(rows) < N:
            rows.append(g.iloc[i])
    i += 1
posts = [{"text": r.clean, "topic": r.topic} for r in rows]
out = sys.argv[1] if len(sys.argv) > 1 else "posts.json"
json.dump(posts, open(out, "w"), ensure_ascii=False, indent=0)
print(f"wrote {len(posts)} posts across {len(groups)} topics to {out}")
