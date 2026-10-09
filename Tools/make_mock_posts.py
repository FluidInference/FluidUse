"""Builds Sources/TopicSortDemo/Resources/mock-posts.jsonl from the hand-written fictional posts in
Tools/mock-posts/<theme>.jsonl (six themes x four subtopics x 420 posts). Adds made-up authors, times and media
flags, then shuffles; seeded, so every machine gets the same stream:

    python3 Tools/make_mock_posts.py
"""
import json, pathlib, random

random.seed(20261008)
root = pathlib.Path(__file__).resolve().parent
FIRST = ["sam", "maya", "jordan", "lee", "nora", "omar", "priya", "theo", "zoe", "ines", "kai", "rosa", "ben",
         "lena", "yuki", "max", "ada", "ravi", "mila", "ivo", "june", "tariq", "elsa", "hugo", "noor", "felix", "ana"]
LAST = ["builds", "codes", "bakes", "runs", "travels", "lifts", "saves", "looks_up", "writes", "cooks", "ships",
        "hikes", "rides", "reads", "makes", "draws"]

posts = []
for path in sorted((root / "mock-posts").glob("*.jsonl")):
    for line in path.read_text().splitlines():
        row = json.loads(line)
        posts.append({
            "author": f"{random.choice(FIRST)}_{random.choice(LAST)}",
            "createdAt": f"{random.randint(1, 23)} hours ago", "text": row["text"],
            "media": [random.choice(["photo", "video"])] if random.random() < 0.2 else [],
        })
random.shuffle(posts)
for index, post in enumerate(posts):
    post["id"] = str(100_000 + index)
out = root.parent / "Sources/TopicSortDemo/Resources/mock-posts.jsonl"
out.write_text("".join(json.dumps(post, ensure_ascii=False) + "\n" for post in posts))
print(f"{len(posts)} mock posts → {out}")
