#!/usr/bin/env python3
"""Regenerate the pinned file manifest for the Python-assisted Core ML repositories.

Every file at the pinned revision is recorded with its size and SHA-256. LFS digests come from
the Hub tree API; small git files are downloaded and hashed. Run after changing a revision:

    python3 Tools/pin_published_coreml.py > Sources/FluidUse/Resources/published-coreml-manifest.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
import urllib.request

REVISIONS = {
    "kev-0-5b": ("FluidInference/kev-0-5b-coreml", "06b6bad2c1209d96d1f87600ccfcaee01959a788"),
    "kev-0.6b": ("FluidInference/kev-0.6b-coreml", "f2a24a115626e2be75f0a8816448ff0f34520c16"),
    "decision-1.0-kai": ("FluidInference/decision-1.0-kai-coreml", "bdb0bc4e4c0d360ab48e093b4f14bbce18369e61"),
    "decision-1.0-lex": ("FluidInference/decision-1.0-lex-coreml", "6ca0547adf1997e51197a5ca0b5e45a5cb7802a2"),
    "lfm2-5-350m-rlcd": ("FluidInference/lfm2-5-350m-rlcd-coreml", "78cd6a54f3704c9cd4fc8bce904455da8786f3b0"),
    "jeff": ("FluidInference/jeff-coreml", "37e70eee651a3de62d93de229c459e850afed539"),
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def files(repository: str, revision: str) -> list[dict]:
    tree = json.loads(fetch(f"https://huggingface.co/api/models/{repository}/tree/{revision}?recursive=true"))
    result = []
    for entry in sorted((item for item in tree if item["type"] == "file"), key=lambda item: item["path"]):
        path = entry["path"]
        if "lfs" in entry:
            digest = entry["lfs"]["oid"]
        else:
            quoted = urllib.parse.quote(path)
            body = fetch(f"https://huggingface.co/{repository}/resolve/{revision}/{quoted}")
            if len(body) != entry["size"]:
                raise SystemExit(f"{repository}/{path}: downloaded {len(body)} bytes, tree says {entry['size']}")
            digest = hashlib.sha256(body).hexdigest()
        result.append({"path": path, "size": entry["size"], "sha256": digest})
    return result


def main() -> None:
    manifest = {
        model: {"repository": repository, "revision": revision, "files": files(repository, revision)}
        for model, (repository, revision) in REVISIONS.items()
    }
    json.dump(manifest, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
