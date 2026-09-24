#!/usr/bin/env python3
"""Regenerate the pinned file manifest for FluidInference/cua-s1-4b-coreml.

    python3 Tools/pin_cua_s1_4b.py <revision> > Sources/FluidUse/Resources/cua-s1-4b-manifest.json
"""

from __future__ import annotations

import json
import sys

from pin_published_coreml import files

REPOSITORY = "FluidInference/cua-s1-4b-coreml"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    revision = sys.argv[1]
    manifest = {"repository": REPOSITORY, "revision": revision, "files": files(REPOSITORY, revision)}
    json.dump(manifest, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
