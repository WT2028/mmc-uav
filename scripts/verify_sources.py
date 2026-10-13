#!/usr/bin/env python3
"""Verify the versioned source, model, configuration and dataset checksums."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'reproducibility.json').read_text())['files_sha256']
if not manifest:
    raise SystemExit('Empty source manifest')
failed = []
for relative, expected in manifest.items():
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failed.append(relative)
        continue
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failed.append(relative)
print(json.dumps({'files':len(manifest),'passed':not failed,'mismatches':failed},indent=2))
if failed:
    raise SystemExit(1)
