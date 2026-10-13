"""Shared experiment file helpers."""
from pathlib import Path
import hashlib
import json
ADAPTER = Path(__file__).with_name('controller_factory.py')
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(Path(path).read_text())
def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')
