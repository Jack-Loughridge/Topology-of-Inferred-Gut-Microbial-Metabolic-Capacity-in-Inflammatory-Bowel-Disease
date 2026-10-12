"""Verify immutable payload bytes, never rewrite installed code."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def verify():
    records=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    for name,want in records.items():
        p=ROOT/name
        if p.is_symlink() or not p.is_file():raise RuntimeError('Missing or linked package file: '+name)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=want:raise RuntimeError('Changed package file: '+name)
    found={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json'}
    if found!=set(records):raise RuntimeError('Unexpected package file inventory.')
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
if __name__=='__main__':print('PACKAGE VERIFIED',verify())
