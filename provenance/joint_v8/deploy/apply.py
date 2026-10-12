#!/usr/bin/env python3
"""Apply the checked v8 source overlay, not a new numerical implementation."""
from pathlib import Path, PurePosixPath
import base64,hashlib,json,zlib
R=Path.cwd();D=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
G=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
expected=(D/'parts.sha1').read_text().splitlines()
assert len(expected)==50
parts=[];bad=[]
for i,line in enumerate(expected):
    want,name=line.split();p=D/name;b=p.read_bytes()
    if G(b)!=want:bad.append({'path':name,'actual':G(b),'expected':want})
    parts.append(b)
assert not bad,json.dumps(bad,indent=2)
compressed=base64.b64decode(b''.join(parts),validate=True)
assert H(compressed)=='0d24f37ae79d308a5620c930a34b5c19c799d8ed10d8d73edcdae0a93a251e6e'
raw=zlib.decompress(compressed)
assert H(raw)=='b1948d27b0019ce5269590b2c8226d8e910025745f1656444865f9891e681fe4'
obj=json.loads(raw)
assert obj['schema']=='reviewed-publication-overlay-v1' and len(obj['files'])==55
for rel,v in obj['files'].items():
    p=PurePosixPath(rel);assert not p.is_absolute() and '..' not in p.parts
    assert H(v['text'].encode())==v['sha256'],rel
    dest=R/rel
    assert not dest.exists(),('Refusing to overwrite an existing overlay file',rel)
    assert not any(q.is_symlink() for q in dest.parents)
script=obj['root_updates_script']
assert H(script.encode())=='d62f00d589b40e7fe088558971b7c1e365f4cdbc395ce89176bf98b1c98c53a4'
for rel,v in obj['files'].items():
    dest=R/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(v['text'],encoding='utf-8')
exec(compile(script,'checked_root_updates.py','exec'),{'__name__':'__main__'})
checks={rel:H((R/rel).read_bytes()) for rel in obj['files']}
assert all(checks[rel]==v['sha256'] for rel,v in obj['files'].items())
(R/'provenance/joint_v8/DEPLOYMENT_CHECKS.json').write_text(json.dumps({'payload_sha256':H(raw),'files':checks,'source_model_changed':False},indent=2)+'\n')
print('PASS: exact 55-file reviewed overlay applied. No production model fitting.')
