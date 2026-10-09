"""Retain complete numeric records; repository-backed code stays in git."""
import hashlib
import json
import sys
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
out=ROOT/'deliverables'
out.mkdir(exist_ok=True)
members=[]
for name in ('design_results','validation_results'):
    members.extend(p for sub in ('arrays','runs') for p in (ROOT/name/sub).glob('*') if p.is_file())
members.extend(p for sub in ('independent_inputs','independent_meta') for p in (ROOT/sub).glob('*') if p.is_file())
archive=out/'spectral_residual_seeding_numeric_records_20261009.tar.xz'
if '--verify-existing' not in sys.argv:
    with tarfile.open(archive,'w:xz',preset=3) as tar:
        for p in sorted(members):tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
manifest=dict(archive=archive.name,size_bytes=archive.stat().st_size,
    sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),
    members=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(members)])
(ROOT/'numeric_records_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with tarfile.open(archive,'r:xz') as tar:
    assert len(tar.getmembers())==len(members)
    for record in manifest['members']:
        f=tar.extractfile(record['path'])
        assert f is not None and hashlib.sha256(f.read()).hexdigest()==record['sha256']
print(json.dumps({k:v for k,v in manifest.items() if k!='members'}|dict(member_count=len(members),all_member_hashes_verified=True)))
