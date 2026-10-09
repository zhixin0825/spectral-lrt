"""Validate and restore the retained numerical archive, then rebuild inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'numeric_records_manifest.json').read_text(encoding='utf-8'))
    with args.archive.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['sha256']
    with tarfile.open(args.archive,'r:xz') as tar:
        assert len(tar.getmembers())==len(manifest['members'])
        for item in manifest['members']:
            target=(ROOT/item['path']).resolve()
            target.relative_to(ROOT.resolve())
            member=tar.getmember(item['path'])
            assert member.isfile()
            data=tar.extractfile(member).read()
            assert len(data)==item['size'] and hashlib.sha256(data).hexdigest()==item['sha256']
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
    dest=ROOT/'design_inputs';dest.mkdir(exist_ok=True)
    for path in sorted((ROOT/'design_results/arrays').glob('*.npz')):
        with np.load(path,allow_pickle=False) as a:
            np.savez_compressed(dest/path.name,**{key:a[key] for key in
                ('u','lam','truth_evaluation_only','baseline_km_X')})
    print('Verified and restored full records; rebuilt design_inputs.')

if __name__=='__main__':main()
