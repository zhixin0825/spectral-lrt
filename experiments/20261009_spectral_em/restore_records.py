"""Restore original spectral inputs and trajectories from checked archives."""
import hashlib
import json
import zipfile
from pathlib import Path


def extract_checked(archive, destination, expected):
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected, str(archive)
    root=destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            resolved=(root/member.filename).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError('Archive path escapes destination')
        z.extractall(root)


def main():
    root=Path(__file__).resolve().parent
    final=root/'final'
    manifest=json.loads((final/'archive_manifest.json').read_text())
    for item in manifest['archives']:
        extract_checked(final/'archives'/item['name'], final, item['sha256'])
    independent=root/'independent_20000'
    item=json.loads((independent/'archive_manifest.json').read_text())
    extract_checked(independent/item['file'], independent,item['sha256'])
    print('Restored 240 prior and 120 independent spectral inputs with trajectories.')


if __name__=='__main__': main()
