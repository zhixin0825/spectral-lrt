#!/usr/bin/env python3
"""Package the frozen experiment, predictions, analysis and report."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'deliverables'
ARCHIVE = OUT / 'sbm_likelihood_initialization_experiment.zip'
required = [
    'experiment.py', 'poisson_gaussian.py', 'analyze_results.py',
    'build_report.py', 'package_results.py', 'README.txt', 'requirements.txt',
    'EXPERIMENT_LIMITATIONS.txt', 'audit_final.py', 'audit_results.txt',
    'pilot_diagnostics.py', 'pilot_diagnostics.json', 'final/protocol.json',
    'pilot_v2/protocol.json',
    'deliverables/sbm_likelihood_initialization_report.html',
]
files = [ROOT / p for p in required]
files += list((ROOT / 'final' / 'jobs').glob('*.json'))
files += list((ROOT / 'final' / 'jobs').glob('*.npz'))
files += list((ROOT / 'pilot_v2' / 'jobs').glob('*.json'))
files += list((ROOT / 'pilot_v2' / 'jobs').glob('*.npz'))
files += [p for p in OUT.iterdir() if p.suffix in ('.csv', '.json', '.png', '.pdf')]
files += [p for p in (ROOT / 'final_audit').iterdir() if p.suffix in ('.csv', '.json')]
files = sorted(set(files))
missing = [str(p) for p in files if not p.is_file()]
if missing:
    raise FileNotFoundError(missing)
manifest = []
prefix = 'sbm_likelihood_initialization'
with ZipFile(ARCHIVE, 'w', ZIP_DEFLATED, compresslevel=6) as z:
    for p in files:
        name = p.relative_to(ROOT).as_posix()
        raw = p.read_bytes()
        manifest.append({'path':name, 'bytes':len(raw),
                         'sha256':hashlib.sha256(raw).hexdigest()})
        z.writestr(prefix + '/' + name, raw)
    z.writestr(prefix + '/MANIFEST_SHA256.json',
               json.dumps(manifest, ensure_ascii=False, indent=2))
with ZipFile(ARCHIVE) as z:
    bad = z.testzip()
    if bad:
        raise RuntimeError('Archive integrity error: ' + bad)
print(json.dumps({'archive':str(ARCHIVE), 'files':len(files)+1,
                  'bytes':ARCHIVE.stat().st_size,
                  'sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()}))
