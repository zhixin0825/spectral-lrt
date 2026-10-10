"""Build a data-only evidence archive; source code is versioned separately."""
from pathlib import Path
import csv
import hashlib
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'spectral_partition_mcmc_evidence_20261011.zip'


def main():
    folders = sorted(p for p in ROOT.glob('results_*') if p.is_dir() and p.name != 'results_smoke')
    files = sorted(p for folder in folders for p in folder.rglob('*') if p.is_file())
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow(['path', 'size_bytes', 'sha256'])
    with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            writer.writerow([relative, len(data), hashlib.sha256(data).hexdigest()])
            archive.writestr(relative, data)
        archive.writestr('manifest.csv', manifest.getvalue())
        archive.writestr('README.txt', 'Spectral-only partition MCMC experimental evidence.\n'
                         'Raw A and truth are included for evaluation only.\n'
                         'Primary inference uses only retained U and signed eigenvalues.\n'
                         'Main, confirmatory, long-budget, oracle diagnostics, and post-hoc variants are separate directories.\n'
                         'Best-score and final endpoints must not be combined.\n'
                         'Source and detailed protocol: https://github.com/zhixin0825/spectral-lrt/tree/main/experiments/20261011_spectral_partition_mcmc\n')
    metadata = dict(filename=OUTPUT.name, size_bytes=OUTPUT.stat().st_size,
                    sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                    raw_file_count=len(files), result_directories=[p.name for p in folders],
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))})
    (ROOT / 'archive_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files) + 2
    print(json.dumps({k: v for k, v in metadata.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
