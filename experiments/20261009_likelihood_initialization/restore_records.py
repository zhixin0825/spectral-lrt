#!/usr/bin/env python3
"""Verify and restore the frozen per-graph records using only the Python stdlib.

The manifest lists every original path, byte count, and SHA256. All archives,
members, record hashes, and destination conflicts are checked before writing.
Existing identical files are kept; different files and symlinks are refused.
This is an integrity check against the checked-in manifest, not a signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
MANIFEST_NAME = "record_archives_manifest.json"
SPLITS = ("final", "pilot_v2")
RECORD_RE = re.compile(
    r"(?:final|pilot_v2)/jobs/[a-z][a-z0-9_]*_n[1-9][0-9]*"
    r"_d[0-9]+(?:\.[0-9]+)?_s[0-9]+\.(?:json|npz)"
)
ARCHIVE_RE = re.compile(
    r"(?:final|pilot_v2)/archives/[a-z][a-z0-9_]*_n[1-9][0-9]*"
    r"_d[0-9]+(?:\.[0-9]+)?\.zip"
)
SHA_RE = re.compile(r"[0-9a-f]{64}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_relative_path(value: str, *, archive: bool = False) -> PurePosixPath:
    """Accept only the declared fixed-depth record/archive path shapes."""
    if not isinstance(value, str):
        raise ValueError("Manifest path must be a string")
    pattern = ARCHIVE_RE if archive else RECORD_RE
    path = PurePosixPath(value)
    if not pattern.fullmatch(value) or path.is_absolute():
        raise ValueError(f"Unexpected or unsafe relative path: {value!r}")
    if path.as_posix() != value or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Noncanonical relative path: {value!r}")
    return path


def checked_path(root: Path, relative: str | PurePosixPath) -> Path:
    """Reject symlinks anywhere under root, including a dangling final link."""
    path = root
    for component in PurePosixPath(relative).parts:
        path = path / component
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink: {path}")
    return path


def check_size_hash(entry: dict, data: bytes, name: str) -> None:
    size = entry.get("bytes")
    digest = entry.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"Invalid byte count: {name}")
    if not isinstance(digest, str) or not SHA_RE.fullmatch(digest):
        raise ValueError(f"Invalid SHA256: {name}")
    if len(data) != size or sha256(data) != digest:
        raise ValueError(f"Byte count or SHA256 mismatch: {name}")


def expected_records(root: Path, split: str) -> set[str]:
    """Derive exact member names from the frozen runner protocol."""
    protocol_path = checked_path(root, f"{split}/protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    models = protocol["models"].split(",")
    degrees = [float(value) for value in protocol["degrees"].split(",")]
    start, stop = [int(value) for value in protocol["seeds"].split(":")]
    n = int(protocol["n"])
    expected = {
        f"{split}/jobs/{model}_n{n}_d{degree:g}_s{seed}.{suffix}"
        for model in models
        for degree in degrees
        for seed in range(start, stop)
        for suffix in ("json", "npz")
    }
    for relative in expected:
        check_relative_path(relative)
    return expected


def load_and_validate_manifest(root: Path) -> dict:
    manifest_path = checked_path(root, MANIFEST_NAME)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported record manifest schema")
    if manifest.get("dataset") != "sbm_likelihood_init_20261009":
        raise ValueError("Unexpected record dataset")
    if set(manifest.get("splits", {})) != set(SPLITS):
        raise ValueError("Manifest must contain final and pilot_v2 splits")

    seen_archives: set[str] = set()
    seen_records: set[str] = set()
    split_records = {split: set() for split in SPLITS}
    split_bytes = {split: 0 for split in SPLITS}
    split_archives = {split: 0 for split in SPLITS}
    archive_bytes = 0
    for archive in manifest["archives"]:
        relative = check_relative_path(archive["path"], archive=True)
        split = relative.parts[0]
        if archive["path"] in seen_archives:
            raise ValueError(f"Duplicate archive: {relative}")
        seen_archives.add(archive["path"])
        if archive["record_files"] != len(archive["records"]):
            raise ValueError(f"Archive record count mismatch: {relative}")
        if 2 * archive["graphs"] != archive["record_files"]:
            raise ValueError(f"Archive graph/pair count mismatch: {relative}")
        pair_suffixes: dict[str, set[str]] = {}
        for record in archive["records"]:
            record_path = check_relative_path(record["path"])
            if record_path.parts[0] != split:
                raise ValueError(f"Archive mixes splits: {relative}")
            if record["path"] in seen_records:
                raise ValueError(f"Duplicate record: {record_path}")
            if not isinstance(record["bytes"], int) or record["bytes"] < 0:
                raise ValueError(f"Invalid record length: {record_path}")
            seen_records.add(record["path"])
            split_records[split].add(record["path"])
            split_bytes[split] += record["bytes"]
            pair_suffixes.setdefault(str(record_path.with_suffix("")), set()).add(record_path.suffix)
        if len(pair_suffixes) != archive["graphs"] or any(
            suffixes != {".json", ".npz"} for suffixes in pair_suffixes.values()
        ):
            raise ValueError(f"JSON/NPZ records are not paired: {relative}")
        split_archives[split] += 1
        archive_bytes += archive["bytes"]

    for split in SPLITS:
        if split_records[split] != expected_records(root, split):
            raise ValueError(f"Record paths do not exactly match {split}/protocol.json")
        computed = {
            "graphs": len(split_records[split]) // 2,
            "record_files": len(split_records[split]),
            "uncompressed_bytes": split_bytes[split],
            "archives": split_archives[split],
        }
        if manifest["splits"][split] != computed:
            raise ValueError(f"Split totals mismatch: {split}")
    totals = {
        "graphs": len(seen_records) // 2,
        "record_files": len(seen_records),
        "archives": len(seen_archives),
        "uncompressed_bytes": sum(split_bytes.values()),
        "archive_bytes": archive_bytes,
    }
    if manifest["totals"] != totals:
        raise ValueError("Manifest overall totals mismatch")
    return manifest


def validated_payloads(root: Path, manifest: dict, splits: set[str]) -> list[tuple[str, bytes]]:
    """Read all selected payloads without writing any files."""
    payloads: list[tuple[str, bytes]] = []
    for archive in manifest["archives"]:
        if PurePosixPath(archive["path"]).parts[0] not in splits:
            continue
        path = checked_path(root, archive["path"])
        if not path.is_file():
            raise ValueError(f"Missing regular archive: {path}")
        check_size_hash(archive, path.read_bytes(), archive["path"])
        expected = {record["path"]: record for record in archive["records"]}
        with zipfile.ZipFile(path) as compressed:
            infos = compressed.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError(f"ZIP members differ from manifest: {path}")
            for info in infos:
                check_relative_path(info.filename)
                mode = info.external_attr >> 16
                if info.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError(f"Refusing a nonregular ZIP entry: {info.filename}")
                record = expected[info.filename]
                if info.file_size != record["bytes"]:
                    raise ValueError(f"ZIP entry byte count mismatch: {info.filename}")
                data = compressed.read(info)  # also checks the ZIP CRC
                check_size_hash(record, data, info.filename)
                payloads.append((info.filename, data))
    if len(payloads) != sum(manifest["splits"][split]["record_files"] for split in splits):
        raise ValueError("Selected record count mismatch")
    return payloads


def restore(destination: Path, payloads: list[tuple[str, bytes]]) -> tuple[int, int]:
    """Preflight all destinations, then create only missing regular files."""
    existing = 0
    pending: list[tuple[str, bytes]] = []
    for relative, data in payloads:
        path = checked_path(destination, relative)
        if path.exists():
            if not path.is_file() or path.read_bytes() != data:
                raise ValueError(f"Refusing to overwrite different existing content: {path}")
            existing += 1
        else:
            pending.append((relative, data))
        parent = path.parent
        while parent != destination:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"Destination parent is not a directory: {parent}")
            parent = parent.parent

    for relative, data in pending:
        path = checked_path(destination, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        checked_path(destination, relative)
        # Exclusive creation prevents replacement if a file appears after preflight.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o644)
        with os.fdopen(fd, "wb") as output:
            output.write(data)
    return len(pending), existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check archives and all member bytes without extraction")
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--destination", type=Path, default=ROOT, help="Root under which final/jobs and pilot_v2/jobs will be created")
    args = parser.parse_args()
    manifest = load_and_validate_manifest(ROOT)
    splits = set(SPLITS) if args.split == "all" else {args.split}
    payloads = validated_payloads(ROOT, manifest, splits)
    created = existing = 0
    if not args.verify_only:
        if args.destination.is_symlink():
            raise ValueError(f"Refusing a symlink destination: {args.destination}")
        destination = args.destination.resolve()
        if destination.exists() and not destination.is_dir():
            raise ValueError(f"Destination is not a directory: {destination}")
        created, existing = restore(destination, payloads)
    print(json.dumps({
        "status": "verified" if args.verify_only else "restored",
        "splits": sorted(splits),
        "graphs": len(payloads) // 2,
        "record_files": len(payloads),
        "archives": sum(manifest["splits"][split]["archives"] for split in splits),
        "uncompressed_bytes": sum(len(data) for _, data in payloads),
        "created_files": created,
        "identical_existing_files": existing,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
        print(f"Record verification/restoration failed: {exc}", file=sys.stderr)
        sys.exit(1)
