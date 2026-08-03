"""Golden test data checksum manifest — detect silent drift in test fixtures.

Usage:
    # Generate/update manifest:
    .venv\\Scripts\\python.exe tools\\golden_manifest.py generate

    # Verify golden data against manifest:
    .venv\\Scripts\\python.exe tools\\golden_manifest.py verify
"""

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "tests" / "data"
MANIFEST_PATH = DATA_DIR / "golden" / "MANIFEST.sha256"

# Directories to checksum
GOLDEN_DIRS = [
    DATA_DIR / "golden",
    DATA_DIR / "port_dialect",
]

SKIP_EXTENSIONS = {".pyc", ".pyo", ".log", ".tmp"}


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(1 << 16):
            sha.update(chunk)
    return sha.hexdigest()


def _collect_files() -> list[Path]:
    files = []
    for d in GOLDEN_DIRS:
        if d.exists():
            for f in sorted(d.rglob("*")):
                if (
                    f.is_file()
                    and f.suffix not in SKIP_EXTENSIONS
                    and f.name != "MANIFEST.sha256"
                ):
                    files.append(f)
    return files


def main():
    parser = argparse.ArgumentParser(description="Manage golden files manifest.")
    parser.add_argument("command", choices=["generate", "verify"], help="Command to run")
    args = parser.parse_args()

    if args.command == "generate":
        files = _collect_files()
        with MANIFEST_PATH.open("w", encoding="utf-8") as out:
            for filepath in files:
                rel_path = filepath.relative_to(DATA_DIR).as_posix()
                sha = compute_sha256(filepath)
                size = filepath.stat().st_size
                out.write(f"{sha}  {size}  {rel_path}\n")
        print(f"Generated manifest with {len(files)} files -> {MANIFEST_PATH}")

    elif args.command == "verify":
        if not MANIFEST_PATH.exists():
            print("ERROR: Manifest file not found. Run 'generate' first.")
            sys.exit(1)

        errors = 0
        checked = 0
        manifest_entries = set()
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("  ", 2)
                if len(parts) == 3:
                    expected_sha, expected_size, rel_path = parts
                    expected_size = int(expected_size)
                else:
                    # Legacy format without size
                    expected_sha, rel_path = line.split("  ", 1)
                    expected_size = None
                manifest_entries.add(rel_path)
                filepath = DATA_DIR / rel_path
                if not filepath.exists():
                    print(f"  MISSING: {rel_path}")
                    errors += 1
                else:
                    actual_sha = compute_sha256(filepath)
                    if actual_sha != expected_sha:
                        print(f"  CHANGED: {rel_path}")
                        errors += 1
                    if expected_size is not None:
                        actual_size = filepath.stat().st_size
                        if actual_size != expected_size:
                            print(f"  SIZE: {rel_path} (expected {expected_size}, got {actual_size})")
                            errors += 1
                checked += 1

        # Check for new files not in manifest
        disk_files = {f.relative_to(DATA_DIR).as_posix() for f in _collect_files()}
        new_files = disk_files - manifest_entries
        if new_files:
            print(f"  NEW ({len(new_files)} files not in manifest):")
            for nf in sorted(new_files)[:10]:
                print(f"    {nf}")
            if len(new_files) > 10:
                print(f"    ... and {len(new_files) - 10} more")

        if errors == 0:
            print(f"OK: all {checked} golden files match manifest")
            sys.exit(0)
        else:
            print(f"FAILED: {errors} mismatches found.")
            sys.exit(1)


if __name__ == "__main__":
    main()
