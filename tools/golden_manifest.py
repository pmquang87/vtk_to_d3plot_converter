import argparse
import hashlib
import sys
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Manage golden files manifest.")
    parser.add_argument("command", choices=["generate", "verify"], help="Command to run")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    golden_dir = repo_root / "tests" / "data" / "golden"
    manifest_file = golden_dir / "MANIFEST.sha256"

    if args.command == "generate":
        with manifest_file.open("w", encoding="utf-8") as out:
            for filepath in sorted(golden_dir.rglob("*")):
                if filepath.is_file() and filepath.name != "MANIFEST.sha256":
                    rel_path = filepath.relative_to(golden_dir).as_posix()
                    sha = compute_sha256(filepath)
                    out.write(f"{sha}  {rel_path}\n")
        print(f"Generated manifest at {manifest_file}")

    elif args.command == "verify":
        if not manifest_file.exists():
            print("ERROR: Manifest file not found.")
            sys.exit(1)
        
        errors = 0
        with manifest_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                expected_sha, rel_path = line.split("  ", 1)
                filepath = golden_dir / rel_path
                if not filepath.exists():
                    print(f"FAIL: Missing file {rel_path}")
                    errors += 1
                else:
                    actual_sha = compute_sha256(filepath)
                    if actual_sha != expected_sha:
                        print(f"FAIL: SHA mismatch for {rel_path}")
                        errors += 1
        
        if errors == 0:
            print("OK: All files match.")
            sys.exit(0)
        else:
            print(f"FAILED: {errors} mismatches found.")
            sys.exit(1)

if __name__ == "__main__":
    main()
