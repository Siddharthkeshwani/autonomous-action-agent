from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent.parent


def read_values(path):
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"\'').strip()
    return values


def main():
    required = read_values(ROOT / ".env.example")
    values = read_values(ROOT / ".env")
    missing = []
    if sys.version_info < (3, 11):
        print("Python 3.11+: MISSING")
        missing.append("Python 3.11+")
    else:
        print(f"Python {sys.version.split()[0]}: OK")
    if shutil.which("npm"):
        print("npm: OK")
    else:
        print("npm: WARNING (needed in Phase 8)")
    for key in required:
        if values.get(key):
            print(f"{key}: OK")
        elif key == "SUPABASE_READONLY_URL":
            print(f"{key}: OPTIONAL (until Phase 1)")
        else:
            print(f"{key}: MISSING")
            missing.append(key)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
