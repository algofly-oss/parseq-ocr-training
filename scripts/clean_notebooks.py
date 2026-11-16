"""Clear all notebook outputs; --check exits nonzero instead of modifying files."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    dirty = []
    for path in sorted((ROOT/"notebooks").glob("*.ipynb")):
        data = json.loads(path.read_text())
        changed = False
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                changed |= bool(cell.get("outputs")) or cell.get("execution_count") is not None
                cell["outputs"], cell["execution_count"] = [], None
        changed |= "widgets" in data.get("metadata", {})
        data["metadata"].pop("widgets", None)
        if changed:
            dirty.append(path.name)
            if not args.check:
                path.write_text(json.dumps(data, indent=1)+"\n")
    if args.check and dirty:
        raise SystemExit("Clear notebook outputs before publishing: "+", ".join(dirty))
    print("Notebook outputs clean" if not dirty else "Cleared: "+", ".join(dirty))


if __name__ == "__main__":
    main()
