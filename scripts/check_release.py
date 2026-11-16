"""Validate the source/notebook release surface, not ignored local artifacts."""
from pathlib import Path
import ast
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/clean_notebooks.py"), "--check"],
        check=True,
    )
    checked = []
    for folder in [
        "parseq_ocr",
        "annotator",
        "scripts",
        "tests",
        "configs",
        "notebooks",
        "results",
    ]:
        for path in (ROOT / folder).rglob("*"):
            if (
                not path.is_file()
                or "_vendor" in path.parts
                or "__pycache__" in path.parts
            ):
                continue
            if (
                path.suffix
                not in (".py", ".yaml", ".json", ".ipynb", ".txt", ".html", ".c", ".cu")
                and path.name != "PILLOW_LICENSE"
            ):
                continue
            if folder == "annotator" and "static" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if any(
                prefix in text
                for prefix in tuple("/" + part + "/" for part in ("app", "mnt", "root"))
                + ("lp_" + "crops_" + "st_balanced",)
            ):
                raise SystemExit(
                    f"Local experiment path found in {path.relative_to(ROOT)}"
                )
            if path.suffix == ".py":
                ast.parse(text, filename=str(path))
            if path.suffix == ".ipynb":
                notebook = json.loads(text)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]), filename=path.name)
            checked.append(str(path.relative_to(ROOT)))
    print(
        f"Validated {len(checked)} release files; no saved notebook outputs or local experiment paths"
    )


if __name__ == "__main__":
    main()
