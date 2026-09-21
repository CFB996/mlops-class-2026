"""
Notebooks must be committed without their outputs.

A notebook stores its outputs inside the same JSON file as its code, so re-running a cell
rewrites the file even when the code did not change. Committed outputs mean every diff is
unreadable, every merge conflicts, and a stray plot can add megabytes to the repository.

Clear outputs before committing: in JupyterLab, Kernel -> Restart Kernel and Clear
Outputs of All Cells, then save.
"""

import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IGNORED_DIRS = {".git", ".ipynb_checkpoints", "artifacts", "db", "node_modules"}


def find_notebooks():
    found = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for filename in filenames:
            if filename.endswith(".ipynb"):
                full = os.path.join(dirpath, filename)
                found.append(os.path.relpath(full, REPO_ROOT))
    return sorted(found)


NOTEBOOKS = find_notebooks()


def test_repository_has_notebooks():
    """Guard against the check silently passing because it found nothing to check."""
    assert NOTEBOOKS, "no notebooks found — has the search path moved?"


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_has_no_outputs(notebook_path):
    with open(os.path.join(REPO_ROOT, notebook_path)) as f:
        notebook = json.load(f)

    offenders = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            offenders.append(f"cell {index}: {len(cell['outputs'])} output(s)")
        if cell.get("execution_count") is not None:
            offenders.append(f"cell {index}: execution_count={cell['execution_count']}")

    assert not offenders, (
        f"{notebook_path} was committed with outputs:\n  "
        + "\n  ".join(offenders)
        + "\n\nClear them with: Kernel -> Restart Kernel and Clear Outputs of All Cells"
    )
