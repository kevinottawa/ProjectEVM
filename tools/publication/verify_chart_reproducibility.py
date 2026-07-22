"""Regenerate publication figures and verify byte-stable output from committed data."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "figures"
PLOTS = (
    "scripts/plot_final_proof.py",
    "scripts/plot_controlled_final.py",
    "scripts/plot_production_evm.py",
    "scripts/plot_ability_workflow.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def figure_hashes() -> dict[Path, str]:
    return {path.relative_to(ROOT): digest(path) for path in FIGURES.rglob("*.png")}


def main() -> None:
    before = figure_hashes()
    failures: list[str] = []
    for relative in PLOTS:
        result = subprocess.run(
            [sys.executable, relative],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            detail = next(
                (
                    line.strip()
                    for line in reversed((result.stderr + "\n" + result.stdout).splitlines())
                    if line.strip()
                ),
                "no diagnostic emitted",
            )
            failures.append(f"{relative}: {detail}")
    after = figure_hashes()
    changed = sorted(
        set(before).symmetric_difference(after)
        | {path for path in before.keys() & after.keys() if before[path] != after[path]}
    )

    print(f"Figure generators: {len(PLOTS) - len(failures)}/{len(PLOTS)} passed")
    print(f"Publication PNGs checked: {len(after)}")
    print(f"Changed or added PNGs: {len(changed)}")
    if failures:
        print("Failed generators: " + " | ".join(failures))
    if changed:
        print("Changed outputs: " + ", ".join(str(path) for path in changed[:10]))
    if failures or changed:
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
