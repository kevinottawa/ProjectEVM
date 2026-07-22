import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Verify local Markdown links and images used by the EVM paper.")
    parser.add_argument("--paper", type=Path, default=ROOT / "docs" / "EVM_Paper.md")
    args = parser.parse_args()
    text = args.paper.read_text(encoding="utf-8")
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    local = []
    missing = []
    for raw in targets:
        target = raw.strip("<>").split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        path = (args.paper.parent / target).resolve()
        local.append(path)
        if not path.exists():
            missing.append(path)
    print(f"Paper references: {len(local)} local | {len(missing)} missing")
    if missing:
        for path in missing:
            print(path)
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
