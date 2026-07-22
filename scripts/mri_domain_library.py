import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / "config" / "mri" / "v2" / "domain_library.json"
SPLITS = ("calibration", "validation", "held_out")


def load_and_validate(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    if data.get("format") != "evm-mri-domain-library-v2":
        failures.append("unsupported format")
    groups = data.get("groups", {})
    domains = data.get("domains", {})
    if not groups or not domains:
        failures.append("groups or domains missing")
    prompts_seen = {}
    for name, domain in domains.items():
        if domain.get("group") not in groups:
            failures.append(f"{name}: unknown group")
        for field in ("description", "excludes"):
            if not domain.get(field):
                failures.append(f"{name}: missing {field}")
        contrasts = domain.get("contrast_domains", [])
        if not contrasts:
            failures.append(f"{name}: contrast domains missing")
        for contrast in contrasts:
            if contrast not in domains or contrast == name:
                failures.append(f"{name}: invalid contrast {contrast}")
        for split in SPLITS:
            prompts = domain.get(split, [])
            minimum = 4 if split == "calibration" else 1
            if len(prompts) < minimum:
                failures.append(f"{name}: {split} requires at least {minimum} prompts")
            for index, prompt in enumerate(prompts, 1):
                if not isinstance(prompt, str) or len(prompt.strip()) < 12:
                    failures.append(f"{name}: invalid {split} prompt {index}")
                normalized = " ".join(prompt.lower().split())
                if normalized in prompts_seen:
                    failures.append(f"duplicate prompt: {name}/{split}/{index} and {prompts_seen[normalized]}")
                prompts_seen[normalized] = f"{name}/{split}/{index}"
    if failures:
        raise ValueError("; ".join(failures))
    return data


def selected_domains(data, names):
    if not names:
        return data["domains"]
    requested = [value.strip() for value in names.split(",") if value.strip()]
    missing = [name for name in requested if name not in data["domains"]]
    if missing:
        raise ValueError(f"unknown domains: {','.join(missing)}")
    return {name: data["domains"][name] for name in requested}


def compile_suite(data, split, names=None, limit=0):
    domains = selected_domains(data, names)
    return {
        "format": "evm-mri-diagnostic-suite-v1",
        "source_format": data["format"],
        "source_version": data["version"],
        "split": split,
        "description": f"Compiled {split} MRI payload suite",
        "domains": {
            name: {
                "description": row["description"],
                "group": row["group"],
                "excludes": row["excludes"],
                "contrast_domains": row["contrast_domains"],
                "prompts": row[split][:limit or None],
            }
            for name, row in domains.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Validate and compile the EVM MRI domain library.")
    parser.add_argument("command", choices=("validate", "compile", "catalog"))
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--split", choices=SPLITS, default="calibration")
    parser.add_argument("--domains", help="comma-separated domain IDs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = load_and_validate(args.library)
    prompt_count = sum(len(row[split]) for row in data["domains"].values() for split in SPLITS)
    if args.command == "validate":
        print(f"MRI library: {len(data['groups'])} groups | {len(data['domains'])} domains | {prompt_count} prompts | PASS")
        return
    if not args.out:
        raise SystemExit("--out is required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "compile":
        suite = compile_suite(data, args.split, args.domains, args.limit)
        args.out.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
        count = sum(len(row["prompts"]) for row in suite["domains"].values())
        print(f"MRI suite: {len(suite['domains'])} domains | {count} {args.split} prompts | PASS")
        return
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("group", "domain", "description", "excludes", "contrast_domains", "calibration", "validation", "held_out"))
        writer.writeheader()
        for name, row in data["domains"].items():
            writer.writerow({"group": row["group"], "domain": name, "description": row["description"], "excludes": row["excludes"],
                             "contrast_domains": ";".join(row["contrast_domains"]),
                             **{split: len(row[split]) for split in SPLITS}})
    print(f"MRI catalog: {len(data['domains'])} domains | PASS")


if __name__ == "__main__":
    main()
