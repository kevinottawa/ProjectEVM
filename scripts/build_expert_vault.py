"""Build a byte-accurate external-vault manifest for MoE expert tensors.

The manifest is safe to generate against a GGUF in place. Optional extraction
copies only a user-selected set of (layer, expert) slices into one pack file.
It does not alter the source model.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402


EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(down|gate|up|gate_up)_exps\.weight$")


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    # A full hash is deliberately opt-in: it reads the whole multi-GB GGUF.
    return {
        "file_name": path.name,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def full_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(model_path: Path, include_hash: bool) -> dict:
    reader = GGUFReader(model_path, "r")
    tensors = []
    total_expert_bytes = 0

    for tensor in reader.tensors:
        match = EXPERT_RE.match(tensor.name)
        if not match:
            continue
        layer = int(match.group(1))
        role = match.group(2)
        n_experts = int(tensor.shape[2])
        if tensor.n_bytes % n_experts:
            raise ValueError(f"{tensor.name}: expert axis does not divide raw byte count")
        bytes_per_expert = tensor.n_bytes // n_experts
        tensors.append({
            "tensor": tensor.name,
            "layer": layer,
            "role": role,
            "quantization": tensor.tensor_type.name,
            "shape": [int(value) for value in tensor.shape],
            "source_offset": int(tensor.data_offset),
            "bytes": int(tensor.n_bytes),
            "expert_count": n_experts,
            "bytes_per_expert": int(bytes_per_expert),
        })
        total_expert_bytes += int(tensor.n_bytes)

    if not tensors:
        raise ValueError("no routed expert tensors matched the expected GGUF names")

    counts = {entry["expert_count"] for entry in tensors}
    if len(counts) != 1:
        raise ValueError(f"mixed expert counts are not supported by this vault format: {counts}")

    model = fingerprint(model_path)
    if include_hash:
        model["sha256"] = full_sha256(model_path)

    return {
        "format": "evm-expert-vault-manifest-v1",
        "model": model,
        "expert_count": counts.pop(),
        "layer_count": len({entry["layer"] for entry in tensors}),
        "layer_ids": sorted({entry["layer"] for entry in tensors}),
        "expert_tensor_count": len(tensors),
        "total_expert_bytes": total_expert_bytes,
        "tensors": sorted(tensors, key=lambda entry: (entry["layer"], entry["role"])),
    }


def load_selection(path: Path, layer_ids: list[int], expert_count: int) -> dict[int, list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selected = raw.get("selected", raw)
    normalized = {}
    valid_layers = set(layer_ids)
    for raw_layer, raw_experts in selected.items():
        layer = int(raw_layer)
        experts = sorted({int(expert) for expert in raw_experts})
        if layer not in valid_layers:
            raise ValueError(f"selection has invalid layer {layer}")
        if any(expert < 0 or expert >= expert_count for expert in experts):
            raise ValueError(f"selection has invalid expert at layer {layer}")
        normalized[layer] = experts
    return normalized


def extract_pack(model_path: Path, manifest: dict, selection: dict[int, list[int]], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / "experts.pack"
    index = []

    with model_path.open("rb") as source, pack_path.open("wb") as destination:
        for tensor in manifest["tensors"]:
            for expert in selection.get(tensor["layer"], []):
                source_offset = tensor["source_offset"] + expert * tensor["bytes_per_expert"]
                pack_offset = destination.tell()
                source.seek(source_offset)
                remaining = tensor["bytes_per_expert"]
                digest = hashlib.sha256()
                while remaining:
                    chunk = source.read(min(16 * 1024 * 1024, remaining))
                    if not chunk:
                        raise IOError(f"short read from {tensor['tensor']} expert {expert}")
                    destination.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                index.append({
                    "layer": tensor["layer"],
                    "expert": expert,
                    "role": tensor["role"],
                    "tensor": tensor["tensor"],
                    "pack_offset": pack_offset,
                    "bytes": tensor["bytes_per_expert"],
                    "quantization": tensor["quantization"],
                    "sha256": digest.hexdigest(),
                })

    pack_manifest = {
        "format": "evm-expert-pack-v1",
        "source_model": manifest["model"],
        "selection": {str(layer): experts for layer, experts in sorted(selection.items())},
        "pack_file": pack_path.name,
        "pack_bytes": pack_path.stat().st_size,
        "slices": index,
    }
    (output_dir / "experts.pack.json").write_text(json.dumps(pack_manifest, indent=2), encoding="utf-8")
    write_pack_index(model_path, manifest, pack_manifest, output_dir)
    return pack_manifest


def write_pack_index(model_path: Path, manifest: dict, pack_manifest: dict, output_dir: Path) -> None:
    index_path = output_dir / "experts.pack.idx"
    with index_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# tensor\texpert\tpack_offset\tbytes\tggml_type\td0\td1\td2\td3\n")
        tensor_meta = {entry["tensor"]: entry for entry in manifest["tensors"]}
        reader = GGUFReader(model_path, "r")
        tensor_types = {tensor.name: int(tensor.tensor_type) for tensor in reader.tensors}
        for entry in pack_manifest["slices"]:
            meta = tensor_meta[entry["tensor"]]
            shape = [*meta["shape"], 1, 1, 1, 1][:4]
            handle.write(
                f"{entry['tensor']}\t{entry['expert']}\t{entry['pack_offset']}\t{entry['bytes']}\t"
                f"{tensor_types[entry['tensor']]}\t"
                f"{shape[0]}\t{shape[1]}\t{shape[2]}\t{shape[3]}\n"
            )


def build_spine(model_path: Path, output_path: Path) -> dict:
    """Write a GGUF containing metadata and every non-routed tensor only."""
    reader = GGUFReader(model_path, "r")
    architecture = reader.get_field("general.architecture")
    if architecture is None:
        raise ValueError("source GGUF has no general.architecture")

    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    writer = GGUFWriter(partial, arch=architecture.contents(), endianess=reader.endianess)
    alignment = reader.get_field("general.alignment")
    if alignment is not None:
        writer.data_alignment = int(alignment.contents())

    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        value = field.contents()
        if value is None:
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, value, value_type, sub_type=sub_type)

    writer.add_string("evm.expert_vault.format", "evm-expert-vault-manifest-v1")
    writer.add_string("evm.expert_vault.source_model", model_path.name)

    retained = []
    skipped_bytes = 0
    for tensor in reader.tensors:
        if EXPERT_RE.match(tensor.name):
            skipped_bytes += tensor.n_bytes
            continue
        retained.append(tensor)
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.n_bytes,
            tensor.tensor_type,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in retained:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    partial.replace(output_path)

    return {
        "spine_file": output_path.name,
        "spine_bytes": output_path.stat().st_size,
        "retained_tensor_count": len(retained),
        "external_expert_bytes": skipped_bytes,
    }


def verify_pack(output_dir: Path) -> bool:
    pack_manifest = json.loads((output_dir / "experts.pack.json").read_text(encoding="utf-8"))
    with (output_dir / pack_manifest["pack_file"]).open("rb") as handle:
        for entry in pack_manifest["slices"]:
            handle.seek(entry["pack_offset"])
            remaining = entry["bytes"]
            digest = hashlib.sha256()
            while remaining:
                chunk = handle.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    return False
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != entry["sha256"]:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Index or extract an EVM expert vault from a GGUF model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha256", action="store_true", help="read the full source model to add an integrity hash")
    parser.add_argument("--selection", type=Path, help="JSON map of layer IDs to expert IDs; enables pack extraction")
    parser.add_argument("--uniform-expert", type=int, help="extract one expert ID from every layer; intended for a vault smoke test")
    parser.add_argument("--all-experts", action="store_true", help="extract every expert into a complete external vault")
    parser.add_argument("--spine", type=Path, help="write a non-expert spine GGUF")
    parser.add_argument("--write-index", action="store_true", help="regenerate the compact index for an existing pack")
    parser.add_argument("--verify", action="store_true", help="hash-check the emitted pack after extraction")
    args = parser.parse_args()

    manifest = build_manifest(args.model, args.sha256)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "expert-vault-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    selection_flags = sum(value is not None for value in [args.selection, args.uniform_expert]) + int(args.all_experts)
    if selection_flags > 1:
        parser.error("use only one of --selection, --uniform-expert, or --all-experts")
    selection = None
    if args.selection:
        selection = load_selection(args.selection, manifest["layer_ids"], manifest["expert_count"])
    elif args.uniform_expert is not None:
        if not 0 <= args.uniform_expert < manifest["expert_count"]:
            parser.error("--uniform-expert is outside the model expert range")
        selection = {layer: [args.uniform_expert] for layer in manifest["layer_ids"]}
    elif args.all_experts:
        selection = {layer: list(range(manifest["expert_count"])) for layer in manifest["layer_ids"]}

    if args.write_index and selection:
        parser.error("--write-index does not accept an extraction selection")
    if args.write_index:
        existing_pack = json.loads((args.out / "experts.pack.json").read_text(encoding="utf-8"))
        write_pack_index(args.model, manifest, existing_pack, args.out)
        pack = existing_pack
    else:
        pack = extract_pack(args.model, manifest, selection, args.out) if selection else None
    verified = verify_pack(args.out) if pack and args.verify else None
    spine = build_spine(args.model, args.spine) if args.spine else None
    print(f"Vault manifest: {manifest['layer_count']} layers | {manifest['expert_count']} experts | {manifest['total_expert_bytes'] / 1024**3:.2f} GB")
    print(f"Pack: {pack['pack_bytes'] / 1024**3:.2f} GB" if pack else "Pack: not requested")
    print(f"Verification: {'PASS' if verified else 'not requested'}")
    print(f"Spine: {spine['spine_bytes'] / 1024**3:.2f} GB" if spine else "Spine: not requested")
    if verified is False:
        print("Status: FAIL")
        raise SystemExit(1)
    print("Status: PASS")


if __name__ == "__main__":
    main()
