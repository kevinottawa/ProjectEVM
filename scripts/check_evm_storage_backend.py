import argparse
import ctypes.util
import json
import os
import platform
import shutil
from pathlib import Path


def find_cufile():
    roots = [
        Path(os.environ.get("CUDA_HOME", "")),
        Path(os.environ.get("CUDA_PATH", "")),
        Path("/usr/local/cuda"),
    ]
    headers = []
    libraries = []
    for root in roots:
        if not str(root) or not root.exists():
            continue
        headers.extend(root.glob("include/cufile.h"))
        libraries.extend(root.glob("lib*/libcufile.so*"))
        libraries.extend(root.glob("lib*/cufile.lib"))
    if ctypes.util.find_library("cufile"):
        libraries.append(Path(ctypes.util.find_library("cufile")))
    return bool(headers), bool(libraries)


def main():
    parser = argparse.ArgumentParser(description="Select the available EVM cold-storage transport.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    system = platform.system()
    has_header, has_library = find_cufile()
    has_kernel_path = system == "Linux" and (
        Path("/dev/nvidia-fs0").exists()
        or Path("/proc/driver/nvidia-fs").exists()
        or shutil.which("gdscheck") is not None
    )
    gds_ready = system == "Linux" and has_header and has_library and has_kernel_path
    result = {
        "platform": system,
        "gds_ready": gds_ready,
        "cufile_header": has_header,
        "cufile_library": has_library,
        "kernel_storage_path": has_kernel_path,
        "selected_backend": "bounded_mmap",
        "gds_integration_status": "candidate_host_detected" if gds_ready else "requirements_missing",
        "recommended_env": {
            "EVM_DISK_BACKING": "1",
            "EVM_DISK_TRIM": "1",
            "EVM_DISK_TRIM_INTERVAL": "1",
        },
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Backend: {result['selected_backend']}")
        print(f"GDS ready: {'yes' if gds_ready else 'no'}")
        print("Status: PASS")


if __name__ == "__main__":
    main()
