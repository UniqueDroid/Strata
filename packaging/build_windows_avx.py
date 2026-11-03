#!/usr/bin/env python3
"""The ready-made Windows engine of the fork UniqueDroid/Strata: strata-windows-x64.zip, built for CPUs with AVX and without AVX2.

    python packaging/build_windows_avx.py engine|vision|dsv41 --out <folder>     build one program into <folder>
    python packaging/build_windows_avx.py package --out <folder>                 write BUILD.json and the zip from <folder>
    python packaging/build_windows_avx.py all --out <folder>                     all of it, one after the other

Run it from an "x64 Native Tools" prompt with CMake, Ninja, the CUDA toolkit (CUDA_PATH) and the pinned llama.cpp in
third_party/llama.cpp or STRATA_GGML_DIR.  .github/workflows/windows-avx-build.yml runs it on GitHub.  Every build stops when
ggml-cpu's compile lines hold anything above AVX: on the target CPU that would be an illegal instruction in the first expert.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR = "avx"
VARIANTS = {"cuda13": {"archs": "75;86;89;120", "toolkit": "13.0", "sm60": "OFF", "asset": "strata-windows-x64.zip"},
            "cuda12": {"archs": "60;61;70;75;80;86;89", "toolkit": "12.9", "sm60": "ON", "asset": "strata-windows-x64-cuda12.zip"}}
TOO_WIDE = re.compile(r"(?:/arch:AVX(?:2|512|10)|-mavx2|-mfma|-mf16c|-mbmi2|-mavx512\w*|-march=native)(?![\w.])", re.I)
FLOOR_FLAG = re.compile(r"(?:/arch:AVX|-mavx)(?![\w.])", re.I)


def source_version() -> str:
    m = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    if not m:
        raise SystemExit("no project version in CMakeLists.txt")
    return m.group(1)


def arch_list(archs: str) -> str:
    """Every architecture as machine code, plus PTX of the newest one: the release engines' list."""
    parts = [a for a in archs.split(";") if a]
    return ";".join(f"{a}-real" for a in parts) + f";{parts[-1]}-virtual"


def check_floor(commands: list, where: str = "") -> int:
    """Every ggml-cpu compile line has the AVX flag and nothing wider.  Returns how many lines were checked."""
    lines = [c for c in commands if re.search(r"[\\/]ggml-cpu[\\/]", c.get("file", "").replace("\\\\", "/"))
             and re.search(r"\.(?:c|cc|cpp|cxx)$", c.get("file", ""), re.I)]
    if not lines:
        raise SystemExit(f"{where}: no ggml-cpu compile line found")
    for c in lines:
        cmd = c.get("command") or " ".join(c.get("arguments", []))
        wide = TOO_WIDE.search(cmd)
        if wide:
            raise SystemExit(f"{where}: {c['file']} is compiled with {wide.group(0)}")
        if not FLOOR_FLAG.search(cmd):
            raise SystemExit(f"{where}: {c['file']} is compiled without the AVX flag")
    return len(lines)


def run(cmd: list) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True, cwd=ROOT)


def verified_build(src: Path, bdir: Path, defs: list, targets: list, jobs: int) -> None:
    if (bdir / "CMakeCache.txt").exists():
        (bdir / "CMakeCache.txt").unlink()             # ggml's CPU options are cached: never reuse another floor's
    run(["cmake", "-G", "Ninja", "-S", src, "-B", bdir, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
         "-DSTRATA_PORTABLE=ON", f"-DSTRATA_ISA_FLOOR={FLOOR}", *defs])
    count = check_floor(json.loads((bdir / "compile_commands.json").read_text(encoding="utf-8")), str(bdir))
    print(f"{bdir}: {count} ggml-cpu compile lines at the AVX floor", flush=True)
    run(["cmake", "--build", bdir, "-j", jobs, "--target", *targets])


def cuda_defs(archs: str, sm60: str = "") -> list:
    cuda = os.environ.get("CUDA_PATH", "").strip().rstrip("\\/").replace("\\", "/")
    if not cuda or not Path(cuda, "bin", "nvcc.exe").is_file():
        raise SystemExit(f"CUDA_PATH must name the toolkit (nvcc.exe not found under '{cuda}/bin')")
    return ["-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl", f"-DCMAKE_CUDA_COMPILER={cuda}/bin/nvcc.exe",
            f"-DCUDAToolkit_ROOT={cuda}", f"-DCMAKE_CUDA_ARCHITECTURES={archs}"] + ([f"-DSTRATA_EXPERIMENTAL_SM60={sm60}"] if sm60 else [])


def llama_dir() -> str:
    d = Path(os.environ.get("STRATA_GGML_DIR") or ROOT / "third_party" / "llama.cpp")
    if not (d / "tools" / "mtmd").is_dir():
        raise SystemExit(f"llama.cpp not found in {d}: set STRATA_GGML_DIR")
    return d.as_posix()


def build_engine(a, v, out: Path) -> None:
    b = Path(a.build_root) / f"engine-{a.variant}"
    verified_build(ROOT, b, ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_NATIVE_EXPERTS=ON",
                             "-DSTRATA_Q6K_EXPERTS=ON", "-DSTRATA_MMQ_KQUANTS=ON", f"-DSTRATA_GGML_DIR={llama_dir()}",
                             *cuda_defs(arch_list(a.archs or v["archs"]), v["sm60"])], ["strata", "strata_kq"], a.jobs)
    for name in ("strata.exe", "strata_kq.dll"):
        shutil.copy2(b / name, out / name)


def build_vision(a, v, out: Path) -> None:
    b = Path(a.build_root) / f"vision-{a.variant}"
    verified_build(ROOT / "tools" / "vision", b, [f"-DLLAMA_DIR={llama_dir()}", "-DSTRATA_VISION_CUDA=ON", "-DGGML_CUDA_NCCL=OFF",
                                                  *cuda_defs(arch_list(a.archs or v["archs"]))], ["strata-vision"], a.jobs)
    shutil.copy2(b / "bin" / "strata-vision.exe", out / "strata-vision.exe")


def build_dsv41(a, v, out: Path) -> None:
    if a.variant != "cuda13":
        raise SystemExit("strata-dsv41 is built for cuda13 only (packaging/build_dsv41_windows.bat)")
    b = Path(a.build_root) / "dsv41"
    verified_build(ROOT, b, ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_BUILD_DSV41=ON", "-DSTRATA_NATIVE_EXPERTS=ON",
                             f"-DSTRATA_GGML_DIR={llama_dir()}", *cuda_defs("86-real;89-real;120")], ["strata-dsv41"], a.jobs)
    shutil.copy2(b / "strata-dsv41.exe", out / "strata-dsv41.exe")


def build_json(variant: str, archs: str, version: str, have_vision: bool, build_id: str) -> dict:
    v = VARIANTS[variant]
    numbers = [int(x) for x in archs.split(";") if x]
    meta = {"version": version, "source": "prebuilt", "variant": variant, "archs": numbers, "ptx": True, "cuda": v["toolkit"],
            "portable": True, "isa_floor": FLOOR, "build_id": build_id}
    return {**meta, "vision_archs": numbers} if have_vision else meta


def package(a, v, out: Path) -> Path:
    version = source_version()
    exe = out / "strata.exe"
    if not exe.is_file():
        raise SystemExit(f"no {exe}")
    if not re.search(rb"engine=" + re.escape(version.encode()) + rb"\n", exe.read_bytes()):
        raise SystemExit(f"{exe} is not engine version {version}")
    build_id = os.environ.get("GITHUB_SHA") or subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                                              text=True, check=False).stdout.strip()
    meta = build_json(a.variant, a.archs or v["archs"], version, (out / "strata-vision.exe").is_file(), build_id[:12])
    (out / "BUILD.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    z = Path(a.dist or out.parent) / v["asset"]
    z.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as f:
        for p in sorted(out.iterdir()):
            if p.is_file():
                f.write(p, p.name)
    print(f"{z}: {z.stat().st_size / 2**20:.0f} MiB, sha256 {hashlib.sha256(z.read_bytes()).hexdigest()}", flush=True)
    print(json.dumps(meta), flush=True)
    return z


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("engine", "vision", "dsv41", "package", "all"))
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="cuda13")
    ap.add_argument("--archs", default="", help="override the architecture list, for example 75;86")
    ap.add_argument("--out", required=True, help="the folder that collects the programs")
    ap.add_argument("--dist", default="", help="where the zip goes (default: beside --out)")
    ap.add_argument("--build-root", default=str(ROOT / "build-avx"))
    ap.add_argument("--jobs", type=int, default=3, help="compile jobs (the CUDA compiler uses a lot of RAM)")
    a = ap.parse_args(argv)
    v = VARIANTS[a.variant]
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.part in ("engine", "all"):
        build_engine(a, v, out)
    if a.part in ("vision", "all"):
        build_vision(a, v, out)
    if a.part == "dsv41" or (a.part == "all" and a.variant == "cuda13"):
        build_dsv41(a, v, out)
    if a.part in ("package", "all"):
        package(a, v, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
