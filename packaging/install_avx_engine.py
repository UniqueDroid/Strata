#!/usr/bin/env python3
"""Install this fork's Windows AVX engine for `strata launch` and the app, which take their engine from the component folder.

    python packaging/install_avx_engine.py                     # the fork's latest release, into %LOCALAPPDATA%\\Strata\\components
    python packaging/install_avx_engine.py --zip FILE          # a zip you already have, not checked against a release

It downloads strata-windows-x64.zip and compares it with the digest the release publishes, gets cuBLAS from NVIDIA's pip packages
(setup.py's CUDA_WHEELS), makes a package with package_engine.py and pins it (.updates/state.json engine.pinned). The pin matters:
an upstream engine needs AVX2, and without the pin a newer one in the same folder would be chosen instead.
Undo: `--unpin`, which leaves the package in place.
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
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging"))

import package_engine  # noqa: E402
from strata_control import engine_component as ec  # noqa: E402

REPO = "UniqueDroid/Strata"
ASSET = "strata-windows-x64.zip"
VARIANT = "cuda13avx"      # its own folder name: an upstream cuda13 package of the same version stays apart
MIN_DRIVER = "580.65"
CUBLAS = "cublas64_13.dll"


def archs_from_build(meta: dict) -> list[str]:
    """package_engine's arch list from BUILD.json: the real architectures, and PTX of the newest one when the build has it."""
    numbers = [int(n) for n in meta.get("archs") or []]
    if not numbers:
        raise SystemExit("BUILD.json lists no architectures")
    return [f"sm_{n}" for n in numbers] + ([f"compute_{max(numbers)}"] if meta.get("ptx") else [])


def check_is_avx(meta: dict) -> None:
    if meta.get("isa_floor") != "avx":
        raise SystemExit(f"this zip is not the AVX build (isa_floor: {meta.get('isa_floor')!r})")


def set_pin(root: Path, folder: str | None) -> None:
    """engine.pinned in .updates/state.json, keeping everything else there."""
    path = root / ec.UPDATE_STATE
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    engine = data.get("engine") if isinstance(data.get("engine"), dict) else {}
    if folder:
        engine["pinned"] = folder
    else:
        engine.pop("pinned", None)
    data["engine"] = engine
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(temp, path)


def wheel_pins() -> list[str]:
    """setup.py's CUDA_WHEELS, read rather than imported: importing setup.py runs its start-up checks."""
    text = (ROOT / "setup.py").read_text(encoding="utf-8")
    found = re.search(r"^CUDA_WHEELS = \[(.*?)\]", text, re.M)
    pins = re.findall(r'"([^"]+)"', found.group(1)) if found else []
    if not pins:
        raise SystemExit("CUDA_WHEELS not found in setup.py")
    return pins


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_release(work: Path) -> Path:
    request = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/latest", headers={"User-Agent": "strata-install-avx"})
    with urllib.request.urlopen(request, timeout=60) as reply:
        release = json.load(reply)
    asset = next((x for x in release.get("assets") or [] if x.get("name") == ASSET), None)
    if not asset or not str(asset.get("digest") or "").startswith("sha256:"):
        raise SystemExit(f"release {release.get('tag_name')} of {REPO} has no {ASSET} with a digest")
    target = work / ASSET
    print(f"Downloading {ASSET} from {release.get('tag_name')} ({asset.get('size', 0) >> 20} MiB) ...", flush=True)
    urllib.request.urlretrieve(asset["browser_download_url"], target)
    if "sha256:" + sha256(target) != asset["digest"]:
        raise SystemExit(f"{ASSET}: the download does not match the digest of the release")
    return target


def find_cublas(work: Path, given: Path | None) -> Path:
    """The folder with cublas64_13.dll: the one named, this Python's own pip packages, else NVIDIA's wheels unpacked into `work`."""
    places = [given] if given else [Path(p) / "nvidia" / "cu13" / "bin" / "x86_64" for p in sys.path if p]
    for place in places:
        if (place / CUBLAS).is_file():
            return place
    if given:
        raise SystemExit(f"no {CUBLAS} in {given}")
    pins = wheel_pins()
    print(f"Getting cuBLAS from pip ({', '.join(pins)}; about 0.4 GB) ...", flush=True)
    target = work / "wheels"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", str(target), *pins], check=True)
    place = target / "nvidia" / "cu13" / "bin" / "x86_64"
    if not (place / CUBLAS).is_file():
        raise SystemExit(f"the wheels have no {CUBLAS}")
    return place


def install(zip_path: Path, components: Path, cublas: Path, platform: str, work: Path) -> Path:
    stage = work / "stage"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(stage)
    meta = json.loads((stage / "BUILD.json").read_text(encoding="utf-8"))
    check_is_avx(meta)
    out = components / "engine"
    folder = out / f"{meta['version']}-{platform}-cuda-{VARIANT}"
    if folder.exists():
        shutil.rmtree(folder)
    args = ["--src", str(stage), "--backend", "cuda", "--version", str(meta["version"]), "--variant", VARIANT, "--out", str(out),
            "--platform", platform, "--min-driver", f"nvidia={MIN_DRIVER}", "--build-id", str(meta.get("build_id") or ""),
            "--search", str(cublas), "--notes", f"fork {REPO}: the AVX build"]
    for arch in archs_from_build(meta):
        args += ["--arch", arch]
    if package_engine.main(args) != 0:
        raise SystemExit("packaging failed")
    if not folder.is_dir():
        raise SystemExit(f"expected the package in {folder}")
    return folder


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Install this fork's Windows AVX engine for `strata launch` and the app.")
    ap.add_argument("--zip", type=Path, help=f"a {ASSET} you already have (not checked against a release)")
    ap.add_argument("--cublas-dir", type=Path, help=f"a folder with {CUBLAS} and cublasLt64_13.dll")
    ap.add_argument("--components-dir", type=Path, help="default: %%LOCALAPPDATA%%\\Strata\\components")
    ap.add_argument("--platform", default="windows-x86_64", help=argparse.SUPPRESS)      # tests on another system
    ap.add_argument("--unpin", action="store_true", help="choose the engine by version again; the package stays")
    a = ap.parse_args(argv)
    components = a.components_dir or ec.components_dir()
    if a.unpin:
        set_pin(components, None)
        print("The pin is gone: Strata picks the newest engine that fits this PC again. Upstream's engines need AVX2.")
        return 0
    if os.name != "nt" and a.platform == "windows-x86_64" and not a.components_dir:
        raise SystemExit("this installs the Windows engine: run it on the Windows PC")
    with tempfile.TemporaryDirectory(prefix="strata-avx-") as temp:
        work = Path(temp)
        zip_path = a.zip or download_release(work)
        folder = install(zip_path, components, find_cublas(work, a.cublas_dir), a.platform, work)
    set_pin(components, folder.name)
    print(f"Installed and pinned: {folder}")
    print("Next: strata launch   (with a model you have: strata launch --model-path FOLDER)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
