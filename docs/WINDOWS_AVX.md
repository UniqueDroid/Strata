# Windows engine for CPUs without AVX2 (fork UniqueDroid/Strata)

This fork builds every Windows engine for CPUs with AVX and without AVX2 (Sandy Bridge, Ivy Bridge, AMD Bulldozer).
It is not an official Strata release.

## What is different from upstream

- `STRATA_ISA_FLOOR` defaults to `avx` on Windows, in the root `CMakeLists.txt` and in `tools/vision/CMakeLists.txt`. Every build
  script that uses them gets it. `-DSTRATA_ISA_FLOOR=` gives the AVX2 build back.
- `STRATA_ISA_FLOOR` and `STRATA_PORTABLE` now work together: a static MSVC runtime, and `ggml-cpu` stops at AVX.
- `setup.py` on Windows downloads `strata-windows-x64.zip` from this fork's release and accepts only an engine whose `BUILD.json` says
  `"isa_floor": "avx"`. It does not fall back to the AVX2 engines of the signed feed. With no matching zip it compiles the engine on the PC.
- `.github/workflows/windows-avx-build.yml` builds `strata.exe`, `strata_kq.dll`, `strata-vision.exe` and `strata-dsv41.exe` with CUDA 13.0 for
  `sm_75`, `sm_86`, `sm_89` and `sm_120`, and publishes the zip as the release of this version. It runs on a push to `main` that touches the
  engine, and by hand under Actions > Windows AVX engine > Run workflow.
- `packaging/build_windows_avx.py` is the build script. It stops when a `ggml-cpu` compile line has anything wider than AVX.

## Using it

    git clone https://github.com/UniqueDroid/Strata
    cd Strata
    START-HERE.bat

`UPDATE.bat` keeps it current from this fork.

## Limits

- The desktop app and `strata update` use upstream's signed feed, whose engines need AVX2. Use `START-HERE.bat` on these CPUs.
- NVIDIA cards only for the ready-made zip. A GTX 10 or older card needs the CUDA 12 engine, which setup compiles on the PC.
- AMD on Windows on a CPU without AVX2 stays refused by setup, as upstream. `tools\hip\build_windows.bat` builds with the floor.
- The CPU's experts run on `ggml-cpu`'s kernels, which upstream describes as a few times slower than on an AVX2 CPU.
