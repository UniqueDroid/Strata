"""Tests for packaging/build_windows_avx.py: the compile-line check and the BUILD.json setup.py reads. No compiler, no GPU.

    python -m unittest tools.test_build_windows_avx
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))
sys.path.insert(0, str(ROOT))
import build_windows_avx as b  # noqa: E402
import setup  # noqa: E402

GGML = "C:/l/ggml/src/ggml-cpu/ops.cpp"


class CheckFloor(unittest.TestCase):
    def test_avx_line_passes(self):
        self.assertEqual(b.check_floor([{"file": GGML, "command": "cl /arch:AVX /O2"}]), 1)

    def test_wider_or_missing_flag_stops(self):
        for cmd in ("cl /arch:AVX2", "cl /arch:AVX512", "cl /O2", "g++ -mavx -mfma", "g++ -march=native"):
            with self.subTest(cmd), self.assertRaises(SystemExit):
                b.check_floor([{"file": GGML, "command": cmd}])

    def test_other_files_and_no_lines(self):
        ok = [{"file": GGML, "command": "cl /arch:AVX"}, {"file": "C:/s/src/kernels/cpu/iq_avx2.cpp", "command": "cl /arch:AVX2"}]
        self.assertEqual(b.check_floor(ok), 1)              # Strata's own kernels are chosen at run time
        with self.assertRaises(SystemExit):
            b.check_floor(ok[1:])


class BuildJson(unittest.TestCase):
    def test_setup_reads_it(self):
        m = b.build_json("cuda13", "75;86;89;120", "0.1.56", True, "abc")
        self.assertEqual(m["isa_floor"], "avx")
        self.assertEqual(setup.arch_numbers(m["archs"]), [75, 86, 89, 120])
        self.assertEqual(setup.arch_numbers(m["vision_archs"]), [75, 86, 89, 120])
        self.assertNotEqual(m["source"], "local")

    def test_arch_list(self):
        self.assertEqual(b.arch_list("75;120"), "75-real;120-real;120-virtual")


if __name__ == "__main__":
    unittest.main()
