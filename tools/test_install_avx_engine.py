#!/usr/bin/env python3
"""packaging/install_avx_engine.py: the arch list, the refusal of a non-AVX zip, the pin and the wheel pins."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging"))

import install_avx_engine as I  # noqa: E402
from strata_control import engine_component as ec  # noqa: E402


class InstallAvxEngine(unittest.TestCase):
    def test_archs(self):
        self.assertEqual(I.archs_from_build({"archs": [75, 86, 89, 120], "ptx": True}), ["sm_75", "sm_86", "sm_89", "sm_120", "compute_120"])
        self.assertEqual(I.archs_from_build({"archs": [86]}), ["sm_86"])
        with self.assertRaises(SystemExit):
            I.archs_from_build({})

    def test_only_the_avx_build(self):
        I.check_is_avx({"isa_floor": "avx"})
        for meta in ({}, {"isa_floor": ""}, {"isa_floor": "none"}):
            with self.assertRaises(SystemExit):
                I.check_is_avx(meta)

    def test_pin_keeps_the_rest_of_the_state(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            state = root / ec.UPDATE_STATE
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"other": 1, "engine": {"previous": "x"}}), encoding="utf-8")
            I.set_pin(root, "pkg")
            self.assertEqual(ec.pinned_folder(root), "pkg")
            data = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual((data["other"], data["engine"]["previous"]), (1, "x"))
            I.set_pin(root, None)
            self.assertIsNone(ec.pinned_folder(root))
            self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["engine"], {"previous": "x"})

    def test_pin_on_a_new_folder(self):
        with tempfile.TemporaryDirectory() as t:
            I.set_pin(Path(t), "pkg")
            self.assertEqual(ec.pinned_folder(Path(t)), "pkg")

    def test_wheel_pins_come_from_setup(self):
        self.assertTrue(any(p.startswith("nvidia-cublas==") for p in I.wheel_pins()))


if __name__ == "__main__":
    unittest.main()
