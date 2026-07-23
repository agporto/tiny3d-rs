# tiny3d-rs Python package
# Copyright (c) 2026 Arthur Porto
# Derived from the MIT-licensed tiny3D/Open3D API lineage.
# See LICENSE and THIRD_PARTY_NOTICES.md in the distribution.
# SPDX-License-Identifier: MIT

import os
import sys
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TCM_ENABLE"] = "1"
from ctypes import CDLL
from ctypes.util import find_library
from pathlib import Path
import warnings
from tiny3d._build_config import _build_config

if sys.platform == "win32":
    _win32_dll_dir = os.add_dll_directory(str(Path(__file__).parent))

__DEVICE_API__ = "cpu"

if __DEVICE_API__ == "cpu":
    from tiny3d.cpu.pybind import (
        geometry,
        io,
        pipelines,
        utility,
    )
    from tiny3d.cpu import pybind

def _insert_pybind_names(skip_names=()):
    submodules = {}
    for modname in sys.modules:
        if "tiny3d." + __DEVICE_API__ + ".pybind" in modname:
            if any("." + skip_name in modname for skip_name in skip_names):
                continue
            subname = modname.replace(__DEVICE_API__ + ".pybind.", "")
            if subname not in sys.modules:
                submodules[subname] = sys.modules[modname]
    sys.modules.update(submodules)

_insert_pybind_names()

__version__ = "2.0.0"

if int(sys.version_info[0]) < 3:
    raise Exception("tiny3d only supports Python 3.")

if sys.platform == "win32":
    _win32_dll_dir.close()
del os, sys, CDLL, find_library, Path, warnings, _insert_pybind_names
