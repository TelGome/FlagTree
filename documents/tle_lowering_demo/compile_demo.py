# Copyright 2026-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""End-to-end compile demo: emit TTIR / ttadapter / npubin dumps for a TLE
distributed kernel on Ascend.

Usage:
    TRITON_ALWAYS_COMPILE=1 TRITON_DEBUG=1 python compile_demo.py
    TRITON_ALWAYS_COMPILE=1 TRITON_DEBUG=1 python compile_demo.py --fix-core-type

The ``distributed.core_type`` attribute is now emitted by the production
frontend (``dsa/ascend/distributed.py::_dispatch``), so a plain run completes
the whole pipeline. ``--fix-core-type`` keeps the old in-process monkeypatch as
a regression comparison; it is idempotent and only matters when reproducing the
pre-fix failure:

    Dialect `distributed' not found for custom op 'distributed.extern_call'
"""

import argparse
import glob
import os
import sys


def patch_core_type():
    """Deprecated no-op kept for command-line compatibility.

    The frontend now emits ``distributed.core_type`` itself
    (``dsa/ascend/distributed.py::_dispatch``), and it deliberately picks
    ``vector`` rather than ``cube_and_vector``: the latter turns the kernel into
    a mix kernel whose body is cloned into AIC and AIV, so every distributed op
    would run twice per PE.
    """
    print("note: --fix-core-type is a no-op; the frontend sets "
          "distributed.core_type=vector itself")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-core-type", action="store_true",
                        help="attach distributed.core_type to the barrier extern_call")
    args = parser.parse_args()

    os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
    os.environ.setdefault("TRITON_DEBUG", "1")

    import triton
    import torch
    import torch_npu  # noqa: F401

    if args.fix_core_type:
        patch_core_type()

    from tle_dist_kernel import ring_exchange

    print("triton      :", triton.__file__)
    print("version     :", triton.__version__)
    print("core-type fix:", "ON" if args.fix_core_type else "OFF")

    x = torch.zeros(1, device="npu")
    out = torch.zeros(1, device="npu")
    kernel = ring_exchange.warmup(x, out, grid=(1,))

    print("asm keys    :", sorted(kernel.asm.keys()))
    ttadapter = kernel.asm.get("ttadapter", "")
    if isinstance(ttadapter, (bytes, bytearray)):
        ttadapter = ttadapter.decode("utf-8", "ignore")
    for line in ttadapter.splitlines():
        if "hivm.hir.custom" in line:
            print("  " + line.strip())

    dumps = sorted(glob.glob(os.path.expanduser("~/.triton/dump/*/")),
                   key=os.path.getmtime)
    if dumps:
        print("dump dir    :", dumps[-1])
        print("  kernel.ttir.mlir / kernel.ttadapter.mlir / kernel.mlir /"
              " kernel.npuir.mlir / kernel.o")
    print("OK: compile finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
