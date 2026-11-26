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

"""IR-level lowering demo: AST -> TTIR -> distributed->hivm (no NPU run).

Usage:
    python ir_demo.py

Prints:
    1. TTIR before conversion (only distributed.*, no tle.*)
    2. the collected distributed ops
    3. IR after convert-triton-distributed-to-hivm
    4. whether any distributed op was left unconverted
"""

import os
import sys

from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend

from tle_dist_kernel import ring_exchange


def make_context():
    ctx = libtriton.ir.context()
    if hasattr(ctx, "disable_multithreading"):
        ctx.disable_multithreading()
    libtriton.ir.load_dialects(ctx)
    libtriton.distributed.ir.load_dialects(ctx)
    libtriton.ascend.ir.load_dialects(ctx)
    libtriton.ascend.load_dialects(ctx)
    try:
        from triton._C.libtriton import buffer_ir
        buffer_ir.load_dialects(ctx)
    except Exception:
        pass
    return ctx


def main():
    ctx = make_context()
    target = GPUTarget("npu", "Ascend910B1", 0)
    backend = make_backend(target)
    options = backend.parse_options({})
    src = ASTSource(ring_exchange, {"x_ptr": "*fp32", "out_ptr": "*fp32"}, {})
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options),
                      backend.get_module_map(), ctx)

    print("===== 1. TTIR before conversion =====")
    print(str(mod))

    ops = []
    mod.walk(lambda op: ops.append(op.get_name())
             if op.get_name().startswith(("distributed.", "tle.")) else None)
    print("===== 2. distributed / tle ops =====")
    print(ops)

    pm = libtriton.ir.pass_manager(ctx)
    libtriton.distributed.ascend_passes.ttgpuir.add_convert_triton_distributed_to_hivm(pm)
    os.environ["MLIR_DISABLE_MULTITHREADING"] = "1"
    pm.run(mod, "")

    print("===== 3. IR after convert-triton-distributed-to-hivm =====")
    text = str(mod)
    print(text)

    leftovers = [line.strip() for line in text.splitlines() if "distributed." in line]
    print("===== 4. left-over distributed ops =====")
    if leftovers:
        print("FAIL: %d op(s) were not converted:" % len(leftovers))
        for line in leftovers:
            print("  " + line)
        return 1
    print("OK: no distributed.* left; every op became hivm.hir.custom")
    return 0


if __name__ == "__main__":
    sys.exit(main())
