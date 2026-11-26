# Copyright 2026-     FlagOS Contributors
# SPDX-License-Identifier: MIT
"""IR-level regression tests for the Ascend TLE distributed primitives.

These tests stop at IR: they build TTIR for a kernel and run
``convert-triton-distributed-to-hivm`` in-process, so they need neither a
multi-rank launch nor an NPU in the test process. Each kernel must satisfy:

1. TTIR contains only ``distributed.*`` (no ``tle.*`` leftover),
2. every ``distributed.extern_call`` carries ``distributed.core_type``,
3. the conversion pass leaves no ``distributed.*`` op behind.
"""

import os

import pytest

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._flagtree_backend import FLAGTREE_BACKEND

pytestmark = pytest.mark.skipif(FLAGTREE_BACKEND != "ascend", reason="Ascend-only distributed lowering")


def _lower(kernel, signature):
    """Build TTIR, run the distributed->HIVM pass, return (ttir, lowered, ops)."""
    from triton._C import libtriton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend

    if not hasattr(libtriton, "distributed"):
        pytest.skip("build without TRITON_BUILD_TD")

    ctx = libtriton.ir.context()
    libtriton.ir.load_dialects(ctx)
    libtriton.distributed.ir.load_dialects(ctx)
    libtriton.ascend.ir.load_dialects(ctx)
    libtriton.ascend.load_dialects(ctx)

    target = GPUTarget("npu", "Ascend910B1", 0)
    backend = make_backend(target)
    options = backend.parse_options({})
    src = ASTSource(kernel, signature, {})
    module = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx)

    ops = []
    module.walk(lambda op: ops.append(op.get_name()) if op.get_name().startswith("distributed.") else None)
    ttir = str(module)

    pm = libtriton.ir.pass_manager(ctx)
    libtriton.distributed.ascend_passes.ttgpuir.add_convert_triton_distributed_to_hivm(pm)
    os.environ["MLIR_DISABLE_MULTITHREADING"] = "1"
    pm.run(module, "")

    lowered = []
    module.walk(lambda op: lowered.append(op.get_name()) if op.get_name().startswith("distributed.") else None)
    return ttir, str(module), ops, lowered


@triton.jit
def _ring_exchange(x_ptr, out_ptr):
    value = tl.load(x_ptr)
    rank = tle.shard_id(axis=-1)
    tle.distributed_barrier()
    peer = (rank + 1) % 8
    remote_ptr = tle.remote(x_ptr, peer)
    tl.store(remote_ptr, value)
    tl.store(out_ptr, value)


@triton.jit
def _signal_kernel(sig_ptr, x_ptr):
    rank = tle.shard_id(axis=-1)
    peer = (rank + 1) % tle.n_pes()
    tle.signal(sig_ptr, peer, slot_id=1, op="inc")
    tle.signal_wait(sig_ptr, slot_id=1, wait_kind="signal", target=1)
    tl.store(x_ptr, tl.load(x_ptr))


@triton.jit
def _barrier_inter_node(x_ptr):
    tle.distributed_barrier(space="inter_node")
    tl.store(x_ptr, tl.load(x_ptr))


@triton.jit
def _barrier_arrive(x_ptr):
    tle.distributed_barrier(barrier_kind="arrive")
    tl.store(x_ptr, tl.load(x_ptr))


@triton.jit
def _wait_shadow(sig_ptr, x_ptr):
    tle.signal_wait(sig_ptr, slot_id=0, wait_kind="shadow", target=1)
    tl.store(x_ptr, tl.load(x_ptr))


@triton.jit
def _signal_bad_op(sig_ptr, x_ptr):
    tle.signal(sig_ptr, 0, slot_id=0, op="set")
    tl.store(x_ptr, tl.load(x_ptr))


def test_ring_exchange_lowers_without_distributed_leftovers():
    ttir, _, ops, lowered = _lower(_ring_exchange, {"x_ptr": "*fp32", "out_ptr": "*fp32"})

    assert set(ops) == {"distributed.get_rank", "distributed.extern_call", "distributed.symm_at"}
    # The barrier must advertise which cores execute it, otherwise the
    # conversion pass fails and the later bytecode stage chokes on the dialect.
    # It is VECTOR on purpose: ``cube_and_vector`` turns the kernel into a mix
    # kernel whose body is cloned into AIC and AIV, so every distributed op
    # would execute twice per PE.
    assert 'distributed.core_type = "vector"' in ttir
    assert "cube_and_vector" not in ttir
    assert lowered == []


def test_signal_and_n_pes_lower_to_aclshmem_symbols():
    ttir, lowered_ir, ops, lowered = _lower(_signal_kernel, {"sig_ptr": "*i32", "x_ptr": "*fp32"})

    assert "distributed.get_num_ranks" in ops
    assert "distributed.notify" in ops
    assert lowered == []
    assert "aclshmem_n_pes" in lowered_ir
    assert "aclshmemx_signal_op" in lowered_ir
    # ``signal_wait`` must stay a side-effecting extern call: a ``distributed.wait``
    # whose token is never consumed is dropped by the greedy rewriter.
    assert "aclshmem_wait_int32" in lowered_ir
    # 64B per slot / 4B per i32 element -> 16 elements of address arithmetic.
    assert "arith.constant 16 : i32" in ttir


@pytest.mark.parametrize("kernel", [_barrier_inter_node, _barrier_arrive, _wait_shadow, _signal_bad_op])
def test_unsupported_forms_fail_instead_of_degrading(kernel):
    from triton.compiler.errors import CompilationError

    signature = {"sig_ptr": "*i32", "x_ptr": "*fp32"} if "sig_ptr" in kernel.arg_names else {"x_ptr": "*fp32"}
    with pytest.raises(CompilationError) as info:
        _lower(kernel, signature)
    # The code generator wraps frontend errors; make sure the failure is the
    # explicit "Ascend cannot express this" rejection, not an incidental one.
    assert isinstance(info.value.__cause__, (NotImplementedError, ValueError))
