# Copyright 2026- Xcoresigma Technology Co., Ltd

import triton.language.core as tl
import triton.language as dl
import triton
from triton.language.core import builtin, tensor
from typing import List

_SHMEM_DTYPE_MAP = {
    tl.float16: "half",
    tl.float32: "float",
    tl.bfloat16: "bfloat16",
    tl.int8: "int8_t",
    tl.int16: "int16_t",
    tl.int32: "int32_t",
    tl.int64: "int64_t",
    tl.uint8: "uint8_t",
    tl.uint16: "uint16_t",
    tl.uint32: "uint32_t",
    tl.uint64: "uint64_t",
}


def _shmem_dtype(dtype):
    if hasattr(dtype, 'element_ty'):
        return _SHMEM_DTYPE_MAP.get(dtype.element_ty, str(dtype.element_ty))
    return _SHMEM_DTYPE_MAP.get(dtype, str(dtype))


def _scalar_dtype(t):
    d = t.dtype if hasattr(t, 'dtype') else t
    return d.element_ty if hasattr(d, 'element_ty') else d


# ``convert-triton-distributed-to-hivm`` rejects any distributed op that does
# not carry a ``distributed.core_type`` attribute; the value decides whether the
# resulting ``hivm.hir.custom`` lands in the CUBE function, the VECTOR function,
# or both.
#
# Everything TLE emits here is VECTOR. Marking the collective barrier as
# ``cube_and_vector`` (what TDA's ``libaclshmem_device.barrier_all`` does) makes
# BiShengIR compile the whole kernel in "mix" mode, and in mix mode the body is
# cloned into both the AIC and the AIV function. Every distributed op then runs
# twice per PE: measured on Ascend950DT, one ``signal`` delivered two increments
# and ``signal_wait`` (wait-until-equal) hung. TDA tolerates that because its
# kernels program core roles explicitly; TLE's distributed subset does not, so a
# vector-only barrier is required for single issuance.
#
# Consequence: on Ascend the barrier synchronises the AIV side of every PE. A
# kernel that also has cube work must synchronise its AIC part separately
# (``tle.dsa`` cube sync); mixing cube compute with TLE distributed ops is not
# supported yet.
_CORE_TYPE_BY_SYMBOL = {
    "aclshmem_barrier": "vector",
    "aclshmem_barrier_all": "vector",
    "aclshmem_sync": "vector",
    "aclshmem_sync_all": "vector",
    "aclshmemx_barrier_vec": "vector",
    "aclshmemx_barrier_all_vec": "vector",
    "aclshmemx_sync_vec": "vector",
    "aclshmemx_sync_vec_all": "vector",
}
# Default to VECTOR for the same reason: a ``cube_or_vector`` fallback also
# pushes the kernel into mix mode and duplicates the op.
_DEFAULT_EXTERN_CORE_TYPE = "vector"


def _mark_core_type(call, symbol: str, _semantic):
    core_type = _CORE_TYPE_BY_SYMBOL.get(symbol, _DEFAULT_EXTERN_CORE_TYPE)
    call.set_attr("distributed.core_type", _semantic.builder.get_string_attr(core_type))
    return call


def _dispatch(
    func,
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    is_pure: bool,
    _semantic=None,
):
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            dtype = arg.dtype
            if hasattr(dtype, 'element_ty'):
                dtype = dtype.element_ty
            arg_types.append(dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_types = arg_type_symbol_dict[arg_types][1]
        if not isinstance(ret_types, (List, tuple)):
            ret_types = [ret_types]

        if symbol == "":
            raise ValueError("Symbol can not be empty")
        call = func(
            lib_name,
            lib_path,
            symbol,
            arg_list,
            [ret_type.to_ir(_semantic.builder) for ret_type in ret_types],
            is_pure,
        )
        _mark_core_type(call, symbol, _semantic)

        if len(ret_types) == 0:
            return tensor(call, tl.void)
        if len(ret_types) == 1:
            return tensor(call.get_result(0), ret_types[0])
        return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(ret_types))


def _extern_call(
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    is_pure: bool,
    _semantic=None,
):
    """Dispatch an external function call. Supports ptr/block arguments."""
    dispatch_args = args.copy()
    for i in range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])

    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    func = _semantic.builder.create_extern_call
    return _dispatch(
        func,
        lib_name,
        lib_path,
        dispatch_args,
        arg_type_symbol_dict,
        is_pure,
        _semantic,
    )


class MeshConfig:
    """Lightweight mesh configuration for Ascend.

    Only ``device`` level is implemented; ``node``, ``block_cluster``,
    ``block`` accept values but trigger assert at usage time.
    """

    def __init__(self, device=None, node=None, block_cluster=None, block=None):
        self.device = device
        self.node = node
        self.block_cluster = block_cluster
        self.block = block

    def __repr__(self):
        return (f"MeshConfig(device={self.device}, node={self.node}, "
                f"block_cluster={self.block_cluster}, block={self.block})")


class device_mesh:
    """Lightweight logical view of physical device topology."""

    def __init__(self, topology: MeshConfig):
        if not isinstance(topology, MeshConfig):
            raise TypeError(f"topology must be MeshConfig, got {type(topology).__name__}")
        self._mesh_config = topology
        # Only device level is implemented
        assert topology.node is None, "TODO: node level not yet implemented"
        assert topology.block_cluster is None, "TODO: block_cluster level not yet implemented"
        assert topology.block is None, "TODO: block level not yet implemented"
        assert topology.device is not None and topology.device > 0, "device count must be > 0"

    @property
    def device_count(self) -> int:
        return self._mesh_config.device

    @property
    def shape(self) -> tuple:
        return (self.device_count, )

    def __repr__(self):
        return f"DeviceMesh(device={self.device_count})"


@builtin
def remote(
    tensor,
    shard_id=None,
    scope=None,
    space: str = None,
    dtype: tl.dtype = None,
    offset: int = None,
    _semantic=None,
):
    """Distributed pointer shuffle: remote pointer access by peer rank.

    Signature aligned with distributed.py::remote().
    Currently only ``device`` level is supported.
    """
    scope = tl._unwrap_if_constexpr(scope)
    if scope is not None:
        assert isinstance(scope, device_mesh), f"scope must be device_mesh, got {type(scope).__name__}"

    space = tl._unwrap_if_constexpr(space)
    if space is not None and not isinstance(space, str):
        raise TypeError(f"space must be str or None, got {type(space).__name__}")
    assert space is None or space == "device", \
        f"TODO: space='{space}' not yet implemented (only 'device' is supported)"

    dtype = tl._unwrap_if_constexpr(dtype)
    if dtype is not None:
        assert dtype in _SHMEM_DTYPE_MAP, \
            f"dtype {dtype} is not in the supported SHMEM type map: {list(_SHMEM_DTYPE_MAP.keys())}"

    offset = tl._unwrap_if_constexpr(offset)
    if offset is not None and not isinstance(offset, (int, tl.tensor)):
        raise TypeError(f"offset must be int, tl.tensor, or None, got {type(offset).__name__}")

    assert not tensor.type.is_block() and tensor.type.is_ptr(), "only support scalar pointer"
    rank = _semantic._convert_elem_to_ir_value(shard_id, require_i64=False)
    remote_ptr = tl.tensor(_semantic.builder.create_symm_at(tensor.handle, rank), tensor.type)

    # offset is accepted for interface compatibility but not yet applied at
    # the HIVM level — the TT addptr is lost during TT→HIVM bufferization.
    # Callers should use `remote_ptr + offset` at the call site instead.
    assert offset is None, "offset not applied at the HIVM level"
    # TODO: wire offset into symm_at / HIVM custom op

    return remote_ptr


@builtin
def shard_id(
    mesh=None,
    axis=-1,
    device_dptr=None,
    _semantic=None,
):
    """Return current shard coordinate.

    Signature aligned with distributed.py::shard_id().
    Currently only ``device`` axis is supported.
    """
    mesh = tl._unwrap_if_constexpr(mesh)
    if mesh is not None:
        assert isinstance(mesh, device_mesh), f"mesh must be device_mesh, got {type(mesh).__name__}"

    axis = tl._unwrap_if_constexpr(axis)
    if isinstance(axis, str):
        assert axis == "device", f"TODO: axis='{axis}' not yet implemented (only 'device' is supported)"
        axis = 0
        if device_dptr is not None:
            assert isinstance(device_dptr, tl.tensor) and device_dptr.dtype.is_ptr(), \
                "device_dptr must be a pointer tensor"
    elif not isinstance(axis, int):
        raise TypeError(f"axis must be int or str, got {type(axis).__name__}")
    else:
        assert axis == -1 or axis == 0, f"TODO: axis={axis} not yet implemented (only 'device'/0/-1)"

    axis = _semantic._convert_elem_to_ir_value(axis, require_i64=False)
    return tl.tensor(_semantic.builder.create_get_rank(axis), tl.int32)


@builtin
def n_pes(
    dev_mem_ptr=None,
    _semantic=None,
    ret_dtype=tl.int32,
):
    """Return the number of participants (PEs) in the current communicator.

    Ascend reads the PE count from the ACLSHMEM runtime, so ``dev_mem_ptr`` is
    accepted for signature compatibility with the portable implementation but
    is not dereferenced.
    """
    dev_mem_ptr = tl._unwrap_if_constexpr(dev_mem_ptr)
    if dev_mem_ptr is not None and not isinstance(dev_mem_ptr, tl.tensor):
        raise TypeError(f"dev_mem_ptr must be a tl.tensor or None, got {type(dev_mem_ptr).__name__}")

    axis = _semantic._convert_elem_to_ir_value(-1, require_i64=False)
    value = tl.tensor(_semantic.builder.create_get_num_ranks(axis), tl.int32)
    if ret_dtype == tl.int32:
        return value
    return tl.cast(value, ret_dtype, _semantic=_semantic)


# Ascend signal slots live in the symmetric-heap buffer that the kernel receives
# as its distributed context. Every slot owns a full 64B line so that adjacent
# slots are not fetched together with the signal they synchronise.
_SIGNAL_SLOT_STRIDE_BYTES = 64

# ``aclshmem_<type>_wait_until`` device wrappers that accept a local signal
# address plus the value to wait for.
_ACLSHMEM_WAIT_SYMBOL = {
    tl.int32: "aclshmem_wait_int32",
    tl.int64: "aclshmem_wait_int64",
}


def _dist_ir():
    try:
        from triton._C.libtriton.distributed import ir as dist_ir
    except ImportError as exc:
        raise NotImplementedError("Ascend TLE signal primitives require a FlagTree build with "
                                  "TRITON_BUILD_TD=ON") from exc
    return dist_ir


def _signal_slot_ptr(device_dptr, slot_id, _semantic):
    if not isinstance(device_dptr, tensor) or device_dptr.type.is_block() or not device_dptr.dtype.is_ptr():
        raise TypeError("device_dptr must be a scalar pointer into the ACLSHMEM signal buffer")
    elem_ty = device_dptr.dtype.element_ty
    if elem_ty not in (tl.int32, tl.int64, tl.uint64):
        raise TypeError(f"signal slots must be int32/int64/uint64, got {elem_ty}")
    elem_bits = 32 if elem_ty == tl.int32 else 64

    slot = slot_id if isinstance(slot_id, tl.tensor) else _semantic.to_tensor(slot_id)
    if not slot.dtype.is_int():
        raise TypeError(f"slot_id must be an integer, got {slot.dtype}")
    stride_elems = _SIGNAL_SLOT_STRIDE_BYTES * 8 // elem_bits
    slot_offset = _semantic.mul(_semantic.to_tensor(stride_elems), slot, sanitize_overflow=False)
    return _semantic.add(device_dptr, slot_offset, sanitize_overflow=False)


@builtin
def signal(
    device_dptr,
    peer,
    slot_id,
    value=None,
    op="inc",
    space="intra_node",
    group_kind="block",
    context_id=0,
    scope="system",
    _semantic=None,
):
    """Atomically update the sync slot of a remote peer.

    Ascend lowers this to ``distributed.notify`` -> ``aclshmemx_signal_op`` /
    ``aclshmem_atomic_add_*``. ``slot_id`` indexes the symmetric-heap signal
    buffer passed as ``device_dptr``; slots are 64B apart.

    ``space`` / ``group_kind`` / ``scope`` are accepted for API compatibility
    with the portable implementation. The ACLSHMEM data plane owns transport
    selection, so they do not change the emitted IR; ``context_id`` selects a
    pre-allocated network context and therefore only ``0`` is supported.
    """
    op_name = str(tl._unwrap_if_constexpr(op)).lower()
    if op_name not in ("inc", "add"):
        raise ValueError(f"op must be 'inc' or 'add', got {op_name!r}")

    value = tl._unwrap_if_constexpr(value)
    if op_name == "inc":
        if value is not None:
            raise ValueError("value must be omitted when op='inc'")
        signal_value = 1
    else:
        if value is None:
            raise ValueError("value is required when op='add'")
        signal_value = value

    space = tl._unwrap_if_constexpr(space)
    space = "intra_node" if space is None else str(space).lower()
    comm_scope = {
        "intra": "INTRA_NODE",
        "intra_node": "INTRA_NODE",
        "inter": "INTER_NODE",
        "inter_node": "INTER_NODE",
        "world": "INTER_NODE",
    }.get(space)
    if comm_scope is None:
        raise ValueError(f"space must be intra_node, inter_node, or world, got {space!r}")

    context_id = tl._unwrap_if_constexpr(context_id)
    if context_id is None:
        context_id = 0
    if not isinstance(context_id, int) or context_id != 0:
        raise NotImplementedError("Ascend TLE signal has no pre-allocated network contexts; "
                                  f"context_id must be 0, got {context_id!r}")

    group_kind = tl._unwrap_if_constexpr(group_kind)
    if group_kind is not None and str(group_kind).lower() not in ("thread", "warp", "block"):
        raise ValueError(f"group_kind must be thread, warp, or block, got {group_kind!r}")
    scope = tl._unwrap_if_constexpr(scope)
    if scope is not None and str(scope).lower() not in ("system", "device"):
        raise ValueError(f"scope must be 'system' or 'device', got {scope!r}")

    if not hasattr(_semantic.builder, "create_notify"):
        raise NotImplementedError("Ascend TLE signal requires a FlagTree build with TRITON_BUILD_TD=ON")

    dist_ir = _dist_ir()
    slot_ptr = _signal_slot_ptr(device_dptr, slot_id, _semantic)
    peer_t = peer if isinstance(peer, tl.tensor) else _semantic.to_tensor(peer)
    if not peer_t.dtype.is_int():
        raise TypeError(f"peer must be an integer rank, got {peer_t.dtype}")
    signal_t = signal_value if isinstance(signal_value, tl.tensor) else _semantic.to_tensor(signal_value)
    signal_t = tl.cast(signal_t, tl.int64, _semantic=_semantic)

    _semantic.builder.create_notify(slot_ptr.handle, signal_t.handle, peer_t.handle,
                                    dist_ir.SIGNAL_OP.ADD, getattr(dist_ir.COMM_SCOPE, comm_scope))
    return None


@builtin
def signal_wait(
    device_dptr,
    slot_id,
    wait_kind,
    target=None,
    group_kind="block",
    context_id=0,
    order="acquire",
    _semantic=None,
):
    """Block until the local sync slot reaches ``target``.

    Ascend lowers this to a side-effecting ``aclshmem_wait_<dtype>`` extern call,
    which spins until the slot equals ``target``. Only ``wait_kind="signal"`` is
    supported: ACLSHMEM has no runtime shadow buffer, so ``"shadow"`` and the
    counter form ``"counter"`` are rejected instead of being approximated.
    """
    wait_kind = str(tl._unwrap_if_constexpr(wait_kind)).lower()
    if wait_kind != "signal":
        raise NotImplementedError(f"Ascend TLE signal_wait only supports wait_kind='signal', got {wait_kind!r}")
    if target is None:
        raise ValueError("target is required when wait_kind='signal'")

    context_id = tl._unwrap_if_constexpr(context_id)
    if context_id is None:
        context_id = 0
    if not isinstance(context_id, int) or context_id != 0:
        raise NotImplementedError("Ascend TLE signal_wait has no pre-allocated network contexts; "
                                  f"context_id must be 0, got {context_id!r}")
    group_kind = tl._unwrap_if_constexpr(group_kind)
    if group_kind is not None and str(group_kind).lower() not in ("thread", "warp", "block"):
        raise ValueError(f"group_kind must be thread, warp, or block, got {group_kind!r}")
    order = tl._unwrap_if_constexpr(order)
    order = "acquire" if order is None else str(order).lower()
    if order != "acquire":
        raise NotImplementedError(f"Ascend TLE signal_wait only provides the acquire form, got order={order!r}")

    slot_ptr = _signal_slot_ptr(device_dptr, slot_id, _semantic)
    elem_ty = slot_ptr.dtype.element_ty
    symbol = _ACLSHMEM_WAIT_SYMBOL.get(elem_ty)
    if symbol is None:
        raise TypeError(f"signal_wait slots must be int32 or int64, got {elem_ty}")

    target_t = target if isinstance(target, tl.tensor) else _semantic.to_tensor(target)
    target_t = tl.cast(target_t, elem_ty, _semantic=_semantic)
    num_barriers = tl.cast(_semantic.to_tensor(1), tl.int64, _semantic=_semantic)

    # ``distributed.wait`` is not used here on purpose: the op only exposes its
    # ordering through the token it returns, and the greedy rewriter deletes a
    # wait whose token is never consumed by a ``consume_token``. ``tle.signal_wait``
    # returns ``None`` (same as the portable implementation), so the wait has to
    # be a side-effecting extern call to survive DCE.
    _extern_call(
        "libshmem_device",
        "",
        [slot_ptr, num_barriers, target_t],
        {
            (elem_ty, tl.int64, elem_ty): (symbol, (tl.int32,)),
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return None


# Barrier options the Ascend implementation can honour. ``aclshmem_barrier_all``
# is a convergent, full-communicator barrier: it has no team scope, no
# arrive/wait split and no per-call channel. Anything else has to be rejected
# instead of silently degrading to a full barrier.
_BARRIER_DEFAULT_OPTIONS = {
    "group_kind": "block",
    "barrier_kind": "sync",
    "order": "acqrel",
    "index": 0,
    "context_id": 0,
    "memory_scope": "system",
}


def _check_barrier_option(name: str, value) -> None:
    value = tl._unwrap_if_constexpr(value)
    if value is None:
        return
    if isinstance(value, tl.tensor):
        raise TypeError(f"tle.distributed_barrier: {name} must be a compile-time constant")
    expected = _BARRIER_DEFAULT_OPTIONS[name]
    if isinstance(value, str):
        value = value.lower()
    if value != expected:
        raise NotImplementedError(
            f"tle.distributed_barrier: {name}={value!r} is not supported on Ascend; "
            f"aclshmem_barrier_all only implements the convergent full-communicator "
            f"barrier, so only {name}={expected!r} (or None) is accepted")


@builtin
def distributed_barrier(
    mesh=None,
    device_dptr=None,
    space: str = None,
    group_kind=None,
    barrier_kind=None,
    order=None,
    index: int = None,
    context_id: int = None,
    memory_scope: str = None,
    _semantic=None,
):
    """
    Distributed barrier across all ranks.

    Ascend lowers this to ``aclshmem_barrier_all``, which synchronises the whole
    communicator. Team-scoped barriers (sub-meshes, ``space="inter_node"``) and
    the non-default ``group_kind`` / ``barrier_kind`` / ``order`` / ``index`` /
    ``context_id`` / ``memory_scope`` options are rejected rather than ignored.
    """
    mesh = tl._unwrap_if_constexpr(mesh)
    if mesh is not None:
        assert isinstance(mesh, device_mesh), f"mesh must be device_mesh, got {type(mesh).__name__}"

    space = tl._unwrap_if_constexpr(space)
    if space is not None and not isinstance(space, str):
        raise TypeError(f"space must be str or None, got {type(space).__name__}")
    if space is not None:
        space = space.lower()
        aliases = {"device": "device", "intra": "device", "intra_node": "device", "world": "world"}
        if space not in aliases:
            raise NotImplementedError(
                f"tle.distributed_barrier: space={space!r} is not supported on Ascend; "
                f"only None, 'device' (aliases 'intra'/'intra_node') and 'world' map to "
                f"aclshmem_barrier_all")
        space = aliases[space]

    for name, value in (("group_kind", group_kind), ("barrier_kind", barrier_kind), ("order", order),
                        ("index", index), ("context_id", context_id), ("memory_scope", memory_scope)):
        _check_barrier_option(name, value)

    if device_dptr is not None:
        assert isinstance(device_dptr, tl.tensor) and device_dptr.dtype.is_ptr(), \
            "device_dptr must be a pointer tensor"

    return _extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_barrier_all", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def swizzle2d_Nz(
    iter_id,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    comm_npu_split=1,
):
    """Ascend Nz-format 2D swizzle for communication tiles."""
    data_row_loop_num = dl.cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = dl.cdiv(data_col_shape, tile_col_shape)
    data_loop_num = data_row_loop_num * data_col_loop_num
    rank_stride = rank_size // comm_npu_split
    swizzle_offset = comm_npu_split
    rank_loop_num = dl.cdiv(rank_size, swizzle_offset)
    rank_tile_idx = iter_id // (swizzle_offset * data_loop_num)
    data_rank_tile_idx = iter_id % (swizzle_offset * data_loop_num)
    rank_tile_size = swizzle_offset
    if rank_tile_idx == rank_loop_num - 1:
        rank_tile_size = rank_size - swizzle_offset * rank_tile_idx
    data_tile_idx = data_rank_tile_idx // rank_tile_size
    rank_idx = rank_tile_idx * swizzle_offset + data_rank_tile_idx % rank_tile_size
    rank_idx = (rank_idx * rank_stride) % rank_size + (rank_idx * rank_stride) // rank_size
    rank_idx = (rank_idx + data_tile_idx) % rank_size
    data_row_idx = data_tile_idx // data_col_loop_num
    data_col_idx = data_tile_idx % data_col_loop_num
    comm_row_size = dl.minimum(data_row_shape - data_row_idx * tile_row_shape, tile_row_shape)
    comm_col_size = dl.minimum(data_col_shape - data_col_idx * tile_col_shape, tile_col_shape)
    return data_row_idx, data_col_idx, rank_idx, comm_row_size, comm_col_size


@triton.jit
def gemm_swizzle2d_Nz(
    iter_id,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset=7,
):
    """Ascend Nz-format 2D swizzle for GEMM compute tiles."""
    data_row_loop_num = dl.cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = dl.cdiv(data_col_shape, tile_col_shape)
    col_loop_num = dl.cdiv(data_col_loop_num, swizzle_offset)
    n_tile_idx = iter_id // (swizzle_offset * data_row_loop_num)
    m_n_tile_idx = iter_id % (swizzle_offset * data_row_loop_num)
    n_tile_size = swizzle_offset
    if n_tile_idx == col_loop_num - 1:
        n_tile_size = data_col_loop_num - swizzle_offset * n_tile_idx
    data_row_idx = m_n_tile_idx // n_tile_size
    data_col_idx = n_tile_idx * swizzle_offset + m_n_tile_idx % n_tile_size
    if n_tile_idx % 2 == 1:
        data_row_idx = data_row_loop_num - data_row_idx - 1
    return data_row_idx, data_col_idx
