# TLE 分布式原语在 Ascend 上的 lowering 验证

> 目的：给出可复现的验证方法，并用一次**真实编译**展示 TLE 分布式原语如何一步步变成 `distributed.*`，再变成 `hivm.hir.custom` + ACLSHMEM 符号。
> 相关文档：[Ascend Distributed 调研与集成总结](ascend_distributed_summary_zh.md)

## 阅读指引：只需要关注一条 IR 变化

TLE 前端有三个出口，本文只涉及第一个：

```text
a) distributed 子集（shard_id / remote / distributed_barrier）
     → distributed.*（TTIR 方言）→ HIVM          ← 本文主路径，与 dl 同路
b) DSA / pipe 子集（tle.dsa.* / tle.pipe）
     → CommonIR(tile.*) → HIVM                   ← 单设备路径，与跨卡无关
c) signal / wgmma / remote_pointers 等
     → TLE 方言 → GPU LLVM                       ← Ascend 未接入
```

| IR 变化 | 是否存在 | 与本文关系 | 需要掌握的程度 |
|---|---|---|---|
| `distributed.*` → `hivm.hir.custom` | ✅ | **主路径** | 必须 |
| TLE distributed → `distributed.*`（TTIR） | ✅ | **主路径** | 必须 |
| TLE DSA / pipe → CommonIR（`tile.*`） | ✅ | 不相关 | 知道边界即可 |
| CommonIR（`tile.*`）→ HIVM | ✅（`commonir-to-hivm`） | 不相关 | 知道存在即可 |

> `dl` 与 `tle` 是 Python API，不构成 IR 层；分布式原语不会经过 CommonIR
> （CommonIR 没有跨 rank 语义），因此不需要为本文去研究 tile.* 的 lowering。

## 0. 结论速览

**范围**：只讨论 TLE 的 distributed 子集；`tle.load` / `extract_tile` / `pipe` / `dsa.*` 属于单设备能力，与 `dl.*` 无关。

**“变成 dl 原语”的含义**：Ascend 下不存在 TLE → dl 的文本或 IR 转换。`tle.language` 在导入时把
distributed 原语替换成 `dsa/ascend/distributed.py` 的实现，后者调用与 `dl.*` **相同的 builder 方法**，
因此直接产出 `distributed.*` 方言。`dl` 只是 Python 命名空间，IR 中没有 `dl.*`，只有 `distributed.*`。

| TLE 原语 | 对应 `dl` / 底层 | 验证结果 | 降为 |
|---|---|---|---|
| `tle.shard_id` | `dl.rank`（同一 `create_get_rank`） | ✅ 通过（仅 `device` / `0` / `-1`） | `distributed.get_rank` → `hivm.hir.custom symbol="aclshmem_my_pe"`（VECTOR） |
| `tle.remote`（`space=None/"device"`） | `dl.symm_at`（同一 `create_symm_at`） | ✅ 通过（标量指针；`offset` 不支持） | `distributed.symm_at` → `hivm.hir.custom symbol="aclshmem_ptr_float"`（VECTOR） |
| `tle.distributed_barrier` | `dl.extern_call` / `libshmem_device.barrier_all` | ⚠️ **发现缺陷**：映射正确，但缺 `distributed.core_type`，pass 报错、整条编译失败；补 `"cube_and_vector"` 后通过 | `distributed.extern_call(symbol="aclshmem_barrier_all")` → `hivm.hir.custom`（CUBE_AND_VECTOR；npuir 中拆成 CUBE / VECTOR 两份） |
| `tle.device_mesh` | 无（纯前端对象） | —（只支持 `device` 轴） | 不产生 IR；拓扑需 host 以 constexpr 传入 kernel |
| `tle.sharding` | 无（注解） | — | 不产生代码，仅作为元数据 |
| `tle.n_pes`（≈ `dl.num_ranks`） | `dl.num_ranks` | ❌ 未覆盖 | Ascend 未替换该入口，也不在白名单 |
| `tle.signal` / `signal_wait` | 概念上对应 `dl.notify` / `dl.wait` | ❌ 未接入 | 仍是 FlagCX 槽位模型（`slot_id`），不在 Ascend 白名单 |
| `tle.reshard` | 无 | ❌ 未实现 | `NotImplementedError`（M4） |
| `tle.distributed_dot` | 无 | ❌ 未实现 | `NotImplementedError`（M5） |

> 严格按“端到端可编译”统计（截至 HEAD `921e9a21e`），当时只有 `shard_id` 和 `remote` 完整可用；
> `distributed_barrier` 映射正确，但需要先补 `distributed.core_type`。
>
> **后续状态（本文写成后的工作树）**：`distributed.core_type` 已在
> `dsa/ascend/distributed.py::_dispatch` 统一补齐，`distributed_barrier` 无需任何 patch 即可端到端编译；
> 同时接通了 `n_pes` / `signal` / `signal_wait`，并收紧了 barrier 选项校验。改动清单与验证记录见
> [TLE 分布式原语在 Ascend 上的映射、缺口与新增原语判据](tle_ascend_distributed_primitive_mapping_zh.md)。
>
> 其中 barrier 的 `core_type` 最终取 **`vector`** 而不是本文验证时用的 `cube_and_vector`：
> 后者会让 BiShengIR 以 mix 模式编译，整个 body 被克隆到 AIC/AIV 两份，distributed op 每个 PE
> 执行两次（多卡实测 signal 计数翻倍、等值 wait 挂死）。详见该文档 §5.1。
> 本文以下内容保留为缺陷复现记录。

验证环境：FlagTree 分支 `ascend_triton_distributed`，源码 HEAD `921e9a21e`。本页结果由**用该 commit 现场构建的 wheel** 复现：

```
flagtree-0.7.0+ascend.git921e9a21-cp312-cp312-linux_aarch64.whl   (590 MB)
```

构建开关 `TRITON_BUILD_TD=1`、`TRITON_BUILD_NPUIR=ON`，Triton 3.6.0，安装在隔离 venv `/tmp/venv-flagtree-td`（`--system-site-packages`）；机器 8 × Ascend950DT，CANN 环境可用（`npu-smi` 存在）。

可执行 demo 见 [`documents/tle_lowering_demo/`](tle_lowering_demo/README.md)（`ir_demo.py` / `compile_demo.py` / `run_demo.sh`）。

## 1. 被测 kernel

```python
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def _ring_exchange(x_ptr, out_ptr):
    v = tl.load(x_ptr)
    rank = tle.shard_id(axis=-1)          # 期望：dl.rank
    tle.distributed_barrier()             # 期望：libshmem_device.aclshmem_barrier_all
    peer = (rank + 1) % 8
    remote_ptr = tle.remote(x_ptr, peer)  # 期望：dl.symm_at
    tl.store(remote_ptr, v)
    tl.store(out_ptr, v)
```

选择依据：在 Ascend 上，`tle.language` 会把分布式原语整体替换为 `tle/language/dsa/ascend/distributed.py` 的实现（`tle/language/__init__.py:71`），这三个原语分别对应 `create_get_rank`、`extern_call(aclshmem_barrier_all)`、`create_symm_at`——正是 `dl.*` 使用的同一批 builder 方法。

## 2. 方法 A：IR 级单测（不跑硬件，最快）

### 2.1 脚本

```python
# lower_demo.py
import os
import triton, triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend

# kernel 定义见第 1 节 ...

ctx = libtriton.ir.context()
libtriton.ir.load_dialects(ctx)
libtriton.distributed.ir.load_dialects(ctx)
libtriton.ascend.ir.load_dialects(ctx)
libtriton.ascend.load_dialects(ctx)

target = GPUTarget("npu", "Ascend910B1", 0)          # 纯 IR 阶段，架构仅影响选项
backend = make_backend(target)
options = backend.parse_options({})
src = ASTSource(_ring_exchange, {"x_ptr": "*fp32", "out_ptr": "*fp32"}, {})
mod = src.make_ir(target, options, backend.get_codegen_implementation(options),
                  backend.get_module_map(), ctx)

print(str(mod))                                       # ① 看 TTIR

ops = []
mod.walk(lambda op: ops.append(op.get_name())
         if op.get_name().startswith(("distributed.", "tle.")) else None)
print(ops)                                            # ② 看 op 集合

pm = libtriton.ir.pass_manager(ctx)
libtriton.distributed.ascend_passes.ttgpuir.add_convert_triton_distributed_to_hivm(pm)
os.environ["MLIR_DISABLE_MULTITHREADING"] = "1"
pm.run(mod, "")
print(str(mod))                                       # ③ 看 conversion 结果
```

### 2.2 实测输出

**① TTIR（conversion 之前）：只有 `distributed.*`，没有任何 `tle.*`**

```mlir
%v = tt.load %x_ptr : !tt.ptr<f32>
%rank = arith.constant -1 : i32
%rank_0 = distributed.get_rank %rank : i32
distributed.extern_call {libname = "libshmem_device", libpath = "", pure = false,
                         symbol = "aclshmem_barrier_all"} : () -> ()
%peer_13 = arith.remsi %peer_10, %peer_12 : i32
%remote_ptr = distributed.symm_at %x_ptr, %peer_13, : !tt.ptr<f32>
tt.store %remote_ptr, %v : !tt.ptr<f32>
```

**② op 集合**

```
['distributed.get_rank', 'distributed.extern_call', 'distributed.symm_at']
```

**③ 跑 `convert-triton-distributed-to-hivm` 之后**

```mlir
%rank = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed,
          hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>,
          hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect,
          symbol = "aclshmem_my_pe"} "dist.aclshmem_my_pe" -> i32

%remote_ptr = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed,
          hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>,
          hivm.vf_mode = #hivm.vf_mode<SIMD>,
          symbol = "aclshmem_ptr_float"} "dist.aclshmem_ptr_float"
          ins(%x_ptr, %peer_0 : !tt.ptr<f32>, i32) -> !tt.ptr<f32>
```

**注意**：`distributed.extern_call`（barrier）**没有被转换**，pass 同时报出：

```
error: distributed op requires a distributed.core_type attribute;
       expected one of: cube, vector, cube_and_vector, cube_or_vector
```

这是第 4 节要分析的缺陷。

## 3. 方法 B：完整编译 + 三阶段 IR dump（端到端）

### 3.1 打开 dump

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_DEBUG=1
python compile_demo.py
```

`compile_demo.py` 的内容与方法 A 相同，只是把 `make_ir` 换成一次真实的 `warmup`：

```python
import torch, torch_npu  # noqa
_ring_exchange.warmup(torch.zeros(1, device="npu"),
                      torch.zeros(1, device="npu"), grid=(1,))
```

### 3.2 产物

```
~/.triton/dump/<hash>/
├── kernel.ttir.mlir          # ① Python AST → TTIR（含 distributed.*）
├── kernel.ttadapter.mlir     # ② distributed → hivm.custom
├── kernel.mlir               #    bufferize 之后的 Ascend 侧 IR
├── kernel.mlirbc             #    MLIR bytecode（此处若无 distributed 方言会直接失败）
├── kernel.npuir.mlir         #    cube/vector 拆分后的 NPUIR
└── kernel.o                  #    bishengir-compile 产出的设备目标文件
```

编译日志里可以直接看到真实的 pass pipeline（节选）：

```
triton-opt kernel.ttir.mlir
  --pass-pipeline=any(convert-triton-distributed-to-hivm{...},
                      triton-control-flow-opt,
                      ...,
                      commonir-to-hivm,
                      ...,
                      triton-to-linalg{...},
                      commonir-to-hivm,
                      ...,)
```

由此可确认两点：

1. `convert-triton-distributed-to-hivm` 位于 ttadapter 的**最前面**，早于 Triton→Linalg；
2. CommonIR（`commonir-to-hivm`）在 linalg 前后各跑一次，和分布式 lowering 是**两条并行路径**。

### 3.3 实测结果

**① `kernel.ttir.mlir`**

```mlir
%rank_2 = distributed.get_rank %rank : i32
distributed.extern_call {distributed.core_type = "cube_and_vector", libname = "libshmem_device",
                         libpath = "", pure = false, symbol = "aclshmem_barrier_all"} : () -> ()
%remote_ptr = distributed.symm_at %x_ptr, %peer_9, : !tt.ptr<f32>
```

（这里的 `distributed.core_type` 是第 4 节修复后才出现的，见下文。）

**② `kernel.ttadapter.mlir`**

```mlir
%rank = hivm.hir.custom {..., hivm.tcore_type = #hivm.tcore_type<VECTOR>,
          no_side_effect, symbol = "aclshmem_my_pe"} "dist.aclshmem_my_pe" -> i32

hivm.hir.custom {already_sub_block_bound, ..., hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>,
          libname = "libshmem_device", symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"

%remote_ptr = hivm.hir.custom {..., hivm.tcore_type = #hivm.tcore_type<VECTOR>,
          symbol = "aclshmem_ptr_float"} "dist.aclshmem_ptr_float"
          ins(%x_ptr, %peer_3 : memref<?xf32>, i32) -> memref<?xf32>
```

**③ `kernel.npuir.mlir`：barrier 被拆成 cube 与 vector 各一份**

```mlir
hivm.hir.custom {..., hivm.tcore_type = #hivm.tcore_type<CUBE>,
                 symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
...
hivm.hir.custom {..., hivm.tcore_type = #hivm.tcore_type<VECTOR>,
                 symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
```

这正是 `distributed.core_type="cube_and_vector"` 的语义：barrier 同时存在于 cube func 与 vector func 中，是计算-通信重叠成立的前提。

**④ `kernel.o`**

```
$ strings kernel.o | grep aclshmem
aclshmem_ptr_float.vector
```

## 4. 发现的缺陷：`tle.distributed_barrier` 缺 `distributed.core_type`

### 4.1 现象（可复现）

不施加任何修复，直接跑完整编译：

```
error: distributed op requires a distributed.core_type attribute;
       expected one of: cube, vector, cube_and_vector, cube_or_vector
...
MLIRCompilationError:
[LinalgIRToBytecodeByTritonMLIROpt] encounters error:
  error: Dialect `distributed' not found for custom op 'distributed.extern_call'
```

解释：conversion pass 因为缺属性拒绝转换，`distributed.extern_call` 残留；下一个阶段 `triton-mlir-opt --emit-bytecode` 不认识 distributed 方言，整条编译失败。

### 4.2 根因

| 位置 | 行为 |
|---|---|
| TDA `python/triton_dist/language/extra/ascend/libaclshmem_device.py:108` | 每个 `aclshmem_*` 包装都会 `_set_attr(call, {"distributed.core_type": ...})`；barrier 标为 `cube_and_vector` |
| FlagTree `python/triton/experimental/tle/language/dsa/ascend/distributed.py:88-120` | 自己实现的 `_extern_call` 只调用 `builder.create_extern_call`，**从未设置 `distributed.core_type`** |

### 4.3 修复与验证

最小修复（演示用 monkeypatch，生产修复应落在 `_extern_call` 或直接复用 `libshmem_device.barrier_all`）：

```python
import triton.experimental.tle.language.dsa.ascend.distributed as ad
_orig = ad._extern_call

def _extern_call_with_core_type(*args, **kwargs):
    ret = _orig(*args, **kwargs)
    sem = kwargs.get("_semantic")
    ret.handle.set_attr("distributed.core_type",
                        sem.builder.get_string_attr("cube_and_vector"))
    return ret

ad._extern_call = _extern_call_with_core_type
```

加上之后：完整编译成功（约 2.3 s），`kernel.ttadapter.mlir` 中出现 `CUBE_AND_VECTOR` 的 `aclshmem_barrier_all`，`kernel.o` 正常产出。

建议的正式修复（二选一）：

1. 在 TLE 的 `_extern_call` 传入 `core_type`，按符号查表（`aclshmem_barrier_all` / `aclshmem_barrier` → `cube_and_vector`，`aclshmemx_*_vec` → `vector`，其余 `cube_or_vector`）；
2. 更简单：Ascend 分支的 `distributed_barrier` 直接调用 `triton_dist.language.extra.libshmem_device.barrier_all()`，完全复用 TDA 已验证的实现，不重复造 `extern_call`。

## 5. 验证判定标准

一次 lowering 算“验证通过”，必须同时满足：

| # | 检查点 | 判定 |
|---|---|---|
| 1 | `kernel.ttir.mlir` | 只有 `distributed.*`，没有 `tle.*` 残留 |
| 2 | TTIR 中每个 `distributed.extern_call` | 带 `distributed.core_type` |
| 3 | `kernel.ttadapter.mlir` | 无 `distributed.*` 残留；每个 `hivm.hir.custom` 有 `symbol` 和 `hivm.tcore_type` |
| 4 | `kernel.mlirbc` | 能成功产出（方言残留会在此步暴露） |
| 5 | `kernel.o` | 能产出，且 `strings` 能看到期望的 `aclshmem_*` 符号 |
| 6 | 多卡运行 | 对称堆初始化、signal 缓冲、parity 正确，与 PyTorch collective 对比通过（见 `docs_ascend/zh/developer-guide/kernel-debugging/precision.md`） |

前 5 条是本次真实编译覆盖的范围；第 6 条需要多卡运行环境，本次未覆盖。

## 6. 复现步骤

仓库内已提供可执行版本：[`documents/tle_lowering_demo/`](tle_lowering_demo/README.md)，可直接
`PYTHON=<venv>/bin/python bash run_demo.sh` 复现下面三条命令。

```bash
# 1. 安装带 distributed 插件的 FlagTree（TRITON_BUILD_TD=ON）
python -c "from triton._C.libtriton import distributed; print(distributed.ascend_passes)"

# 2. IR 级验证（方法 A）
python lower_demo.py

# 3. 端到端验证（方法 B）
TRITON_ALWAYS_COMPILE=1 TRITON_DEBUG=1 python compile_demo.py
ls -t ~/.triton/dump/ | head -1

# 4. 检查
D=~/.triton/dump/<hash>
grep "distributed\."            $D/kernel.ttir.mlir
grep "hivm.hir.custom"          $D/kernel.ttadapter.mlir
grep "aclshmem_barrier_all"     $D/kernel.npuir.mlir
strings $D/kernel.o | grep aclshmem
```

## 7. 用最新 commit 的 wheel 现场复现（2026-11-23）

### 7.1 出包与安装

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/cdy/FlagTree
export FLAGTREE_BACKEND=ascend TRITON_BUILD_TD=1 TRITON_BUILD_PROTON=OFF \
       TRITON_BUILD_NPUIR=ON ASCENDNPUIR_BUILD_ROOT=/tmp/npuir-build \
       ASCENDNPUIR_REUSE_BUILD=1 MAX_JOBS=1 TMPDIR=/tmp
python3 -m pip wheel . --no-build-isolation --no-deps -w /tmp/flagtree-dist/
# -> flagtree-0.7.0+ascend.git921e9a21-cp312-cp312-linux_aarch64.whl  (590 MB)

python3 -m venv --system-site-packages /tmp/venv-flagtree-td
/tmp/venv-flagtree-td/bin/pip install --no-deps --force-reinstall \
  /tmp/flagtree-dist/flagtree-0.7.0+ascend.git921e9a21-*.whl
```

> wheel 只放在 `/tmp/flagtree-dist/`，不提交进仓库（体积原因）；AscendNPU-IR 已在 pin `8bc48fb585`，
> BiShengIR payload 复用 `/tmp/npuir-build/bishengir-output`。

### 7.2 结果

| 步骤 | 命令 | 结果 |
|---|---|---|
| 1 | `ir_demo.py` | `get_rank` / `symm_at` 转换成功；`extern_call` 残留，pass 报缺 `core_type` |
| 2 | `compile_demo.py`（不修复） | 失败：`Dialect 'distributed' not found for custom op 'distributed.extern_call'` |
| 3 | `compile_demo.py --fix-core-type` | 成功：`asm keys = [bcmlir, mlirbc, npubin, source, ttadapter, ttir]`，生成 `kernel.o` |

步骤 3 的 `kernel.ttadapter.mlir` 实测：

```mlir
%rank = hivm.hir.custom {..., hivm.tcore_type = #hivm.tcore_type<VECTOR>, no_side_effect,
          symbol = "aclshmem_my_pe"} "dist.aclshmem_my_pe" -> i32
hivm.hir.custom {..., hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>,
          libname = "libshmem_device", symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
%remote_ptr = hivm.hir.custom {..., hivm.tcore_type = #hivm.tcore_type<VECTOR>,
          symbol = "aclshmem_ptr_float"} "dist.aclshmem_ptr_float" ...
```

`kernel.npuir.mlir` 中 `aclshmem_barrier_all` 出现两次（CUBE 与 VECTOR 各一份）；`strings kernel.o` 可见
`aclshmem_ptr_float.vector`。

### 7.3 结论

- **HEAD `921e9a21e` 仍然存在该缺陷**，不是旧 wheel（`git0543cf98`）的残留问题；
- 缺陷只影响 `tle.distributed_barrier`；`tle.shard_id` / `tle.remote` 的 lowering 在最新 commit 上正常；
- 修复很小：按 TDA 的映射补齐 `distributed.core_type` 即可；建议合入后把本 demo 的步骤 1–3 固化进 CI。
- **修复已落地**（工作树）：`_dispatch` 按符号表补 `core_type`，CI 用例为
  `python/test/tle/unit/test_tle_ascend_distributed_lowering.py`。

## 8. 局限与后续

- 本次只覆盖 `device` 级的 `shard_id` / `remote` / 全组 barrier；子 mesh barrier、`space="node"` 的远端访问、`signal` / `signal_wait` 未验证（后者不在 Ascend 白名单）。
- `tle.remote` 的 `offset` 在 Ascend 前端被 assert 拒绝（`offset not applied at the HIVM level`），调用方需自行 `remote_ptr + offset`。
- `tle.shard_id(axis=-1)` 降为 `aclshmem_my_pe`，`axis` 参数被忽略；多维 mesh 坐标尚未实现。
- 上述 core_type 修复落地后，建议把第 5 节的检查点 1–5 固化成 CI 用例（可直接复用 TDA 的 `test_external_call_op_core_types.py` 结构）。
