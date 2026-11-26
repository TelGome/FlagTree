# TLE 分布式原语 lowering demo

本目录是 [`documents/tle_distributed_lowering_verification_zh.md`](../tle_distributed_lowering_verification_zh.md)
的可执行版本：用同一个 kernel 演示 TLE 分布式原语在 Ascend 上如何变成 `distributed.*`，再变成
`hivm.hir.custom` + ACLSHMEM 符号。

先说明一点：**这不是文本或源码到源码的转换**。Ascend 前端在导入期把 TLE 的 distributed 入口替换成
`dsa/ascend/distributed.py` 的实现，后者调用与 `dl.*` 相同的 builder 方法；`dl` 只是 Python 命名空间，
IR 里没有 `dl.*`，只有 `distributed.*`。

## 涉及哪几种 IR

TLE 前端有**三个出口**，只有第一个与跨卡通信有关：

```text
a) distributed 子集：shard_id / remote / distributed_barrier
     → distributed.*（TTIR 方言）→ HIVM          ← 本 demo 走这条，和 dl 同路
b) DSA / pipe 子集：tle.dsa.* / tle.pipe
     → CommonIR(tile.*) → HIVM                   ← 单设备路径，没有跨 rank 语义
c) signal / wgmma / remote_pointers 等
     → TLE 方言 → GPU LLVM                       ← Ascend 未接入
```

本 demo 的实际链路是 **4 层有语义的 IR**：

```text
① TTIR：tt.* + distributed.*
② Ascend 适配 IR：hivm.hir.custom + Linalg/MemRef/Tensor   （distributed 在此消失）
③ AscendNPU IR：HIVM / HFusion / Annotation                （cube/vector 拆分）
④ LLVM IR → 设备二进制 kernel.o
```

| IR 变化 | 是否存在 | 与本 demo 关系 | 需要掌握的程度 |
|---|---|---|---|
| `distributed.*` → `hivm.hir.custom` | ✅ | **主路径** | 必须 |
| TLE distributed → `distributed.*`（TTIR） | ✅ | **主路径** | 必须 |
| TLE DSA / pipe → CommonIR（`tile.*`） | ✅ | 不相关（单设备） | 知道边界即可 |
| CommonIR（`tile.*`）→ HIVM | ✅（`commonir-to-hivm`） | 不相关 | 知道存在即可 |

> CommonIR 的 op 只有 alloc/copy/load/store/flag/cube_launch 等，**没有跨 rank 语义**，
> 因此分布式原语不能也不应该走 CommonIR。

## 文件

| 文件 | 作用 |
|---|---|
| `tle_dist_kernel.py` | 被测 kernel：`tle.shard_id` / `tle.distributed_barrier` / `tle.remote` |
| `ir_demo.py` | 方法 A：AST → TTIR → 手动跑 `convert-triton-distributed-to-hivm`，不执行 kernel |
| `compile_demo.py` | 方法 B：真实编译，产出 `~/.triton/dump/<hash>/` 下各阶段 IR 与 `kernel.o` |
| `run_demo.sh` | 依次运行上面两个脚本 |
| `multirank_probe.py` | 多卡运行验证：`shard_id` / `n_pes` / barrier / `remote` / `signal` 的实测校验 |

## 前置条件

1. 安装的 FlagTree 是带 `distributed` 插件的 Ascend 构建（`TRITON_BUILD_TD=1`）：

   ```bash
   python -c "from triton._C.libtriton import distributed; \
              print(hasattr(distributed, 'ascend_passes'))"
   # True
   ```

2. CANN 环境已 source，且至少一张 NPU 可见（只做编译，不做多卡运行）：

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   npu-smi info
   ```

3. 若使用独立 venv，建议 `--system-site-packages`，并确保 `torch`/`torch_npu` 可导入。

## 运行

```bash
cd documents/tle_lowering_demo
bash run_demo.sh
# 或指定解释器
PYTHON=/tmp/venv-flagtree/bin/python bash run_demo.sh
```

### 预期输出

**步骤 1（`ir_demo.py`）**：TTIR 中只有 `distributed.*`

```mlir
%rank_2 = distributed.get_rank %rank : i32
distributed.extern_call {libname = "libshmem_device", libpath = "", pure = false,
                         symbol = "aclshmem_barrier_all"} : () -> ()
%remote_ptr = distributed.symm_at %x_ptr, %peer_9, : !tt.ptr<f32>
```

跑完 conversion pass 后，`get_rank` / `symm_at` 变成：

```mlir
hivm.hir.custom {... symbol = "aclshmem_my_pe"} "dist.aclshmem_my_pe" -> i32
hivm.hir.custom {... symbol = "aclshmem_ptr_float"} "dist.aclshmem_ptr_float" ...
```

但 `distributed.extern_call` **会残留**，并打印：

```
error: distributed op requires a distributed.core_type attribute;
       expected one of: cube, vector, cube_and_vector, cube_or_vector
```

> 该缺陷已在生产代码中修复：`dsa/ascend/distributed.py::_dispatch` 现在按符号表补
> `distributed.core_type`（`_CORE_TYPE_BY_SYMBOL`）。上面的报错只会在修复前的构建里出现，
> 现在运行 `ir_demo.py` 会直接输出 “OK: no distributed.* left”。

**步骤 2（无修复的完整编译）**：修复前的构建预期失败

```
MLIRCompilationError:
  error: Dialect `distributed' not found for custom op 'distributed.extern_call'
```

修复后同一条命令直接成功。

**步骤 3（`--fix-core-type`）**：现在是 no-op，只保留命令行兼容

```
note: --fix-core-type is a no-op; the frontend sets distributed.core_type=vector itself
asm keys    : ['bcmlir', 'mlirbc', 'npubin', 'source', 'ttadapter', 'ttir']
OK: compile finished
```

修复后的 `kernel.npuir.mlir` 只有一个 `func.func @ring_exchange`（不再拆成
`*_mix_aic` / `*_mix_aiv`），`aclshmem_barrier_all` 也只出现一次：

```mlir
hivm.hir.custom {..., hivm.tcore_type = #hivm.tcore_type<VECTOR>,
                 symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
```

这不是“少做事”：把 barrier 标成 `cube_and_vector` 会让 BiShengIR 用 mix 模式编译，
整个 body 被克隆到 AIC 和 AIV 两份，distributed op 每个 PE 执行两次。详见
[映射与判据文档](../tle_ascend_distributed_primitive_mapping_zh.md) §5.1。

## 历史问题（已修复）

`tle.distributed_barrier` 的 Ascend 实现（`python/triton/experimental/tle/language/dsa/ascend/distributed.py`）
曾经自己拼 `create_extern_call`，但没有像 `triton_dist` 的
`libaclshmem_device.py` 那样写入 `distributed.core_type`，conversion pass 因此拒绝转换，
后续 bytecode 阶段报 “Dialect `distributed' not found”。

现在的修复位于 `dsa/ascend/distributed.py::_dispatch`：所有 extern call 统一按符号表打
`core_type`（barrier / sync → `cube_and_vector`，`*_vec` → `vector`，其余 `cube_or_vector`）。
`--fix-core-type` 只在验证旧构建时有用。回归用例见
`python/test/tle/unit/test_tle_ascend_distributed_lowering.py`。

## 多卡运行验证

`multirank_probe.py` 在真实多卡上校验 `shard_id`、`n_pes`、barrier 后的 `remote` 读、
`signal(inc)` / `signal(add)` 的计数，每个 rank 输出 PASS/FAIL：

```bash
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=8990 \
  torchrun --nproc-per-node=4 multirank_probe.py
```

注意两点：

- 必须**独占**所有卡。和别的多卡任务并发时，会出现 `AclrtCreateStreamWithConfig`
  driver error 507899 或 worker 被 SIGKILL，看起来像用例失败，其实是设备争用。
- torchrun 要用装了本仓 wheel 的那个解释器（例如 `python -m torch.distributed.run`），
  系统 `torchrun` 会用系统 Python 里的另一份 triton。

实测 2 / 4 / 6 / 8 卡全部通过，结果见
[映射与判据文档](../tle_ascend_distributed_primitive_mapping_zh.md) §6.3。

## 查看 dump

```bash
D=$(ls -dt ~/.triton/dump/*/ | head -1)
grep "distributed\."        "$D/kernel.ttir.mlir"
grep "hivm.hir.custom"      "$D/kernel.ttadapter.mlir"
grep "aclshmem_barrier_all" "$D/kernel.npuir.mlir"
strings "$D/kernel.o" | grep aclshmem
```
