# TLE 分布式原语在 Ascend 上的映射、缺口与新增原语判据

> 相关文档：[Ascend Distributed 调研与集成总结](ascend_distributed_summary_zh.md)、
> [TLE 分布式原语 lowering 验证](tle_distributed_lowering_verification_zh.md)、
> 可执行 demo：[`documents/tle_lowering_demo/`](tle_lowering_demo/README.md)

## 0. 结论速览

1. **不存在 “TLE → dl” 的转换**。`tle.*` 和 `dl.*` 都只是 Python 命名空间；Ascend 上
   `tle/language/__init__.py` 在导入期把分布式入口替换成
   `tle/language/dsa/ascend/distributed.py` 的实现，两者调用同一批 builder，直接产出
   `distributed.*` 方言。IR 里既没有 `dl.*` 也没有 `tle.*`（分布式子集）。
2. **不能表达的情况分四类**，只有两类需要“新增原语”，另外两类要么只需组合现有原语，要么
   根本不该做。
3. **新增原语的成本次序**：Python 语义层组合 ≪ FlagTree 侧 TLE/`distributed` op + pass
   ≪ AscendNPU-IR（`Shmem.cpp` 新设备符号）≪ ACLSHMEM 上游新设备 API。
4. **分布式不经过 CommonIR**。CommonIR（`tile.*`）只有单设备 buffer 语义；分布式 lowering
   在 ttadapter 最前面就把 `distributed.*` 变成 `hivm.hir.custom`，与 `commonir-to-hivm`
   是并行两条路径。

---

## 1. 映射关系：TLE 原语 → `distributed.*` → ACLSHMEM 设备符号

`distributed.*` 方言和 `convert-triton-distributed-to-hivm` 都 vendored 在本仓
`third_party/ascend/` 下，转换规则是「op + 操作数类型 → 设备符号」的查表。真正锁住能力边界的
不是方言，而是 AscendNPU-IR 的
`third_party/ascend/AscendNPU-IR/bishengir/lib/Template/lib/Shmem/Shmem.cpp`
——里面是固定的一批 `_mlir_ciface_aclshmem_*` 设备端包装（约 45 个基础符号 + 按 dtype /
transport 展开的宏）。

| TLE 原语 | `distributed.*` | 设备符号 | 状态 |
|---|---|---|---|
| `shard_id` | `get_rank` | `aclshmem_my_pe` | ✅ 已验证（`axis` 仅 `-1`/`0`/`"device"`） |
| `n_pes` | `get_num_ranks` | `aclshmem_n_pes` | ✅ 本次接通（返回 communicator PE 数） |
| `remote(space="device")` | `symm_at` | `aclshmem_ptr_<dtype>` | ✅ 已验证（`offset` 仍不支持） |
| `signal(op="inc"/"add")` | `notify` | `aclshmemx_signal_op`（i32）/ `aclshmem_atomic_add_*`（i64） | ✅ 本次接通（IR 级 + 编译验证） |
| `signal_wait(wait_kind="signal")` | `extern_call` | `aclshmem_wait_<dtype>` | ✅ 本次接通（见 §5.2 为何不走 `distributed.wait`） |
| `distributed_barrier(space=None/"device"/"world")` | `extern_call` | `aclshmem_barrier_all` | ✅ 已验证；非默认选项现在直接报错 |
| `device_mesh` / `sharding` / `make_sharded_tensor` | 无 IR | — | 纯前端注解 |
| `remote(space="cluster"/"node")` | — | — | ❌ 见 §2-A / §2-C |
| `reshard` | — | — | ❌ M3，先在前端展开为 put/get + signal 组合 |
| `distributed_dot` | — | — | ❌ 不做，见 §2-A |

slot / 地址约定（本次确定的 ABI）：

- `signal` / `signal_wait` 的 `device_dptr` 指向对称堆里的信号 buffer（`int32` 或 `int64`）；
- 每个 slot 独占 `_SIGNAL_SLOT_STRIDE_BYTES = 64` 字节，相邻 slot 不共享 cache line；
- `slot_id` 是槽索引，`peer` 是该通信域内的 rank。

---

## 2. “不能表达”的四类

| 类 | 具体项 | 根因 | 结论 |
|---|---|---|---|
| **A 设备没这个能力** | `space="cluster"`（DSMEM）、cluster sub-mesh / grid barrier、`distributed_dot`、`wait_kind="shadow"` | Ascend 没有 CTA cluster / DSMEM；ACLSHMEM 没有运行时 shadow buffer | 不新增原语，fail-fast，写进白名单/文档 |
| **B 能力有、只缺胶水** | `n_pes`、`signal`、`signal_wait`、team barrier、`space="node"` 的 put/get | 设备符号已存在（`Shmem.cpp` 有对应包装） | **组合现有原语**，只改 Python |
| **C 形态不同，需要保留 provenance** | `remote(space="node")` + 配对 load/store | TLE 是 pointer+访存，ACLSHMEM 是 `getmem/putmem` 一次性搬移；`symm_at` 之后指针退化成普通 `!tt.ptr`，后端无法区分远端/本地 | 需要把 TLE op 留在 TTIR 并加 pass 做 load/store 配对 |
| **D 是编译器动作而非原语** | `reshard` | 本质是“比较源/目标 spec，插入集合通信” | 先在前端展开成 B 类原语的组合，后续再下沉 |

C 类是唯一的“必须新增 IR”的场景，而且新 op 只做**标记**，最终仍落到已有的
`aclshmem_putmem/getmem` 符号上——不需要动 AscendNPU-IR。

---

## 3. 两种方法的判据

| 情况 | 选法 | 改动面 |
|---|---|---|
| 一次函数调用能表达完整语义 | 组合现有 `distributed.*` + `extern_call` | 仅 `dsa/ascend/distributed.py` |
| 需要 SSA/依赖契约（token、`core_type`、pipe 属性） | 组合，但必须用带契约的现有 op | 仅 Python |
| 需要跨语句分析（node put/get 配对、sharding-aware 访存） | TTIR 保留/新增 **TLE op** + FlagTree 侧 pass | TLE 方言 + FlagTree pass |
| 需要新设备符号 | 先看 `Shmem.cpp` 有没有；没有则必须动 AscendNPU-IR（+ ACLSHMEM 头里有该 device API） | AscendNPU-IR |
| 硬件没有该能力 | 不加，显式报错 | 无 |

两条容易踩坑的推论：

- **新增 `distributed.*` op 往往只是 FlagTree 本地改动**：方言、转换 pass、Python builder
  （`third_party/ascend/python/src/dist/ir.cc`）都在本仓；`distributed.extern_call` 本身是
  通用逃生口，能表达就不必加 op。
- **不要把“能加 op”当成“该加 op”**：`signal` / `signal_wait` / `n_pes` 都能在不改任何 IR 的
  前提下落地，这类缺口优先用组合解决，避免方言和 TDA 双向漂移。

---

## 4. 三层 IR 变化

```text
TLE Python 语义层 ─→ TTIR(distributed.*) ─→ hivm.hir.custom ─→ BiShengIR/ACLSHMEM ─→ ccec
                        ↑ 变化②                ↑ 变化①（仅新符号时）
CommonIR(tile.*) ───────┘ 不参与
```

| 变化 | 是否需要 | 说明 |
|---|---|---|
| ① AscendNPU-IR / TDA | 仅当需要新的 `_mlir_ciface_aclshmem_*` 符号 | 由 ACLSHMEM 设备 API 决定，与方言无关 |
| ② TLE 原语 → TTIR | 需要跨语句分析时才做 | 默认档：前端直接产出 `distributed.*`；需要 provenance 时改用 TLE op + pass |
| ③ CommonIR → AscendNPU IR | **不需要** | CommonIR 无跨 rank 语义；不要把分布式语义塞进 CommonIR |

`convert-triton-distributed-to-hivm` 位于 ttadapter 管线最前面（早于 `triton-to-linalg`），
`commonir-to-hivm` 在 linalg 前后各跑一次，两者互不依赖。

---

## 5. 本次落地

### 5.1 修复 `distributed.core_type` 缺失（M0）

`dsa/ascend/distributed.py` 的 `_extern_call` 之前只调 `create_extern_call`，从不设置
`distributed.core_type`，转换 pass 直接拒绝，`distributed.extern_call` 残留导致后面
`--emit-bytecode` 报 `Dialect 'distributed' not found`。

现在 `_dispatch` 统一按符号表打标（`_CORE_TYPE_BY_SYMBOL`，`dsa/ascend/distributed.py:41`），
**TLE 发出的所有 distributed op 都标 `vector`**，包括 barrier / sync。

为什么和 TDA 不一样（TDA 的 `libaclshmem_device.barrier_all` 标 `cube_and_vector`）：只要内核里
出现一个 `cube_and_vector` 的 op，BiShengIR 就会用 **mix** 模式编译，整个 body 被克隆进 AIC 和
AIV 两个函数，于是每个 distributed op 在每个 PE 上执行两次。实测（2 卡 / 单卡均可复现）：

| 内核形态 | `mix_mode` | 现象 |
|---|---|---|
| 无 barrier，只有 signal/wait | `aiv` | 正常：`inc` 计数 = world |
| barrier(cube_and_vector) + signal | `mix` | `add` 计数翻倍（2 卡：期望 3，实得 6） |
| barrier(cube_and_vector) + 等值 wait | `mix` | 计数跳过目标值，`wait-until-equal` 永久自旋，aicore 超时 |

TDA 能容忍这种重复执行，是因为它的内核显式按 core 角色编程（`global_vec_id`、按核数分摊工作量）；
TLE 的分布式子集没有这层约定，所以必须保证“单次发射”。

代价与边界：Ascend 上的 `tle.distributed_barrier` 只同步每个 PE 的 AIV 侧。如果内核里同时有 cube
计算，需要另外同步 AIC（`tle.dsa` 的 cube 同步），当前不支持在同一个 kernel 里混用 cube 计算与
TLE 分布式原语。

`_DEFAULT_EXTERN_CORE_TYPE` 同样取 `vector`，避免将来新增的 extern 调用走 `cube_or_vector` 又触发
mix。

### 5.2 `signal` / `signal_wait`（M1）

- `signal` → `distributed.notify`，槽地址 = `device_dptr + slot_id * 16`（i32）；
- `signal_wait` → `extern_call("aclshmem_wait_<dtype>")`。

**为什么 `signal_wait` 不用 `distributed.wait`**：`distributed.wait` 只通过返回的 token 暴露
顺序，而 greedy rewriter 会把“token 没人消费”的 wait 整条删掉——实测 TTIR 里
`distributed.wait` 转换后直接消失，等于没等。TLE 的 `signal_wait` 返回 `None`（与可移植实现
一致），因此 Ascend 版本改为副作用型 extern 调用，保证不被 DCE。这条结论对后续任何“只要同步、
不要返回值”的原语都适用。

### 5.3 `n_pes`（M0）

用 `create_get_num_ranks`（→ `aclshmem_n_pes`）。`dev_mem_ptr` 仅为签名兼容，Ascend 从
ACLSHMEM 运行时读 PE 数，不解引用。

### 5.4 `distributed_barrier` 收紧

之前 `group_kind` / `barrier_kind` / `order` / `index` 被静默忽略，`space` 只接受
`None`/`"device"`。现在：

- 接受 `None` / `"device"`（含 `"intra"`、`"intra_node"`）/ `"world"` → `aclshmem_barrier_all`；
- `"inter_node"` 以及任何非默认的 `group_kind` / `barrier_kind` / `order` / `index` /
  `context_id` / `memory_scope` → `NotImplementedError`，不再“降级成全组 barrier”。

`space="device"` 语义上要求同步 device 组；在单节点部署下 device 组等于整个 communicator，
因此映射成立，文档需保留这一前提。

---

## 6. 验证

### 6.1 新增回归用例

`python/test/tle/unit/test_tle_ascend_distributed_lowering.py`（6 passed）：

| 用例 | 断言 |
|---|---|
| `test_ring_exchange_lowers_without_distributed_leftovers` | TTIR 只有 `get_rank`/`symm_at`/`extern_call`；`distributed.core_type = "cube_and_vector"` 存在；转换后无残留 |
| `test_signal_and_n_pes_lower_to_aclshmem_symbols` | 出现 `aclshmem_n_pes` / `aclshmemx_signal_op` / `aclshmem_wait_int32`；slot 步长为 16 个 i32 元素 |
| `test_unsupported_forms_fail_instead_of_degrading` | `space="inter_node"`、`barrier_kind="arrive"`、`wait_kind="shadow"`、`signal(op="set")` 全部显式失败 |

用例只在 `FLAGTREE_BACKEND == "ascend"` 且构建带 `distributed` 插件时运行，CUDA CI 自动 skip。

### 6.2 端到端编译（本机 8 × Ascend950DT，仅编译不执行）

```bash
# barrier / shard_id / remote：不再需要 demo 里的 --fix-core-type
PYTHON=/tmp/venv-flagtree-td/bin/python bash documents/tle_lowering_demo/run_demo.sh
# -> asm keys 含 mlirbc / npubin，kernel.o 产出
```

实测结果（HEAD + 本次改动）：

| 检查点 | 结果 |
|---|---|
| `kernel.ttir.mlir` | `distributed.extern_call {distributed.core_type = "vector", symbol = "aclshmem_barrier_all"}` |
| `kernel.ttadapter.mlir` | 三个 op 全部变 `hivm.hir.custom`，无 `distributed.*` 残留 |
| `kernel.npuir.mlir` | 只有 `func.func @ring_exchange`（未拆 AIC/AIV），`aclshmem_barrier_all` 出现 1 次 |
| `kernel.o` | 产出，`strings` 可见 `aclshmem_ptr_float` |
| signal kernel | `kernel.o` 可见 `aclshmemx_signal_op`、`aclshmem_wait_int32` |

### 6.3 多卡运行（本机 8 × Ascend950DT，torchrun + ACLSHMEM）

运行用例：[`documents/tle_lowering_demo/multirank_probe.py`](tle_lowering_demo/multirank_probe.py)
（`torchrun --nproc-per-node=N multirank_probe.py`），每个 rank 校验六项：
`shard_id` 等于 torch rank、`n_pes` 等于 world、`barrier` 后 `remote(peer)` 读到对端分片、
`signal(op="inc")` 的计数等于 world、`signal(op="add", value=rank+1)` 的计数等于 world 之和。

| 规模 | 结果 |
|---|---|
| 2 卡 | 6/6 通过（每 rank） |
| 4 卡 | 6/6 通过（每 rank） |
| 6 卡 | 6/6 通过（每 rank） |
| 8 卡 | 6/6 通过（每 rank，机器空闲时） |

> 8 卡首次尝试失败（rank 6/7 在 `AclrtCreateStreamWithConfig` 报 driver error 507899），
> 排查后确认是**同机并发**导致：当时另有一个 8 卡 pytest 套件在跑。机器空闲后同样命令
> 直接全部通过。做多卡验证时务必独占设备。

2 卡实测输出（节选）：

```
[rank 0] PASS  shard_id got=0 want=0          [rank 1] PASS  shard_id got=1 want=1
[rank 0] PASS  n_pes    got=2 want=2          [rank 1] PASS  n_pes    got=2 want=2
[rank 0] PASS  remote read got=[1000..1007]   [rank 1] PASS  remote read got=[0..7]
[rank 0] PASS  signal inc sum got=2 want=2    [rank 0] PASS  signal add sum got=3 want=3
OVERALL: ALL PASS
```

未覆盖：`space="node"`（跨节点，M2 未实现）、多轮 launch 的 parity / 复用、`wait_kind="counter"`。
§5.2 的 `signal_wait` 顺序保证在运行期只验证了“计数正确 + 不挂死”，没有验证“wait 之后读到对端
最新数据”的可见性（TLE 侧还缺显式 `quiet/fence` 原语）。

### 6.4 回归对比

同一台机器、同一个 venv，只替换这 3 个 Python 文件，跑整个
`python/test/tle/unit`（排除本次新增用例）：

| 状态 | 结果 |
|---|---|
| 改动前（备份文件） | 14 failed / 129 passed / 77 skipped / 1 collection error |
| 改动后 | 14 failed / 129 passed / 77 skipped / 1 collection error |

完全一致，无回归。这 14 个失败是本机既有的 GPU / 多卡用例（例如 `test_tle_d2d_barrier`
失败于 `communication.py` 的 `NameError: name 'dist' is not defined`），与本次改动无关。

扩展 IR 矩阵（均通过）：i64 slot 的 `signal(op="add")` → `aclshmem_atomic_add_int64`、
`signal_wait` → `aclshmem_wait_int64`（步长 8 个 i64 元素）；`space="device"`/`"intra_node"`/
`"world"` 都落到 `aclshmem_barrier_all`；`uint64` slot 的 `signal_wait`、缺 `value` 的
`signal(op="add")` 均显式报错。

---

## 7. 路线图

| 阶段 | 内容 | 改动面 | 状态 |
|---|---|---|---|
| M0 | `core_type` 修复、`n_pes`、barrier 选项收紧 | Python | ✅ 本次 |
| M1 | `signal` / `signal_wait`（slot ABI 64B） | Python | ✅ 本次（IR 级验证） |
| M1b | team barrier（host 建 team、handle 作 kernel 参数）、`shard_id` 多轴 | Python + host | 待做 |
| M2 | `space="node"`：TTIR 保留 TLE op + pass 做 load/store 配对 → `putmem/getmem` | TLE 方言 + pass | 待做 |
| M3 | `reshard` 子集（[B]→[S]、[S]→[B]、[P]→[B]）前端展开 | Python | 待做 |
| M4 | `distributed_dot`、cluster 空间：明确 unsupported | 文档/白名单 | 待做 |

M2 之前，`tle.remote(space="node")` 应继续显式报错，不要退化成 `symm_at`——同节点指针语义
跨节点不成立。
