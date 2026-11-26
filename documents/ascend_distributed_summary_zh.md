# 华为 Ascend Distributed 调研与 FlagTree 集成总结

> 结论：在 FlagTree 的 Ascend 后端上打通了 Triton-distributed 的「前端 DSL → Distributed 方言 → HIVM → BiShengIR 设备模板 → ACLSHMEM 运行时」全链路；在 4 卡 Ascend 950DT 上 distributed API 套件 **3060 passed / 14 skipped / 0 failed**。

---

## 0. 10 分钟速读

### 做了什么

1. **把 Triton-distributed-ascend（TDA）接进 FlagTree**：新增 `TRITON_BUILD_TD` 开关，在 Ascend 后端内加入 Distributed 方言、`distributed → HIVM` 转换和 `distributed` Python 绑定，并内置 `triton_dist` Python 包、Ascend 教程和测试。
2. **打通编译链路**：Triton 前端用 `DistributedOpBuilder` 生成 `distributed.*` 方言算子，转换 pass 把它降成 `hivm.custom`，BiShengIR 再链接 ACLSHMEM 设备端包装 `_mlir_ciface_aclshmem_*`，最终由 `ccec` 生成昇腾设备二进制。
3. **打通运行链路**：`torchrun + HCCL` 负责进程组和控制面，`ACLSHMEM` 负责对称内存和单边通信数据面；基于 MTE / UDMA 的 AllGather、GEMM+ReduceScatter、MoE dispatch/combine 等 kernel 可以在 950DT 上跑通。

### 关键数字

| 项目 | 结果 |
|---|---|
| PR 规模（5 个 commit，裁剪后） | 169 files，+44,900 行 |
| `triton_dist` Python 包 | 约 2.5 万行 |
| Ascend 教程 / 用例 | 25 个示例入口 + 1 个共享 `utils.py` |
| API 测试文件 | 25 个，覆盖同步、RMA、原子、Team、传输后端、autotune |
| 最终验证 | 本机 8 × Ascend950DT，使用其中 4 卡：3060 passed / 14 skipped / 0 failed |
| 构建开关默认值 | `TRITON_BUILD_TD=OFF`，对现有后端零影响 |

### 最重要的 5 个认知

1. **Ascend 的“分布式”不是一条路，而是多层组合**：HCCL 管集合通信，ACLSHMEM 管单边通信，Triton Distributed 方言管“在 kernel 里写通信”，三者职责不同。
2. **ACLSHMEM 的编程模型是 OpenSHMEM 风格**：PE / Team / 对称堆 / put-get / signal-wait / quiet，但底层传输引擎换成了昇腾的 MTE、SDMA、UDMA、RoCE。
3. **真正难的是编译栈，不是 Python API**：要让一个 `tl.wait()` 最终变成设备指令，需要前端 builder、MLIR 方言、`distributed → HIVM` 转换、BiShengIR 模板、ACLSHMEM 设备包装五层同时对齐。
4. **版本 pin 是主要风险源**：`_mlir_ciface_aclshmem_*` undefined 的根因不是 FlagTree 代码，而是 AscendNPU-IR 的 pin 太旧、`Shmem.cpp` 缺设备包装。升级 pin 后才真正解决。
5. **环境问题和代码问题必须分开定位**：同一个“kernel 跑失败”，可能分别来自 Python import、IR 转换、设备符号链接、shmem 引擎配置、硬件能力五种层次。

### 当前状态

- ✅ 构建、IR 转换、分布式 API、两卡/四卡运行用例已打通。
- ⚠️ RoCE 集合通信用例未验证：当前安装的 shmem wheel 编译时 `RDMA Backend=NONE`，属于环境问题。
- ⚠️ AscendNPU-IR pin 在 `origin/stable` 的某个 commit 上，尚无 release tag，需关注可复现性。
- ⚠️ 目前以正确性验证为主，尚未做系统的性能对比（vs HCCL / torch 原生实现）。

---

## 1. 背景：为什么要做这件事

### 1.1 FlagTree 与 Ascend 后端

FlagTree 是一个多后端 Triton 发行版，Ascend 后端在本次工作之前已经完成了 Triton 3.6.x 的适配（#144 / #147 / #148 / #149），可以编译普通 Triton kernel。

但 **Triton-distributed 的 kernel 编译不了**：这类 kernel 里会出现 `dl.wait()`、`dl.notify()`、`dl.symm_at()` 等通信原语，它们不属于普通 Triton 方言，需要一整套额外的编译支持。

### 1.2 Triton-distributed 是什么

Triton-distributed 最初由字节跳动开源，核心目标是 **通信与计算融合（compute-communication fusion）**：

- 用 Triton 写 kernel，在 kernel 内部直接发起远端读写、信号等待、屏障等操作；
- 避免“先算完、再调集合通信库”的串行模式；
- 典型算子：AllGather+GEMM、GEMM+ReduceScatter、GEMM+AllReduce、MoE dispatch/combine、All-to-All 变体。

上游主要面向 NVIDIA（NVSHMEM）和 AMD（ROCSHMEM）。华为的 Ascend 适配版（Triton-distributed-ascend，TDA）把底层从 NVSHMEM 换成了 **ACLSHMEM**，编译后端换成 **HIVM / BiShengIR**。

### 1.3 本次目标

在 FlagTree 的 Ascend 后端里以 **opt-in** 方式集成 TDA：

- 不破坏现有 Ascend / 其他后端构建；
- 产出带 `distributed` 插件的 wheel；
- 能编译并运行 `triton_dist` 的 Ascend kernel；
- 建立可回归的 API 测试与教程用例。

---

## 2. 一张图看懂 Ascend Distributed 全景

```
┌─────────────────────────────── 应用层 ───────────────────────────────┐
│ PyTorch / torch_npu / 训练框架                                        │
│   ├─ 集合通信：HCCL（AllReduce / AllGather / ReduceScatter ...）      │
│   └─ 单边通信：ACLSHMEM（put / get / signal / barrier / atomic）      │
├─────────────────────────────── 算子层 ───────────────────────────────┤
│ Triton kernel（triton_dist.jit）                                      │
│   ├─ triton_dist.language：wait / notify / symm_at / rank / token     │
│   └─ libshmem_device：aclshmem_* / aclshmemx_* 设备端 extern call     │
├─────────────────────────────── 编译层 ───────────────────────────────┤
│ Triton 前端 → TTIR + Distributed 方言                                 │
│   → ConvertTritonDistributedToHIVM → HIVM（hivm.custom）              │
│   → BiShengIR / AscendNPU-IR → ccec → 昇腾设备二进制                  │
├─────────────────────────────── 运行层 ───────────────────────────────┤
│ CANN runtime / driver / 片间互联（HCCS/UB）/ RoCE / UB / HBM          │
└──────────────────────────────────────────────────────────────────────┘
```

可以把它理解成四层“分布式”：

| 层 | 组件 | 解决的问题 |
|---|---|---|
| 通信库层 | HCCL | 标准集合通信，host/stream 语义 |
| 单边通信层 | ACLSHMEM | 设备发起的远端读写、信号、屏障 |
| 编译器层 | Distributed 方言 + HIVM | 让 Triton 能表达并 lowering 通信原语 |
| 算子层 | `triton_dist.kernels` | 具体的融合算子实现 |

---

## 3. 调研篇：华为 Ascend Distributed 技术要点

> 本节内容整理自仓库内的 TDA 源码、AscendNPU-IR 的 SHMEM 子项目文档和测试代码。硬件参数属于参考值，精确值应以 `npu-smi` / CANN `platform_config` 为准。

### 3.1 硬件与执行模型

#### 本次验证的机器：Ascend950DT

本次验证跑在 **8 卡 Ascend950DT** 上（最终全量用其中 4 卡）。下面这些是本次在本机直接查到的值：

| 项目 | 实测值 | 来源 |
|---|---|---|
| 芯片 | `Ascend950DT_9582`（SoC version 260） | `torch.npu.get_device_properties(0)` |
| 卡数 | 8 | `npu-smi info` |
| AIC（Cube 核） | 32 / 卡 | `cube_core_num` |
| AIV（Vector 核） | 64 / 卡 | `vector_core_num` |
| HBM | 96 GB / 卡（98,515,812,352 B） | `total_memory` / `npu-smi` |
| L2 | 128 MB | `L2_cache_size = 134,217,728 B` |
| AI Core 频率 | 1.65 GHz | `npu-smi info -t common` |
| UB | 256 KB / core | 本仓 `ub_size_in_kbytes_for_arch()` |
| L0C | 256 KB | 本仓 `cv_autotune/hardware_consts.py` |
| npu-smi | 25.1.rc2 | `npu-smi info` |

> UB / L0C 的 256 KB 来自仓库里针对 `Ascend950` 的常量，不是 `npu-smi` 直接读出的字段；精确值以 CANN `platform_config` 为准。

#### AIC / AIV 分离架构

Ascend950DT 采用 **AIC（Cube 核）+ AIV（Vector 核）分离**：

- AIC 负责矩阵乘（Cube 单元）；
- AIV 负责向量计算和低时延通信；
- AIC 与 AIV 之间没有直连数据通道，交互经 L2 完成。

这带来两个直接影响：

1. 通信通常跑在 AIV 上，计算可能跑在 AIC 上，融合 kernel 要显式划分核的角色；
2. 编译时需要告诉后端“这段代码给 cube 还是 vector 执行”，也就是后文的 `core_type` 和 `core_ratio`。

本机 950DT 的核比例正好是 **32 AIC : 64 AIV = 1:2**，这也解释了 `core_ratio` 为什么支持 `1:2` 这一档。

#### 存储层次

| 存储 | 950DT（本次环境） | 作用 |
|---|---|---|
| GM / HBM | 96 GB | 对称堆、输入输出、signal |
| UB | 256 KB / core | DMA 中转、本地计算中间结果 |
| L1 | 512 KB | Cube 输入缓存 |
| L0A / L0B | 各 64 KB | Cube 操作数 |
| L0C | 256 KB | Cube 累加器 |
| L2 | 128 MB | 片上共享缓存 |

**UB 很小**，所以通过 UB 中转的 DMA 必须分块。本项目 AllGather 里使用 124 KB 的 `COPY_BLOCK`，属于相对保守的取值；950DT 的 UB 是 256 KB，实际分块可以在此基础上重新评估。

#### 互联拓扑

对本次 950DT 环境，能确认的是：

- UDMA 是 950 系列上的统一 DMA 引擎，kernel 里按 256 MB 上限分块；
- 跨节点 RDMA 需要显式指定后端（shmem 构建参数 `-rdma_backend`，例如 XSCALE）；
- **本机链路数、单链路带宽本次没有实测**，本文不写具体带宽数字。

设计上的结论：

- 单机内融合 kernel 常用 **mesh 模式**，让每个 rank 直接和其余 rank 通信；
- 跨节点才走 RoCE / RDMA，且需要 shmem 编译时启用对应后端。

### 3.2 软件栈：HCCL、ACLSHMEM、CANN 各管什么

| 组件 | 定位 | 典型用法 | 在本次工作中的角色 |
|---|---|---|---|
| **HCCL** | 华为集合通信库 | `torch.distributed` 的 `hccl` backend；AllReduce / AllGather / Barrier | 进程组初始化、rank 同步、golden 校验前 barrier |
| **ACLSHMEM** | 对称内存 / 单边通信库 | `shmem` Python 包；`aclshmem_init`、`aclshmem_create_tensor`、设备端 put/get/signal | 融合 kernel 的数据面 |
| **CANN** | 昇腾异构计算栈 | ACL runtime、`ccec`、`bishengir-compile` | 编译与运行基础 |
| **AscendNPU-IR** | 华为 MLIR 编译器 | HIVM 方言、BiShengIR 优化与代码生成 | 承接 `hivm.custom` 并生成设备代码 |

一个容易混淆的点：**ACLSHMEM 有自己的 bootstrap，不走 HCCL**。教程里同时存在两套控制面：

- `torch.distributed.init_process_group(backend="hccl")`：HCCL 进程组；
- `aclshmem_init(InitAttr(ip_port=...))`：ACLSHMEM 自己的 TCP bootstrap，由 `ASH_MASTER_ADDR` / `ASH_MASTER_PORT` 控制。

所以多任务并行时要显式设置不同的端口，否则会出现端口冲突或 `aclshmem_init` 返回 `-1`。

### 3.3 ACLSHMEM 编程模型

ACLSHMEM 的设计基本遵循 **OpenSHMEM**：

| 概念 | 含义 | 对应实现 |
|---|---|---|
| **PE** | 通信参与者，通常等于 rank | `aclshmem_my_pe()` / `aclshmem_n_pes()` |
| **Team** | 通信域 / 子分组 | `ACLSHMEM_TEAM_WORLD`、`team_split_strided`、`team_split_2d` |
| **Symmetric Heap** | 所有 PE 按相同规则分配的对称内存 | `aclshmem_malloc` / `aclshmem_create_tensor` |
| **RMA** | 远端内存访问 | put / get / putmem / getmem |
| **AMO** | 远端原子操作 | add / inc / and / or / xor / set / swap / CAS |
| **Signal** | 信号变量 | set / add + wait_until |
| **Quiet / Fence** | 完成与顺序保证 | `aclshmem_quiet` / 引擎专用 quiet / fence |

#### 初始化

Python 侧的核心参数：

```python
attributes = ash.InitAttr()
attributes.my_rank = rank
attributes.n_ranks = world_size
attributes.local_mem_size = local_mem_size   # 对称堆大小
attributes.ip_port = "tcp://127.0.0.1:8666"  # bootstrap 地址
attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE  # 或 UDMA
ash.aclshmem_init(attributes)
```

几个约束：

- 对称堆按 **2 MB 页** 分配；
- 单 PE 对称堆上限 **40 GB**（`ACLSHMEM_MAX_LOCAL_SIZE`）；
- 所有 PE 必须按 **相同顺序、相同大小** 分配和释放；否则远端地址会错位，而且不一定立刻报错；
- 教程里会先 `set_conf_store_tls(False)`，避免本地测试环境被 TLS 配置卡住。

#### 对称地址与远端访问

- `aclshmem_ptr(local_ptr, pe)` 把本地对称地址换算成远端 PE 的对应地址；
- Triton 侧对应 `dl.symm_at(ptr, rank)`，最终 lowering 成 `aclshmem_ptr_<type>`；
- **拿到远端地址不等于可以随便读写**：跨 PE 传输必须走 ACLSHMEM 的 put/get/engine 接口，不能直接把远端地址交给普通 `DataCopy`。这是 SHMEM 文档里反复强调的边界。

#### 数据搬运接口

ACLSHMEM 的接口大致分三层：

| 层次 | 例子 | 特点 |
|---|---|---|
| 标准 / 高阶 | `aclshmem_putmem`、`aclshmem_getmem`、`aclshmem_put_signal` | 屏蔽引擎细节，使用默认 buffer，不支持并发 |
| 低阶 / 引擎专用 | `aclshmemx_mte_put_nbi`、`aclshmemx_udma_*`、`aclshmemx_roce_*` | 显式选引擎、buffer、sync id，支持并发 |
| Host 侧 | `aclshmem_put` / `aclshmem_get` / stream 版本 | CPU 发起，排入 stream |

`*_nbi` 是 **非阻塞**：调用返回不代表数据已经到达，必须配合 `quiet` / `fence` / barrier / signal 才能保证完成。

### 3.4 四种数据搬运引擎

| 引擎 | 发起方 / 位置 | 典型通路 | 适用场景 | 注意事项 |
|---|---|---|---|---|
| **MTE** | AICore 内 | GM ↔ UB ↔ GM | 单机 P2P、低时延、小到中等数据 | 需要 UB 中转；950DT 的 UB 为 256 KB/core，需按块搬运 |
| **SDMA** | Host / 系统 DMA | H2D / D2H / D2D | 大块连续拷贝 | 需要预留 UB buffer，完成后用 `sdma_quiet` |
| **UDMA** | 设备侧统一 DMA | GM ↔ GM | Ascend 950 上的高带宽单边通信 | 仅 950 系列；支持 QP；本项目的 AllGather 使用了它 |
| **RDMA / RoCE** | 网卡 / RDMA 引擎 | 跨节点 GM ↔ GM | scale-out 集群 | 需要 shmem 启用 RDMA 后端；同一 PE 目标上避免无序并发 RMA/AMO |

选择建议：

- 单机、追求低时延 → MTE；
- 950 上追求高带宽、少占核 → UDMA；
- 跨节点 → RoCE；
- Host 侧大块搬运 → SDMA。

### 3.5 同步原语与典型模式

#### 基本原语

| 原语 | 作用 | 备注 |
|---|---|---|
| `aclshmemx_signal_op` | 更新远端 signal 变量 | 支持 SET / ADD |
| `aclshmem_signal_wait_until` | 等待本地 signal 满足条件 | 比较符 EQ/NE/GT/GE/LT/LE |
| `aclshmem_put_signal` | 搬运数据并更新 signal | 适合“数据到达后通知” |
| `aclshmem_barrier_all` / `team_sync` | 集合同步 | 保证之前的相关操作完成 |
| `aclshmem_quiet` | 等待本 PE 已发出的全引擎 RMA 完成 | MTE-only 时优先用 `aclshmemx_mte_quiet` |
| `aclshmem_fence` | 顺序栅栏 | 当前实现与 quiet 接近 |

一个容易踩的语义差异：

- **`sync` 只保证此前 memory store 的完成与可见，不保证未完成的 RMA 全部完成**；
- 进入集合通信前，如果有未完成的 UDMA 操作，应调用 `quiet_lowlevel(peer, transport=UDMA)`；
- SHMEM 没有专门的 UDMA sync 入口，因此显式指定 `transport=UDMA` 的 sync 仍会被拒绝。

#### 典型通信模式：signal + token

一个典型的 producer/consumer 模式：

```text
producer:  putmem_nbi(data)  →  notify(signal, SET/ADD)
consumer:  wait(signal, value) → consume_token(data_ptr) → load → quiet → notify(ack)
```

其中 `consume_token` 很关键：

- 它在运行时几乎是恒等函数（wrapper 直接 `return value`）；
- 它的作用是 **在 IR 里建立数据依赖**，让编译器不能把后续的 load 重排到 wait 之前；
- 在 Triton 前端里它表现为 `distributed.consume_token`，最终 lowering 成 `aclshmem_consume_token_<type>`。

#### 复用 signal 的 parity 技巧

信号变量跨多次 launch 不会自动清零。如果每次 launch 前都 `memset` 一整个 signal buffer，开销很大。常见做法是 **奇偶翻转**：

```text
第 1 次 launch：wait ack=0 → set ready=1 → wait ready=1 → set ack=1
第 2 次 launch：wait ack=1 → set ready=0 → wait ready=0 → set ack=0
```

本项目 AllGather kernel 用的就是这个模式，`launch_parity` 在 host 侧每次 launch 翻转一次。

### 3.6 编译栈：Triton → HIVM → BiShengIR

这是整个集成中最值得理解的部分。

#### 前端：Distributed 方言

Triton-distributed 在 TTIR 之上引入了一组 `distributed.*` 方言算子（见 `DistributedOps.td`）：

| 算子 | 语义 |
|---|---|
| `distributed.wait` | 等待 signal 达到指定值，返回 token |
| `distributed.consume_token` | 把 token 依赖挂到数据上 |
| `distributed.get_rank` / `get_num_ranks` | 获取 PE 编号 / 数量 |
| `distributed.symm_at` | 获取指定 rank 的对称地址 |
| `distributed.notify` | 更新指定 rank 的 signal |
| `distributed.extern_call` | 调用外部设备库函数（ACLSHMEM） |

属性：

- `SignalOp`：SET / ADD；
- `CommScope`：GPU / INTRA_NODE / INTER_NODE。

#### 中端：`distributed → HIVM`

Ascend 的转换 pass 做两件事：

1. 把 `distributed.core_ratio`（如 `"1:2"`）转成 `hivm.core_ratio`，挂到函数上；
2. 把每个 `distributed.*` 算子改写成 `hivm.custom`。

`hivm.custom` 的名字统一是 `dist.<symbol>`，并带上这些属性：

| 属性 | 作用 |
|---|---|
| `hivm.is_distributed` | 标记这是分布式算子 |
| `symbol` | 真实设备符号名，如 `aclshmem_sync_all` |
| `core_type` | 指定 cube / vector / cube_and_vector / cube_or_vector |
| `pipe` | 默认 `PIPE_S`（标量流水） |
| `VFMode` | `SIMD` |
| `gm_addr_args_indices` | 哪些操作数是 GM 地址，供后端做地址 lowering |
| `already_sub_block_bound` | 集合类操作必须所有 sub-block 参与，禁止被绑定到单个 sub-block |

#### 符号映射表

这是理解“一个 Python API 最终变成什么设备调用”的关键：

| 前端 | 设备符号 |
|---|---|
| `get_rank` | `aclshmem_my_pe` |
| `get_num_ranks` | `aclshmem_n_pes` |
| `symm_at` | `aclshmem_ptr_<type>` |
| `wait` | `aclshmem_wait_<type>` |
| `consume_token` | `aclshmem_consume_token_<type>` |
| `notify`（i32 signal） | `aclshmemx_signal_op` |
| `notify`（i64/uint64，SET） | `aclshmem_int64_p` / `aclshmem_uint64_p` |
| `notify`（i64/uint64，ADD） | `aclshmem_atomic_add_int64` / `aclshmem_atomic_add_uint64` |
| `extern_call` | 由 `libaclshmem_device` 指定的符号，如 `aclshmem_sync_all` |

#### 后端：BiShengIR 与设备包装

`hivm.custom` 里的符号最终不是直接调用 C 函数，而是调用 BiShengIR 提供的 **`_mlir_ciface_*` 包装**：

```text
hivm.custom "dist.aclshmem_sync_all"
        ↓
_mlir_ciface_aclshmem_sync_all(void)
        ↓
aclshmem_sync_all()
```

包装层存在的原因：

- MLIR 侧用 memref 描述指针和形状，而 ACLSHMEM 的 C API 用裸指针；
- 包装函数负责把 `memref.aligned + memref.offset` 还原成指针；
- 这层 ABI 由 BiShengIR 统一生成，Triton 侧不需要关心。

`--build-shmem-template` 会把 ACLSHMEM 的设备端包装编进模板 bitcode（`meta_op.aic/aiv/mix.aic/mix.aiv` 等）。**如果模板里没有这些符号，kernel 链接就会报 `_mlir_ciface_aclshmem_*` undefined**——这正是本次工作中最关键的一个坑。

#### core_type 与 core_ratio

Ascend 的 kernel 可能是：

- 纯 Vector（AIV）；
- 纯 Cube（AIC）；
- Mix（AIC + AIV）。

因此需要两类信息：

| 信息 | 位置 | 取值 |
|---|---|---|
| `core_type` | 每个 `extern_call` 上 | `cube` / `vector` / `cube_and_vector` / `cube_or_vector` |
| `core_ratio` | 模块属性，转成 `hivm.core_ratio` | `0:1`、`1:0`、`1:1`、`1:2` |

限制：

- `0:1`（纯 vector）不能用于含 `tl.dot` 的 kernel；
- `core_ratio` 会进入 JIT cache key，避免同一份 Python 源码用不同比例时误复用旧二进制。

### 3.7 与 NVIDIA NVSHMEM 的对照

两边的**编程模型几乎一样**，差别在“底座”：

| 维度 | NVIDIA | Ascend |
|---|---|---|
| 单边通信库 | NVSHMEM | ACLSHMEM |
| 对称堆 | `nvshmem_malloc` | `aclshmem_malloc` / `aclshmem_create_tensor` |
| 远端地址 | `nvshmem_ptr` | `aclshmem_ptr` / `symm_at` |
| 传输引擎 | IBGDA / NVLink / PCIe | MTE / SDMA / UDMA / RoCE |
| 设备库 | `libnvshmem_device.bc` | BiShengIR Shmem 模板 + `_mlir_ciface_*` 包装 |
| 编译后端 | TritonGPU / NVGPU | HIVM / BiShengIR |
| 同步 | signal / wait / quiet | signal / wait / quiet / team sync |
| 执行单元 | warp / CTA | AIC / AIV + `core_ratio` + sub-block |
| 启动 | `torchrun` + NVSHMEM init | `torchrun` HCCL + shmem TCP bootstrap |

所以可以这样理解：**TDA 复用了 Triton-distributed 的上层 DSL 和算子设计，只替换了“单边通信库 + 编译后端 + 执行模型”三块。**

---

## 4. 集成篇：FlagTree 里是怎么落地的

### 4.1 调用链总览

```text
@triton_dist.jit kernel
  └─ triton_dist.language.wait / notify / symm_at / ...
       └─ triton._C.libtriton.distributed.ir（DistributedOpBuilder）
            └─ TTIR + distributed 方言
                 └─ [Ascend 插件] ConvertTritonDistributedToHIVM
                      └─ HIVM：hivm.custom "dist.aclshmem_*"
                           └─ BiShengIR + Shmem 设备模板
                                └─ _mlir_ciface_aclshmem_*
                                     └─ ACLSHMEM → MTE / UDMA / RoCE
```

### 4.2 构建开关与产物

| 变量 | 默认 | 作用 |
|---|---|---|
| `TRITON_BUILD_TD` | `OFF` | 构建 `distributed` 插件，打包 `triton_dist` |
| `TRITON_BUILD_NPUIR` | `OFF`（推荐 `ON`） | 构建并打包 BiShengIR（含 `--build-shmem-template`） |
| `TRITON_BUILD_PROTON` | `ON` | Ascend 下与 FlagPrism 互斥，本次设为 `OFF` |
| `TRITON_BUILD_FLAGPRISM` | `ON` | Ascend 默认启用的 profiler/debugger |
| `MAX_JOBS` | 默认 | 并行编译数 |

打开 `TRITON_BUILD_TD` 后：

1. `TRITON_PLUGIN_NAMES` 增加 `"distributed"`；
2. 在 Ascend 后端目录内编译 Distributed 方言与转换：`include|lib/Dialect/Distributed`、`include|lib/TritonDistributedToHIVM`、TableGen；
3. `TritonDistributed` 插件只保留 Python IR 绑定（`python/src/dist`），转换 pass 的 Python wrapper 由 `triton_ascend.cc` 注册到 `TritonAscend` 插件；`TRITON_USE_ASCEND` 宏用于在绑定代码里区分 Ascend 路径；
4. `setup.py` 把 `triton_dist.*` 加入 wheel 包列表；
5. 若同时开 `TRITON_BUILD_NPUIR`，BiShengIR 会被复制到 `triton/backends/ascend/bishengir/`，并校验模板文件是否齐全。

### 4.3 前端：DistributedOpBuilder 与 Python API

`triton_dist.language` 暴露的核心接口：

```python
from triton_dist.language import wait, notify, rank, num_ranks, symm_at, consume_token
```

Ascend 的差异点：

- `notify` 的 signal 指针允许 `int32`、`int64`、`uint64`（CUDA 侧要求 64 位）；
- 64 位 signal 的 `sig_op="add"` **只在 Ascend 950 及更新设备上支持**；
- `comm_scope` 映射到 `GPU / INTRA_NODE / INTER_NODE`。

前端 builder 的挂载位置也很关键：

- 共享文件 `python/triton/compiler/code_generator.py` 保持基线原样（与其他后端共享，不做 Ascend 专用修改）；
- Ascend 专用逻辑放在 `third_party/ascend/spec/triton/compiler/code_generator.py`，通过 `spec_path()` 覆盖共享模块；
- 这样不会污染其他后端的代码路径（这是 review 后专门修正的一处）。

### 4.4 中端：Distributed → HIVM

转换 pass 的核心逻辑：

1. 扫描模块里有没有 `tl.dot` / `tl.dot_scaled`，决定是不是 mix kernel；
2. 校验 `distributed.core_ratio` 是否合法（`0:1`、`1:0`、`1:1`、`1:2`）；
3. 把 ratio 写成函数上的 `hivm.core_ratio`；
4. 用一组 pattern 把 `distributed.*` 改写成 `hivm.custom`；
5. 对 `extern_call` 读取 `distributed.core_type`，决定算子落在 cube 还是 vector；
6. 对集合类符号（`aclshmem_barrier_*`、`aclshmem_sync*`）加 `already_sub_block_bound`，避免被 NPU-IR 的 sub-block binding pass 包成单核执行。

### 4.5 后端：ACLSHMEM 绑定与匹配

`python/triton_dist/language/extra/ascend/libaclshmem_device.py` 做了几件事：

- 用 `@core.extern` 把 ACLSHMEM API 暴露成 Triton 可调用的 extern；
- 每个 API 通过 `_mark_core_type()` 打上 `distributed.core_type`；
- 维护 dtype 矩阵，例如：
  - RMA 支持 fp16/fp32/int8/16/32/64/uint8/16/32/64/bf16；
  - 原子操作主要支持 int32/uint32/int64/uint64，950 上增加 fp32；
  - 950 之前的 `atomic_add` 支持更有限的类型；
- 用 `ACLSHMEMTransport`（AUTO / UDMA / ROCE / MTE）在编译期选择传输后端；
- 对 pointer type 打 hash 补丁，让它能作为 extern 签名字典的 key（一个很隐蔽的 Python 层坑）。

运行时获取外部库的路径时，Ascend 分支返回：

```python
extern_libs = {"libshmem": ""}
```

也就是 **不依赖外部 bitcode 文件**，符号由 BiShengIR 模板提供。这与 CUDA 的 `libnvshmem_device.bc` 路径不同，是 Ascend 集成的关键差异之一。

### 4.6 运行时：HCCL bootstrap + ACLSHMEM 数据面

典型启动流程：

```text
torchrun（每 rank 一个进程）
  ├─ torch_npu.set_device(rank)
  ├─ dist.init_process_group("hccl")         # 控制面
  ├─ init_aclshmem(rank, world_size, ...)    # ACLSHMEM 自己的 TCP bootstrap
  ├─ ash.aclshmem_create_tensor(...)         # 对称堆
  └─ kernel[...](...)                         # 编译并 launch
```

注意点：

- `ASH_MASTER_ADDR` / `ASH_MASTER_PORT` 与 HCCL 的 `MASTER_ADDR` / `MASTER_PORT` 是两套；
- `MASTER_ADDR=localhost` 可能解析到 `::1`，导致 `aclshmem_init` 失败，显式设 `127.0.0.1` 更稳；
- 多任务并行时要给每个任务分配不同的 `ASH_MASTER_PORT`；
- 测试框架用 `mp.spawn`，每个 worker 自己初始化 HCCL 和 ACLSHMEM。

### 4.7 Kernel 走读：UDMA AllGather

以 `python/triton_dist/kernels/ascend/allgather.py` 为例，看看一个真实的融合 kernel 怎么用这些原语。

**目标**：每个 rank 有一块 `m × hidden_dim` 的本地数据，AllGather 后得到 `world_size × m × hidden_dim`。

**分工**：

- 偶数 AIV 核：
  - 前 `world_size - 1` 个核做 producer，每个负责一个远端 rank；
  - 其余核做本地数据拷贝。
- 奇数 AIV 核：做 receiver，等待远端数据到达后拷贝到 output。

**协议**：

```text
producer:
  for each remote rank:
    wait(ack, parity)
    putmem(local_tile → remote_workspace, max 256MB/chunk)
    notify(ready, parity ^ 1)

receiver:
  wait(ready, parity ^ 1)
  consume_token(remote_ptr)
  copy remote chunk → output
  quiet()
  notify(ack, parity ^ 1)
```

**几个设计点**：

| 设计 | 原因 |
|---|---|
| 每 256 MB 分块 | UDMA 单次传输上限 |
| signal 按 64 B 对齐、8 个 int64 一个 slot | 一个 cache line，减少伪共享 |
| parity 翻转 | 避免每次 launch 清零 signal buffer |
| 每个 receiver 核单独 ack | producer 必须等所有核都拷完才能覆盖 workspace |
| `consume_token` | 建立 wait → load 的数据依赖，防止重排 |
| 把 token 相关指针链放在 `scf.for` 外 | 规避 Ascend IR 转换里的 dominance 校验问题 |

这个 kernel 基本把 Ascend distributed 的关键要素都用上了：对称内存、UDMA、signal、token、分核、parity。

### 4.8 用例与测试矩阵

**教程（25 个示例入口，另有 1 个共享 `utils.py`）**：

- 分布式 GEMM：AllGather+GEMM、GEMM+ReduceScatter、GEMM+AllReduce（oneshot / notify / UDMA）；
- All-to-All：AllToAllV-GMM、MoE dispatch/combine、QKV all2all、transpose all2all、reverse all2all；
- 融合算子：Reverse A2A + RMSNorm + MatMul；
- 每个算子通常有 MTE 和 UDMA 两个版本。

**API 测试（25 个文件）**：

| 类别 | 覆盖 |
|---|---|
| PE / rank | `my_pe`、`num_ranks`、`symm_at` |
| 同步 | `wait_notify`、`signal_op`、`barrier_ops`、`sync_ops`、`team_pe` |
| RMA | `put_get_mem`、transport dispatch / runtime、relay |
| 原子 | `atomic_ops`、atomic transport |
| 传输后端 | AUTO / MTE / UDMA，RoCE dispatch/runtime |
| 编译 | external call core types、gm addr indices、core_ratio |
| 其他 | autotune、swizzle algorithm、transport init |

测试组织方式：

- `pytest ... -m dist` 只跑标记为 `dist` 的用例；
- 多卡用例由 `run_dist_test()` 自动 spawn worker，不需要外部 `torchrun`；
- UDMA 用例只在 950 系列上跑；
- RoCE / relay 用例默认不纳入 `dist`，需要显式环境变量。

---

## 5. 踩坑与解决（按 commit）

### 5.1 `b93c38934`：从 0 到 1 集成

**问题**：Ascend 后端没有 Distributed 方言、没有 `distributed → HIVM` 转换、没有前端 builder，`triton_dist` kernel 直接编译失败。

**做法**：

- 接入 TDA 的 C++ 子集（方言、转换、pybind；现已折叠进 Ascend 后端的 `include/`、`lib/`、`python/src/dist/`）；
- 新增 `TRITON_BUILD_TD`，把插件编进 `libtriton`；
- 前端接入 `DistributedOpBuilder`；
- 打包 `triton_dist` Python 包和教程；
- 修复并行构建时 TritonGPUIR 的 TableGen 依赖竞争。

**验证**：两卡 950DT 上 AllGather 和 MoE dispatch 示例通过。

### 5.2 `4ca54a5f8`：裁剪非 Ascend C++ 代码

**问题**：TDA 子集带着 NVIDIA/AMD/METAX 的 LLVM/TritonGPU 转换、SIMT 方言等，Ascend 构建根本不编译，但目录很大、维护困难。

**做法**：

- 删除非 Ascend 的转换、SIMT 方言、非 Ascend pybind 入口；
- 简化父 CMake；
- `python/triton_dist` 删除 NVIDIA/AMD 的 kernels、layers、function、models、benchmark。

**结果**：Ascend 构建仍产出 7 个分布式编译对象；IR 测试 11 passed；两卡 AllGather 10/10、MoE dispatch 10/10。

**代价**：这个只保留 Ascend 路径的 TDA 子集不再能构建 NVIDIA/AMD 路径；其他后端应使用上游 triton-distributed。

### 5.3 `4848c2b40`：清理 Python 残留 + 修复 spawn import

**问题 1**：`common_ops.py` 在 CUDA/HIP 之外 import 时直接抛异常；profiler、MetaX 等模块 Ascend 根本用不到。

**问题 2**（更隐蔽）：从源码树跑 pytest 时，pytest 的 prepend import mode 会把仓库的 `python/` 加进 `sys.path`；`mp.spawn` 出来的 worker 继承了这个路径，于是 import 到 **未构建的源码 triton**，报：

```text
cannot import name 'getenv' from triton._C.libtriton
```

**做法**：

- 删除不可达/不可导入的模块，同步裁 `setup.py`；
- `run_dist_test()` 在 spawn 期间临时把源码 `python/` 从 `sys.path` 移除，spawn 完再恢复。

**结果**：`test_wait_notify` 4/4、`test_symm_at`、`test_signal_op` 等通过。

### 5.4 `0543cf984`：把 shared code_generator 的补丁收回

**问题**：review 指出共享文件 `python/triton/compiler/code_generator.py` 被改成了创建 `DistributedOpBuilder`，但这是 Ascend-only 逻辑。

**做法**：

- 共享文件恢复成基线原样；
- Ascend 逻辑保留在 `third_party/ascend/spec/triton/compiler/code_generator.py`，通过 spec overlay 生效。

**意义**：多后端项目里，**共享文件要保持干净，平台差异放 spec 层**。

### 5.5 `921e9a21e`：升级 AscendNPU-IR pin，解决 undefined symbol

**现象**：

```text
undefined symbol: _mlir_ciface_aclshmem_sync_all
undefined symbol: _mlir_ciface_aclshmem_team_pe_mapping
undefined symbol: _mlir_ciface_aclshmemx_udma_put_nbi_int32
```

**根因**：

- FlagTree pin 的 AscendNPU-IR 是 `release/v2.0.x@efb714158`，其 `Shmem.cpp` 缺少 TDA 需要的设备端包装；
- 该版本 `--build-shmem-template` 产出的模板里也没有这些符号；
- 所以 kernel 编译能过，链接时找不到 `_mlir_ciface_*`。

**解决**：

1. 把 AscendNPU-IR pin 升到 `origin/stable@8bc48fb585`，其 `Shmem.cpp` 包含 `aclshmem_sync_all`、`team_pe_mapping` 等包装；
2. 新版本要求把 `BiShengIRTensorViewDialect`、`MLIRAVEDialect` 链进 `triton-opt` / `triton-mlir-opt`；
3. `triton_dist_key()` 只在独立 `libtriton_distributed` 存在时才哈希它（Ascend 把绑定链进 `libtriton.so`）；
4. 支持 `ASCENDNPUIR_BUILD_ROOT` / `ASCENDNPUIR_REUSE_BUILD`，让 BiShengIR 构建树可以放在源码树外并复用。

**验证**：

- 模板里能查到 `_mlir_ciface_aclshmem_sync_all` 和 `_mlir_ciface_aclshmem_team_pe_mapping`；
- distributed API 套件 3060 passed / 14 skipped / 0 failed；
- `test_sync_ops` 19/19、`test_team_pe` 4/4、`test_wait_notify` 4/4。

### 5.6 问题定位方法论

这次工作最值钱的不是某一个 fix，而是定位顺序：

```text
1. Python 层：import 对了吗？是不是装到了旧 wheel？
2. 前端层：IR 里有没有 distributed.* 方言？
3. 中端层：有没有 lowering 成 hivm.custom？symbol 对不对？
4. 链接层：nm / 链接错误里缺哪个 _mlir_ciface_*？
5. 模板层：BiShengIR 的 Shmem.cpp / template bitcode 里有没有这个符号？
6. 环境层：shmem wheel 的 transport 后端开了吗？硬件支持吗？
```

只要按这个顺序，绝大多数“kernel 跑不起来”都能快速归位。

---

## 6. 验证结果

### 6.1 环境

| 项 | 值 |
|---|---|
| 硬件 | 本机 8 × Ascend950DT，最终全量用其中 4 卡；早期 2 卡 |
| 芯片 | `Ascend950DT_9582`，32 AIC / 64 AIV，96 GB HBM，L2 128 MB，1.65 GHz |
| npu-smi | 25.1.rc2 |
| CANN | 9.2.1 |
| torch_npu | 2.10.0.post7 |
| ACLSHMEM | 1.7.0 |
| Python | 3.12 |
| 构建 | `FLAGTREE_BACKEND=ascend`、`TRITON_BUILD_TD=1`、`TRITON_BUILD_NPUIR=ON`、`TRITON_BUILD_PROTON=OFF` |

### 6.2 结果

| 项目 | 结果 |
|---|---|
| Ascend 构建 | 成功，`libtriton.so` 包含 `distributed` 子模块 |
| IR 转换用例 | 11 passed |
| distributed API 套件（逐文件，本机 4 × 950DT） | **3060 passed / 14 skipped / 0 failed** |
| `test_sync_ops` | 19/19 |
| `test_team_pe` | 4/4 |
| `test_wait_notify` | 4/4 |
| `test_atomic_add_ci` | 通过（补 mix 模板后） |
| 两卡 AllGather / MoE dispatch | 10/10 / 10/10 |

### 6.3 已知限制

| 项 | 状态 | 说明 |
|---|---|---|
| RoCE 集合通信运行用例 | 未通过 | 安装的 shmem wheel 编译时 `RDMA Backend=NONE`，属环境问题；用例不在 `dist` 标记内 |
| Relay RMA | 默认关闭 | 需要 `-enable_relay` 的 shmem + `BISHENGIR_SHMEM_RELAY_SUPPORT=ON` |
| 64 位 signal 原子 add | 仅 Ascend950 支持 | 受硬件/模板能力限制 |
| 性能 | 未系统测量 | 当前以正确性为主 |
| AscendNPU-IR pin | `origin/stable` commit | 无 release tag 包含 |

> 早期一轮曾出现 4 个 64 位 signal / 原子操作用例超时（`error code 507034`）；最终一轮 4 卡全量结果中未再出现这 4 个失败。

---

## 7. 如何构建与运行（复现指南）

### 7.1 环境准备

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install -r python/requirements.txt
python3 -c "import shmem, torch_npu; print('shmem/torch_npu OK')"
npu-smi info
```

### 7.2 构建 wheel

```bash
cd /home/cdy/FlagTree

export FLAGTREE_BACKEND=ascend
export TRITON_BUILD_TD=1
export TRITON_BUILD_NPUIR=ON
export TRITON_BUILD_PROTON=OFF
# export TRITON_BUILD_FLAGPRISM=OFF
# export MAX_JOBS=$(nproc)

python3 -m pip wheel . --no-build-isolation --no-deps -w dist/
```

产物：`dist/flagtree-<ver>+ascend.git<hash>-cp3xx-cp3xx-linux_aarch64.whl`。

首次打开 `TRITON_BUILD_NPUIR` 会初始化 `llvm-project` / `torch-mlir` 子模块并完整构建 BiShengIR，耗时较长且需要网络。可用：

```bash
export ASCENDNPUIR_BUILD_ROOT=/path/to/big/disk/build
export ASCENDNPUIR_REUSE_BUILD=1   # 增量复用，注意源码变更时可能残留旧对象
```

### 7.3 运行教程

```bash
# 两卡 AllGather + GEMM（UDMA）
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=21000 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/allgather-gemm-udma.py
```

### 7.4 运行 API 测试

```bash
pytest python/triton_dist/test/ascend/api -m dist
```

只有 2 张卡时可以先跑：

```bash
pytest python/triton_dist/test/ascend/api/test_wait_notify.py -m dist
```

### 7.5 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `aclshmem_init` 返回 `-1` | `MASTER_ADDR=localhost` 解析到 `::1`；设 `ASH_MASTER_ADDR=127.0.0.1` |
| 两个任务端口冲突 | 显式设置不同的 `ASH_MASTER_PORT` |
| `_mlir_ciface_aclshmem_*` undefined | BiShengIR 模板缺符号：开 `TRITON_BUILD_NPUIR=ON`，或自建 AscendNPU-IR 并把 `build/bin` 加进 `PATH` |
| `'builder' object has no attribute 'create_distributed_wait'` | wheel 不是从当前代码树构建的，缺 `DistributedOpBuilder`；重新出包 |
| `ModuleNotFoundError: utils` | 没在仓库源码树运行，或缺 `tutorials/ascend` |
| UDMA 用例被 skip | 非 Ascend 950 硬件 |
| `import triton_dist` 得到空 namespace | site-packages 里残留不完整 `triton_dist/`；在干净 venv 安装 wheel |
| spawn worker import 错 triton | 确认 `run_dist_test()` 的 `sys.path` 清理生效，或从已安装 wheel 运行 |

---

## 8. 风险、限制与后续工作

| 优先级 | 事项 | 说明 |
|---|---|---|
| 高 | AscendNPU-IR pin 固定到 release tag | 当前是 `origin/stable` 的 commit，可复现性依赖网络和分支状态 |
| 高 | RoCE 后端验证 | 需要 shmem wheel 启用 RDMA Backend，并补跑跨节点用例 |
| 高 | 性能基准 | 与 HCCL / torch 原生实现对比，给出带宽、时延、融合收益 |
| 中 | CI 覆盖 | 当前依赖本地 4 卡环境；建议至少 nightly 跑 `-m dist` |
| 中 | 文档链接修复 | `api/README_zh.md` 里引用的 `docs_ascend/` 在内置后不存在 |
| 中 | Signal buffer 规模 | AllGather 的 ack 数量随 `chunks × ranks × cores` 增长，大规模 world 需要评估内存 |
| 中 | UDMA sync 语义 | SHMEM 无专用 UDMA sync，集合前需 `quiet_lowlevel`；文档化 |
| 低 | Relay 支持 | 需要专用 shmem 构建和 4 卡环境 |
| 低 | 上游同步策略 | 内置 TDA 子集与上游的 diff 维护成本；建议定期对齐 |
| 低 | 其他后端 | 本树已裁剪 NVIDIA/AMD 路径，其他后端请用上游 triton-distributed |

---

## 9. 团队可复用的经验

1. **先分层，再定位**：Python / 前端 IR / 中端 lowering / 链接符号 / 设备模板 / 运行时环境，六层逐层排除，比盲目改代码快得多。
2. **工具链版本要当代码管理**：一个 pin 的 commit 就能决定设备符号是否存在。构建脚本里应校验版本、校验模板文件，而不是假设“环境是对的”。
3. **共享文件不碰，平台差异放 spec 层**：`code_generator.py` 的教训很典型。
4. **分布式测试要能逐文件隔离**：全量跑一次很贵，先跑 `test_wait_notify`、`test_sync_ops` 这类小集合，再逐文件扩大。
5. **先对齐上游，再裁剪**：vendoring 时保留完整目录便于 diff；确认 Ascend 路径稳定后再删非 Ascend 代码，并且每一步都用测试兜底。
6. **环境问题不要当代码问题修**：RoCE 的 6 个失败、UDMA skip、IPv6 解析，都是环境/配置问题；代码侧不要为了“让测试变绿”加 workaround。
7. **写清楚失败现象和符号名**：`_mlir_ciface_aclshmem_*` 这种信息比“kernel 跑挂了”有价值一百倍。

---

## 附录 A：关键文件地图

### 编译器与插件

| 路径 | 作用 |
|---|---|
| `third_party/ascend/include/Dialect/Distributed/IR/` | Distributed 方言定义（Ops / Attrs / Dialect） |
| `third_party/ascend/lib/Dialect/Distributed/IR/` | 方言实现 |
| `third_party/ascend/include/TritonDistributedToHIVM/` | `distributed → HIVM` 转换的声明与 TableGen |
| `third_party/ascend/lib/TritonDistributedToHIVM/` | `distributed → HIVM` 转换实现 |
| `third_party/ascend/python/src/dist/` | pybind：`ir.cc`、`triton_distributed.cc` |
| `third_party/ascend/triton_ascend.cc` | 转换 pass wrapper 注册（`TritonAscend` 插件） |
| `third_party/ascend/CMakeLists.txt` | TD 插件构建、Distributed 子目录接入、BiShengIR 链接 |
| `cmake/FlagTreeOptions.cmake` | `TRITON_PLUGIN_NAMES`、插件注册 |
| `CMakeLists.txt` | `TRITON_BUILD_TD` 选项 |
| `setup.py` | `triton_dist.*` 包列表 |
| `python/setup_tools/utils/__init__.py` | AscendNPU-IR pin |
| `third_party/ascend/build/packaging.py` | BiShengIR 构建、打包、模板校验 |

> 以上路径按当前目录重构后的布局（`dc31c8383`）；Distributed 方言和转换已折叠进 Ascend 后端的 `include/` / `lib/`，不再有独立的 `third_party/ascend/Triton-distributed-ascend/` 目录。

### Python 包

| 路径 | 作用 |
|---|---|
| `python/triton_dist/language/distributed_ops.py` | `wait` / `notify` / `rank` / `num_ranks` / `symm_at` / `consume_token` |
| `python/triton_dist/language/extra/libshmem_device.py` | 跨后端 SHMEM 代理（NVSHMEM / ROCSHMEM / ACLSHMEM） |
| `python/triton_dist/language/extra/ascend/libaclshmem_device.py` | ACLSHMEM extern 绑定、core_type、dtype 矩阵 |
| `python/triton_dist/language/extra/ascend/aclshmem_constants.py` | `ACLSHMEMAtomicOp` / `CmpOp` / `SignalOp` / `Team` / `Transport` |
| `python/triton_dist/jit.py` | `triton_dist.jit`、`core_ratio`、extern libs、cache key |
| `python/triton_dist/utils.py` | 平台检测、`is_ascend()`、`run_dist_test()` |
| `python/triton_dist/kernels/ascend/` | AllGather、AllReduce、AllToAll、Ulysses 等 Ascend kernel |
| `python/triton_dist/test/ascend/` | 精度/性能测试入口 |
| `python/triton_dist/test/ascend/api/` | API 测试 |

### 教程与文档

| 路径 | 作用 |
|---|---|
| `tutorials/ascend/` | 25 个 Ascend 分布式算子示例 + 共享 `utils.py` |
| `tutorials/ascend/README_zh.md` | 教程索引与运行方式 |
| `python/triton_dist/test/ascend/README_zh.md` | 精度/性能测试说明 |
| `python/triton_dist/test/ascend/api/README_zh.md` | API 测试说明 |
| `compile.md` | 构建与测试完整指南 |
| `third_party/ascend/AscendNPU-IR/bishengir/lib/Template/lib/Shmem/Shmem.cpp` | ACLSHMEM 设备端 `_mlir_ciface_*` 包装 |
| `third_party/ascend/AscendNPU-IR/third-party/shmem/` | ACLSHMEM 源码、文档、示例 |

---

## 附录 B：术语表

| 术语 | 含义 |
|---|---|
| **PE** | Processing Element，SHMEM 的通信参与者，通常等于 rank |
| **Team** | 通信域 / 子分组，`ACLSHMEM_TEAM_WORLD` 是默认全局域 |
| **Symmetric Heap** | 所有 PE 按相同规则分配的对称内存 |
| **RMA** | Remote Memory Access，put / get 等单边访问 |
| **AMO** | Atomic Memory Operation，远端原子操作 |
| **NBI** | Non-Blocking Immediate，非阻塞调用，需 quiet / fence 保证完成 |
| **Quiet** | 等待本 PE 之前发出的 RMA 完成 |
| **Fence** | 内存顺序栅栏 |
| **AIC / AIV** | Cube 核 / Vector 核 |
| **UB** | Unified Buffer，片上缓存；950DT 为 256 KB/core |
| **GM / HBM** | 全局内存 / 高带宽内存 |
| **MTE** | Memory Transfer Engine，AICore 上的搬运引擎 |
| **SDMA** | System DMA，系统级 DMA |
| **UDMA** | Unified DMA，950 系列上的统一 DMA 引擎 |
| **RoCE** | RDMA over Converged Ethernet，跨节点 RDMA |
| **HCCS / UB** | 昇腾单机卡间互联总线；不同代际使用 HCCS 或 UB 协议 |
| **HIVM** | Ascend 硬件 IR 方言 |
| **BiShengIR** | 华为基于 MLIR 的昇腾编译器 |
| **CANN** | 昇腾异构计算软件栈 |
| **HCCL** | 华为集合通信库 |
| **ACLSHMEM** | 华为对称内存 / 单边通信库，OpenSHMEM 风格 |
| **core_ratio** | AIC : AIV 的核比例，如 `1:2` |
| **core_type** | 某个分布式算子由 cube / vector / mix 执行 |

---

## 附录 C：参考资料

仓库内材料（本次调研的主要依据）：

1. `compile.md`：Ascend + Triton-distributed 编译与测试指南。
2. `tutorials/ascend/README_zh.md`：25 个算子示例及运行方式。
3. `python/triton_dist/test/ascend/README_zh.md`：精度/性能测试、`@prof` 计时。
4. `python/triton_dist/test/ascend/api/README_zh.md`：API 测试矩阵与传输后端说明。
5. `python/triton_dist/language/extra/ascend/`：ACLSHMEM 绑定、常量、dtype 矩阵。
6. `third_party/ascend/{include,lib}/Dialect/Distributed/`、`third_party/ascend/{include,lib}/TritonDistributedToHIVM/`、`third_party/ascend/python/src/dist/`：Distributed 方言、转换、pybind。
7. `third_party/ascend/AscendNPU-IR/bishengir/lib/Template/lib/Shmem/Shmem.cpp`：设备端包装。
8. `third_party/ascend/AscendNPU-IR/third-party/shmem/docs/glossary.md`：SHMEM 术语表。
9. `third_party/ascend/AscendNPU-IR/third-party/shmem/docs/api/pythonAPI.md`：Python API。
10. `third_party/ascend/AscendNPU-IR/third-party/shmem/.agents/skills/shmem-ops-design/references/hardware-architecture.md`：昇腾硬件与 SHMEM 约束。

外部通用背景：OpenSHMEM 编程模型（PE / Team / symmetric memory / RMA / AMO / signal）。

> 说明：本文件中的硬件参数和性能数字来自仓库内文档整理，属于参考值；精确参数请以 `npu-smi`、CANN `platform_config` 和官方文档为准。
