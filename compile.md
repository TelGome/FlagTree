# Ascend + Triton-distributed 编译与测试

本文以 FlagTree 仓库根目录为工作目录，目标是产出带 Triton-distributed 支持的 Ascend wheel，并跑分布式测例。

已验证环境：Ascend 950DT × 8（最终 4 rank 实测，本机 32 AIC / 64 AIV、96 GB HBM）、CANN 9.2.1、torch_npu 2.10.0.post7、ACLSHMEM(shmem) 1.7.0、Python 3.12。

## 1. 环境准备

```bash
cd /home/cdy/FlagTree

# CANN 环境（ccec / llvm-link / bishengir 等）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 构建期 Python 依赖
python3 -m pip install -r python/requirements.txt

# 运行期依赖（如需多卡测试）
# - torch / torch_npu
# - ACLSHMEM 的 shmem wheel（提供 aclshmem_init / OpEngineType.UDMA）
python3 -c "import shmem, torch_npu; print('shmem/torch_npu OK')"
npu-smi info
```

## 2. 构建参数

| 变量 | 值 | 作用 |
|---|---|---|
| `FLAGTREE_BACKEND` | `ascend` | 选择 Ascend 后端 |
| `TRITON_BUILD_TD` | `1` | 构建 TritonDistributed 插件（`libtriton` 的 `distributed` 子模块）和 Ascend 后端内的 Distributed 方言/转换，并把 `triton_dist` 打进 wheel |
| `TRITON_BUILD_NPUIR` | `ON` | **推荐**：构建并打包 BiShengIR（含 `--build-shmem-template`），提供 ACLSHMEM 设备端模板库 |
| `TRITON_BUILD_PROTON` | `OFF` | Ascend 下 Proton 与 FlagPrism 互斥 |
| `TRITON_BUILD_FLAGPRISM` | 默认 `ON` | Ascend 默认启用；不想带 profiler/debugger 时设为 `OFF` |
| `MAX_JOBS` | 例如 `$(nproc)` | 并行编译数 |

说明：

- `TRITON_BUILD_NPUIR=ON` 时，`third_party/ascend/build/packaging.py` 会构建 AscendNPU-IR 的 BiShengIR，并复制到 wheel 的
  `triton/backends/ascend/bishengir/`。首次构建会初始化 `llvm-project`/`torch-mlir` 子模块，耗时较长且需要网络。
- 不开 `TRITON_BUILD_NPUIR` 时，CANN 自带的 `bishengir-compile` 缺少 ACLSHMEM 设备模板，链接 kernel 会报
  `unresolved symbol _mlir_ciface_aclshmem_*`。此时必须自建 AscendNPU-IR 并把 `build/bin` 加进 `PATH`（见 2.1）。
- `TRITON_BUILD_TD=OFF`（默认）时，不构建 `distributed` 子模块，也不打包 `triton_dist`，其余后端行为不变。

### 2.1 不使用 TRITON_BUILD_NPUIR 时的替代方案

```bash
cd third_party/ascend/AscendNPU-IR
git submodule update --init --recursive
bash build-tools/build.sh --build-type=Release -o build -t -j 32 \
  --build-triton --build-torch-mlir --build-shmem-template \
  --bishengir-publish ON --collect-binary build/bishengir-output
export PATH=$PWD/build/bin:$PATH   # 每个编译 kernel 的 shell 都要带
```

## 3. 出包

```bash
cd /home/cdy/FlagTree

export FLAGTREE_BACKEND=ascend
export TRITON_BUILD_TD=1
export TRITON_BUILD_PROTON=OFF
export TRITON_BUILD_NPUIR=ON
# export TRITON_BUILD_FLAGPRISM=OFF
# export MAX_JOBS=$(nproc)

# 产出 dist/flagtree-<ver>+ascend.git<hash>-cp3xx-cp3xx-linux_aarch64.whl
python3 -m pip wheel . --no-build-isolation --no-deps -w dist/

# 或直接安装到当前环境（会替换现有 triton）
# python3 -m pip install . --no-build-isolation -v
```

### 3.1 增量开发构建

已有 `build/cmake.linux-aarch64-cpython-3.12` 时，修改 C++ 后可以增量编译：

```bash
ninja -C build/cmake.linux-aarch64-cpython-3.12          # 或 make all
python3 setup.py build_py                                # 同步 Python 侧到 build/lib.*
```

### 3.2 安装与自检

```bash
# 建议独立 venv，避免覆盖系统已有 triton / triton_ascend
python3 -m venv --system-site-packages /opt/venv-ascend-td
/opt/venv-ascend-td/bin/pip install --no-deps --force-reinstall dist/flagtree-*ascend*.whl

/opt/venv-ascend-td/bin/python - <<'EOF'
import triton, triton_dist
from triton._C.libtriton import distributed
print(triton.__file__)
print("ascend_passes:", hasattr(distributed, "ascend_passes"))
print("DistributedOpBuilder:", hasattr(distributed.ir, "DistributedOpBuilder"))
print("triton_dist:", triton_dist.__file__)
EOF
```

期望输出 `ascend_passes: True`、`DistributedOpBuilder: True`。

## 4. 分布式测例

> 必须从**仓库源码树**运行：`python/triton_dist/test/ascend/prof.py` 用 `parents[4]/tutorials/ascend`
> 定位共享的 aclshmem bootstrap（`utils.init_aclshmem`），该目录不随 wheel 安装。

公共环境变量：

```bash
cd /home/cdy/FlagTree
export ASCEND_RT_VISIBLE_DEVICES=0,1   # 按实际空闲卡调整
export ASH_MASTER_ADDR=127.0.0.1       # 本机 localhost 可能解析到 ::1，不设会 aclshmem_init 失败
```

### 4.1 API / IR 测试（pytest）

```bash
# 全部带 dist 标记的分布式用例（会起多进程，需要 NPU）
python3 -m pytest python/triton_dist/test/ascend/api -m dist
# 注：pytest 默认 prepend 模式会把仓库 python/ 目录加进 sys.path，spawn 出的
# 子进程会误 import 未编译的源码树 triton（报 `cannot import name 'getenv'
# from triton._C.libtriton`）。api 的 conftest 已在 spawn 期间临时摘除该路径；
# 如果绕过 conftest 直接跑，请加 `--import-mode=importlib`。

# 纯 IR 级、不需要 NPU：验证 distributed.* -> hivm.hir.custom 转换
python3 -m pytest python/triton_dist/test/ascend/api/test_gm_addr_args_indices.py \
                  python/triton_dist/test/ascend/api/test_external_call_op_core_types.py

# 单文件
python3 -m pytest python/triton_dist/test/ascend/api/test_wait_notify.py -m dist
```

### 4.2 多卡 harness（`python/triton_dist/test/ascend`）

```bash
cd /home/cdy/FlagTree/python/triton_dist/test/ascend

# 整个目录（自动识别 950 打开 UDMA，自动分配 31xxx 端口）
bash run.sh --mode=verify --case=random

# 单独跑
ASH_MASTER_PORT=31007 torchrun --standalone --nproc_per_node=2 \
  allgather.py --mode=verify --case=random
ASH_MASTER_PORT=31008 torchrun --standalone --nproc_per_node=2 \
  moe-dispatch-udma.py --mode=verify --case=random
```

- `--mode=verify`：只校验精度；`--mode=both`：精度 + NPU Event 计时
- `--case=random`：最多抽 10 组；`--case=full`：全量
- `NPROC_PER_NODE=4/8`：扩卡；`SKIP_UDMA=1`：非 950 时跳过 UDMA 用例

### 4.3 Tutorials（`tutorials/ascend`）

```bash
cd /home/cdy/FlagTree
bash tutorials/ascend/run.sh          # 全部教程，端口 21xxx

# 单个
ASH_MASTER_PORT=21000 torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/allgather-gemm.py
```

### 4.4 C++ / lit（可选）

```bash
make test-lit
make test-cpp
```

## 5. 架构相关代码说明

### 5.1 参与 Ascend 构建的分布式 C++ 代码

目录已经重构：Distributed 方言和 `distributed → HIVM` 转换不再放在独立的
`third_party/ascend/Triton-distributed-ascend/` 下，而是折叠进 Ascend 后端自己的
`include/` / `lib/` 布局。`TRITON_BUILD_TD=ON` 时编译：

- Distributed 方言定义：`third_party/ascend/include/Dialect/Distributed/IR/`
- Distributed 方言实现：`third_party/ascend/lib/Dialect/Distributed/IR/`
- `distributed → HIVM` 转换声明 / TableGen：`third_party/ascend/include/TritonDistributedToHIVM/`
- `distributed → HIVM` 转换实现：`third_party/ascend/lib/TritonDistributedToHIVM/`
- Python IR 绑定：`third_party/ascend/python/src/dist/{triton_distributed.cc, ir.cc}`

插件职责也做了拆分：

- `TritonDistributed` 插件只保留 Python IR 绑定，链接 `DistributedIR`，不再承担 BiShengIR 的
  TableGen 依赖；
- 转换 pass 的 Python wrapper 由 `third_party/ascend/triton_ascend.cc` 注册到 `TritonAscend`
  插件，因为 `TritonAscend` 已经链接了 BiShengIR 的 HIVM 方言；
- `triton_distributed.cc` 通过 `init_triton_distributed_ascend_passes` 把
  `distributed.ascend_passes` 子模块暴露出来。

`TRITON_USE_ASCEND` 宏仍保留，用于在绑定代码里区分 Ascend 路径（例如 `ir.cc` 中非 Ascend 的
SIMT 绑定被 `#ifndef TRITON_USE_ASCEND` 排除）。

当前目录只保留 Ascend 路径；NVIDIA/AMD 的 `TritonDistributedToLLVM`、
`TritonDistributedToTritonGPU`、SIMT 方言等已删除。需要使用这些后端的场景请改用上游
triton-distributed。

### 5.2 `python/triton_dist` 的当前范围

`python/triton_dist` 现在只保留 Ascend 路径和共享分发/工具：

- `kernels/ascend/`：AllGather、AllReduce、AllToAll、Ulysses 等 Ascend kernel
- `language/extra/ascend/`：ACLSHMEM 绑定、常量、dtype 矩阵
- `test/ascend/`：精度 / 性能测试与 API 测试
- `tools/`、`language/`、`utils.py`：共享分发、JIT、编译工具

`utils.py` 中的 `is_cuda() / is_hip() / is_maca()` 分支仍保留，用于维持与上游接口的兼容；
但 NVIDIA/AMD 的 kernels、layers、function、models、benchmark 已经删除，`setup.py` 也只打包
Ascend 与共享包（`triton_dist.kernels.ascend`、`triton_dist.language.extra.ascend` 等）。

结论：Ascend 运行期不依赖非 Ascend 后端代码；其他后端请使用上游 triton-distributed。

## 6. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `aclshmem_init` 返回 `-1` | `MASTER_ADDR=localhost` 解析到 `::1`；设 `ASH_MASTER_ADDR=127.0.0.1` |
| 两个任务端口冲突 | 显式设置不同的 `ASH_MASTER_PORT` |
| `_mlir_ciface_aclshmem_*` undefined | 使用的 BiShengIR 没有 ACLSHMEM 设备模板：开 `TRITON_BUILD_NPUIR=ON`，或自建 AscendNPU-IR 后把 `build/bin` 加入 `PATH` |
| `'builder' object has no attribute 'create_distributed_wait'` | wheel 不是从当前代码树构建的（缺少 `DistributedOpBuilder` 前端补丁）；重新出包 |
| `ModuleNotFoundError: utils` / `load_tutorial` 找不到文件 | 未在仓库源码树运行，或缺 `tutorials/ascend` |
| UDMA 用例被 skip | 非 Ascend 950 硬件；`run.sh` 依据 `npu-smi` 的 SoC 名判断 |
| `import triton_dist` 得到空 namespace 包 | site-packages 里残留了不完整的 `triton_dist/`；用 wheel 在干净 venv 中安装 |

## 7. 本轮验证结果与已知问题

验证时间：2026-11-20；环境：本机 8 × Ascend950DT（最终使用 4 卡）、CANN 9.2.1、torch_npu 2.10.0.post7、ACLSHMEM 1.7.0。

| 项目 | 结果 |
|---|---|
| 增量构建（`python3 setup.py build_py`，复用 `build/cmake.linux-aarch64-cpython-3.12`） | 成功；`libtriton.so` 更新，`distributed.ascend_passes` / `DistributedOpBuilder` 可用 |
| IR 级转换用例（`test_gm_addr_args_indices` + `test_external_call_op_core_types`） | **11 passed** |
| distributed API 套件（逐文件，4 × Ascend950DT） | **3060 passed / 14 skipped / 0 failed** |
| `test_sync_ops` | 19/19 |
| `test_team_pe` | 4/4 |
| `test_wait_notify` | 4/4 |
| 两卡 AllGather / MoE dispatch | 10/10 / 10/10 |

### 7.1 已知限制

- **RoCE 集合通信运行用例（6 个）失败**：当前安装的 shmem wheel 编译时 `RDMA Backend=NONE`，属于环境问题；这些用例不在 `dist` 标记内。
- **Relay RMA 默认关闭**：需要 `-enable_relay` 的 shmem 和 `BISHENGIR_SHMEM_RELAY_SUPPORT=ON` 的编译器模板。
- **64 位 signal 原子 add 仅 Ascend950 支持**：受硬件/模板能力限制。
- 早期一轮出现的 4 个 64 位 signal / 原子操作用例超时（`AclrtSynchronizeStreamWithTimeout ... error code 507034`）在最终 4 卡全量中未再出现。

### 7.2 版本依赖（已解决）

当前 pin：AscendNPU-IR `8bc48fb585`（`origin/stable`）。其
`bishengir/lib/Template/lib/Shmem/Shmem.cpp` 已包含 TDA 需要的设备端包装，例如：

- `_mlir_ciface_aclshmem_sync_all`
- `_mlir_ciface_aclshmem_team_pe_mapping`
- `_mlir_ciface_aclshmemx_udma_*`

使用 `TRITON_BUILD_NPUIR=ON` 重新构建时，`--build-shmem-template` 会生成 aic / aiv / mix.aic / mix.aiv
四份模板并随 wheel 打包。

历史问题：旧 pin `release/v2.0.x@efb714158`（shmem v1.6.1）的 `Shmem.cpp` 缺少上述包装，链接 kernel
会报 `unresolved symbol _mlir_ciface_aclshmem_*`；升级 pin 后已解决。

可选环境变量：

- `ASCENDNPUIR_BUILD_ROOT`：把 BiShengIR 构建树放到源码树外，避免占满源码盘；
- `ASCENDNPUIR_REUSE_BUILD=1`：复用已有构建树做增量构建，源码变更时注意可能残留旧对象。
