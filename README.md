# DeepSeek-V4-Flash 在 T-Head PPU 上的 vLLM 推理性能优化 — 项目报告

DeepSeek-V4-Flash（W8A8-INT8）在 T-Head PPU（真武 ZW810E ×8）上，用**开源社区技术栈**（官方 vLLM 0.24.0 + FlagGems + vllm-plugin-FL）替代厂商定制栈的推理性能优化项目。

代码改动以 PR 形式提交在两个配套仓库：

| 仓库 | PR 分支 | 内容 |
|---|---|---|
| [ziyuhu214/vllm-plugin-FL](https://github.com/ziyuhu214/vllm-plugin-FL/pull/1) | `feat/deepseek-v4-flash-ppu-perf` | DeepSeek-V4 PPU op 移植、int8 GEMM/MoE 路径、CUDA graph 支持等（6 个提交） |
| [ziyuhu214/FlagGems](https://github.com/ziyuhu214/FlagGems/pull/1) | `fix/deepseek-v4-ppu-fixes` | fp8 编码、sqlite 锁、int64 偏移等内核修复（5 个提交） |

工程细节（完整修改清单、算子归属图、性能里程碑、诊断存档）另见 [`PORTING_REPORT.md`](PORTING_REPORT.md)。

> ⚠️ 部分代码参考 T-Head 厂商 vLLM fork（0.20.1+ppu），从中移植，如公开发布请确认版权问题。

## 1. 软硬件环境

### 1.1 PPU 环境（本项目）

| 项 | 值 |
|---|---|
| 硬件 | T-Head PPU 真武 ZW810E × 8（16 卡机使用卡 0–7），单卡 96GB HBM2e，SM 8.0 兼容，64 SM，L2 64MB |
| 单卡算力 | BF16 峰值 123 TFLOPS；INT8 = BF16×2 = 246 TOPS |
| 模型 | DeepSeek-V4-Flash-0731，W-INT8 per-channel / A-INT8 per-token 量化（43 层 MoE 256 专家，MLA sparse 注意力） |
| 厂商基线栈 | T-Head fork vLLM 0.20.1+ppu（PPUInt8ScaledMMLinearKernel + FlashMLA sparse，闭源 C++ 融合内核） |
| 社区优化栈 | vLLM 0.24.0（官方，empty platform 构建，**本体零修改**）+ FlagGems v5.3.4 + vllm-plugin-FL（flagos-ai main）+ PPU deep_gemm / acext / flash_mla wheel |
| 部署参数 | TP=8，max-model-len 65536，fp8 KV cache（fp8_ds_mla），util 0.82（短输入可 0.85），batch 32768，无 prefix caching，`VLLM_FL_DSV4_TORCH_COMPILE=1` + FULL_AND_PIECEWISE cudagraph |

### 1.2 H100 对照环境

| 项 | 值 |
|---|---|
| 硬件 | NVIDIA H100 80GB × 8，CUDA 13.0，Driver 580.105.08 |
| 单卡算力 | 官方口径（SXM，dense）：FP8 1979 TFLOPS / BF16 989 TFLOPS |
| 软件 | vllm/vllm-openai:v0.26.0 官方镜像 |
| 模型 | DeepSeek-V4-Flash 原版（FP8 权重） |
| 部署参数 | TP=8，fp8 KV cache，block-size 256，无 prefix caching |

## 2. 性能结果

压测口径统一：vLLM bench serve，random 数据集，ignore-eos，decode 1024，并发 64，256 条请求，多轮取均值。

### 2.1 vs T-Head 厂商栈（同硬件，衡量软件栈完成度）

| 用例 (prefill/decode/conc) | T-Head 栈 tok/s | 社区栈（本项目最终） | 比例 | 社区栈 TPOT | 社区栈 TTFT |
|---|---|---|---|---|---|
| 1024/1024/64 | 1719.8 | **1301** | 76% | 44.0 ms | 5.3 s |
| 4096/1024/64 | 1161.4 | **702** | 60% | 75.8 ms | 15.8 s |
| 16384/1024/64 | 480.6 | **334** | 70% | 159.2 ms | 32.8 s |

优化演进（case1）：初次跑通 706 tok/s → 最终优化 1301 tok/s（**+84%**）；TPOT 82.9 → 44.0 ms。
数据：`results/raw_runs_20260831_145409.csv`、`results/logs/bench_cases23_final.log`、基线 `results/summary_20260826_111911.csv`。

### 2.2 vs H100（跨硬件，衡量硬件代际差）

H100 数据（8×H100 80GB，vLLM 0.26.0，FP8 模型）：

| 用例 | H100 tok/s | PPU 社区栈 tok/s | PPU/H100 | H100 TPOT | PPU TPOT |
|---|---|---|---|---|---|
| 1024/1024/64 | 3328.9 | 1301 | 39% | 18.3 ms | 44.0 ms |
| 4096/1024/64 | 2530.5 | 702 | 28% | 23.4 ms | 75.8 ms |
| 16384/1024/64 | 1272.4 | 334 | 26% | 44.2 ms | 159.2 ms |
| 65536/1024/64 | 392.5 | —（未测） | — | 138.3 ms | — |

### 2.3 单位算力效率（tok/s per TOPS，8 卡合计算力为分母）

统一口径：两边都取 **BF16 峰值 ×2** 作为 INT8/FP8 等效算力——PPU 123 TFLOPS ×2 = 246 TOPS/卡（8 卡 1968T），H100 SXM BF16 989.5 TFLOPS ×2 = 1979 TFLOPS/卡（8 卡 15832T，即官方 FP8 dense 峰值）。与两边部署的主 GEMM 精度（PPU W8A8-INT8 / H100 FP8）对应。

| 用例 | PPU tok/s/TOPS | H100 tok/s/TFLOPS | PPU/H100 效率比 |
|---|---|---|---|
| 1024/1024/64 | 0.661 | 0.210 | **3.1×** |
| 4096/1024/64 | 0.357 | 0.160 | 2.2× |
| 16384/1024/64 | 0.170 | 0.080 | 2.1× |

调优后 PPU 每单位标称算力的实际 token 产出约为 H100 的 2–3 倍，调优前为约1.7倍，**优化效果明显**。

## 3. 优化内容

改动全部落在 vllm-plugin-FL 与 FlagGems（vLLM 0.24.0 本体未改动），共 11 个提交。

### 3.1 vllm-plugin-FL（[PR #1](https://github.com/ziyuhu214/vllm-plugin-FL/pull/1)，6 提交）

1. **移植 PPU deep_gemm int8 GEMM/MoE 路径**（`ppu_deep_gemm*.py` + 调优配置）：厂商包装层移植，MoE 与 dense 层直连 PPU deep_gemm wheel；用 `tuning/tune_dsv4_deepgemm.py` 对本部署全部 GEMM 形状自动调优，产出 DSv4-tp8 专属 153 条配置（decode M 1–1024 + prefill M 1536–32768）。
2. **DeepSeek-V4 专用 op 与运行时补丁**（15 个补丁 + 8 个 op 文件）：int8 sparse-attn indexer（deep_gemm `int8_(paged_)mqa_logits` 替代 NVIDIA-only 路径）、fp8_ds_mla KV cache compress/insert/gather、o_proj int8、mHC opaque op 化、`_C` 算子 triton/FlagGems 桥接、torch.compile 全图模式（`VLLM_FL_DSV4_TORCH_COMPILE=1`）。
3. **acext INT8 W8A8 scaled-MM 线性内核**：厂商 kernel 类移植，行主序权重直入 `acext.int8_gemm`，thead 平台内核 oracle 首选。
4. **thead 启用 CUDA graph + 显存核算修复**：`support_static_graph_mode` 白名单、估算门控 `is_cuda()`→`is_cuda_alike()`、BreakableCUDAGraphWrapper 纳入 profiling、修复 allocator 缓存虚增 ~90 GiB 的估算 bug。单项收益最大（decode 从全 eager 到全图）。
5. **qnorm 插入头填充快速路径**：padding 头槽位（FlashMLA 64 头要求 vs TP8 实际 8 头）跳过 RMSNorm/RoPE。
6. benchmark 脚本对齐。

### 3.2 FlagGems（[PR #1](https://github.com/ziyuhu214/FlagGems/pull/1)，5 提交）

1. **软件 E4M3FN 编码器**：PPU Triton 无 `fp8e4nv` 类型，纯位运算 RNE 实现（含 subnormal），替换 fp8 KV cache 全部 7 处存储点。
2. **sqlite 调优缓存 busy timeout** 5s→120s：修复 TP=8 并发调优 `database is locked` 崩溃。
3. **int64 scale 偏移**（正确性修复）：`cp_gather_indexer_quant_cache` int32 溢出，KV cache >8K 块时 IMA 崩溃。
4. **qnorm kernel `num_real_heads` 快速路径**（与 plugin 第 5 项配套）。
5. **empty_kernel 依赖警告注释**：见 §4 结论 1。

## 4. Profiling 结果与剩余优化方向

稳态 profile（91% GPU 占用，decode 53.6ms/step，与 TPOT 实测吻合）。摘要存档 `results/profiles/profiler_out_0.txt`。

### 4.1 Decode 每步 GPU 时间分解

| 算子 | ms/step | 占比 | 与 T-Head 对比判断 |
|---|---|---|---|
| deep_gemm int8 GEMM（MoE up/down） | 6.6 | 12.4% | 同款内核，无差距 |
| deep_gemm batched_gemvt（dense 小批量） | 5.3 | 9.9% | 同款，无差距 |
| flash sparse decode attention | 4.7 | 8.8% | 同款 wheel，无差距 |
| **empty_kernel（3007 次/步）** | 3.4 | 6.3% | **纯 FL 栈损耗**（结论 1） |
| aiu cutlass GEMM（acext 线性层） | ~3.0 | 5.6% | 同款，无差距 |
| hc_prenorm GEMM（mHC 内 tilelang） | 2.7+1.3 | 7.4% | 同款内核 |
| 通信（twoShot+RING） | 3.8 | 7.2% | pccl 相同；T-Head 或有编译版 custom allreduce 略优 |
| mhc_pre/post tilelang | 3.0 | 5.6% | 同款内核，但**调用次数 ×2**（结论 3） |
| copy_kernel（649 次/步） | 1.2 | 2.3% | opaque op 边界拷贝，FL 栈损耗 |
| per_token_quant_int8（368 次/步） | 1.1 | 2.1% | T-Head 用 C++ 融合量化并进 GEMM 前处理，我们是独立 kernel |

### 4.2 核心结论：差距不在「大算子」，在「粘合部分」

重计算算子（GEMM / attention / MoE，合计约 60%）与 T-Head 完全同源，**这部分没有差距**。剩余差距全部来自 FL 栈的粘合层：

1. **empty_kernel 6.3%**：FlagGems 把 `torch.empty` 换成了「写零」triton kernel（每步 3007 次启动）。直接去掉会触发 compressor IMA——下游代码隐式依赖这个非标准的清零行为。这是真实的 FlagGems 上游问题（empty 语义被污染），修复需先审计依赖方；已提交带警告注释的 commit 存档（FlagGems `8d3179b0`）。
2. **小 kernel 启动过多**：每步 5000+ 次 kernel 启动（empty 3007 + copy 649 + quant 368 + mHC 260 + …），T-Head 的 C++ 融合算子把这些全部合并。profile 给出精确损耗：**约 6–8ms/step，即 TPOT 差距 11ms 的大部分**。突破需编译 C++ 扩展或等 PPU inductor 工具链成熟。
3. **mHC 重复计算（可修，预期收回 ~1ms/step）**：每步 87 次 mhc_pre + 86 次 mhc_post——fused_post_pre 版本没被用上，opaque 包装走了拆开路径。

其余遗留（KV cache 24 vs 37 GiB、compile 模式长输入 OOM 需 util 0.82、首启慢等）见 `PORTING_REPORT.md` 第六、七节。

## 5. 复现步骤

```bash
# 1. 下载 int8 量化模型（ModelScope）
bash scripts/download_dsv4_int8.sh

# 2. 安装社区栈：
#    - vllm-0.24.0（官方源码，empty platform 构建）
#    - FlagGems @ fix/deepseek-v4-ppu-fixes 分支
#    - vllm-plugin-FL @ feat/deepseek-v4-flash-ppu-perf 分支
#    - PPU 版 torch/triton/deep_gemm/flash_mla/acext wheel（随厂商容器提供）

# 3.（可选）重新调优 deep_gemm 配置
python tuning/tune_dsv4_deepgemm.py

# 4. 启动服务（VLLM_FL_DSV4_TORCH_COMPILE=1 + FULL_AND_PIECEWISE，util 0.82）
bash scripts/serve_dsv4_fl_bench.sh

# 5. 压测：case1 用 vllm-plugin-FL 仓库内 benchmarks/benchmark_throughput_serve.py，
#    case2/3（4096/16384）用 scripts/rerun_cases23.sh
```

开关：`VLLM_FL_DISABLE_DEEPGEMM_MOE=1`（回退 triton MoE）；不设 `VLLM_FL_DSV4_TORCH_COMPILE`（回退 breakable cudagraph）。

## 附：目录说明

```
PORTING_REPORT.md   完整移植与优化工程报告
scripts/            服务启动、压测、模型下载脚本（thead 基线 + FL 栈）
tuning/             deep_gemm 自动调优脚本与分 shard 输出（tune_out/）
results/            benchmark summary/raw CSV、最终日志、profiler 摘要（profiles/）
chat_template/      DeepSeek-V4 chat template（与官方 encoding_dsv4.py 逐字符对齐验证）
```

---
*代码与压测由 Claude Code 完成（2026-08-25 ~ 2026-09-01）。*

## 附录：原始数据

社区栈最终配置的最后一次完整压测原始数据（2026-08-31，每用例 4 轮）。

### A.1 Case1：1024/1024/conc64/256 prompts（util 0.85）

来源：`results/raw_runs_20260831_145409.csv`。4 轮全部 256/256 成功：

| Run | Duration (s) | Output tok/s | Peak tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT | P99 TTFT | Mean TPOT (ms) | Median TPOT | P99 TPOT | Mean ITL (ms) | P99 ITL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 201.51 | 1300.88 | 1600 | 2601.76 | 5306.21 | 4412.90 | 7533.35 | 44.04 | 44.46 | 48.02 | 44.04 | 48.12 |
| 2 | 200.94 | 1304.57 | 1600 | 2609.14 | 5290.56 | 4375.84 | 7490.08 | 43.91 | 44.65 | 48.21 | 43.91 | 48.19 |
| 3 | 200.97 | 1304.42 | 1600 | 2608.83 | 5291.60 | 4391.35 | 7507.17 | 43.92 | 44.50 | 48.03 | 43.92 | 48.16 |
| 4 | 202.52 | 1294.40 | 1600 | 2588.80 | 5435.88 | 4384.44 | 7500.00 | 44.16 | 45.09 | 48.44 | 44.16 | 48.38 |

同一 CSV 中 4096/16384 档位在 util 0.85 下失败（inductor 图执行 OOM，服务中断），因此这两档降 util 至 0.82 后用 `scripts/rerun_cases23.sh` 重跑，即 A.2/A.3。

### A.2 Case2：4096/1024/conc64/256 prompts（util 0.82）

来源：`results/logs/bench_cases23_final.log`。4 轮全部 256/256 成功：

| Run | Duration (s) | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | P99 TTFT | Mean TPOT (ms) | P99 TPOT |
|---|---|---|---|---|---|---|---|---|
| 1 | 380.97 | 0.67 | 688.09 | 3440.44 | 16476.35 | 36476.07 | 76.15 | 102.41 |
| 2 | 406.46 | 0.63 | 644.94 | 3224.72 | 16390.73 | 32858.03 | 81.12 | 99.70 |
| 3 | 318.02 | 0.80 | 824.31 | 4121.54 | 12960.68 | 32796.06 | 65.00 | 76.69 |
| 4 | 412.12 | 0.62 | 636.09 | 3180.43 | 17990.40 | 38905.37 | 81.34 | 105.41 |

### A.3 Case3：16384/1024/conc64/256 prompts（util 0.82）

来源同 A.2。4 轮全部 256/256 成功：

| Run | Duration (s) | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | P99 TTFT | Mean TPOT (ms) | P99 TPOT |
|---|---|---|---|---|---|---|---|---|
| 1 | 902.74 | 0.28 | 290.39 | 4936.58 | 37183.91 | 163123.86 | 173.83 | 199.65 |
| 2 | 799.83 | 0.32 | 327.75 | 5571.74 | 33067.73 | 143241.26 | 162.66 | 193.35 |
| 3 | 776.49 | 0.33 | 337.60 | 5739.26 | 32681.06 | 143228.59 | 157.65 | 186.96 |
| 4 | 775.23 | 0.33 | 338.15 | 5748.54 | 32684.00 | 143247.92 | 157.34 | 186.46 |

汇总口径说明：§2 各表引用的最终值为**4 轮去首轮取均值**（首轮含 warmup 影响）——case1 (1304.57+1304.42+1294.40)/3 ≈ 1301，case2 ≈ 702，case3 ≈ 334。各轮数据如上完整列出供复核。

### A.4 T-Head 厂商栈基线（2026-08-26，同硬件同压测参数）

来源：`results/raw_runs_20260826_111911.csv`。三个用例各 4 轮，全部 256/256 成功（util 0.85）：

**1024/1024/conc64：**

| Run | Duration (s) | Output tok/s | Peak tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT | P99 TTFT | Mean TPOT (ms) | Median TPOT | P99 TPOT | Mean ITL (ms) | P99 ITL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 155.29 | 1688.10 | 2180 | 3376.20 | 4607.58 | 4839.80 | 7671.98 | 33.43 | 34.39 | 37.15 | 33.43 | 38.93 |
| 2 | 152.40 | 1720.07 | 2213 | 3440.14 | 3970.43 | 2981.39 | 5503.85 | 33.35 | 34.17 | 36.78 | 33.35 | 38.94 |
| 3 | 152.52 | 1718.72 | 2202 | 3437.45 | 3972.35 | 3009.68 | 5529.72 | 33.38 | 34.09 | 36.77 | 33.38 | 39.01 |
| 4 | 152.36 | 1720.55 | 2200 | 3441.10 | 3949.39 | 2915.18 | 5435.24 | 33.36 | 34.24 | 36.73 | 33.36 | 39.09 |

**4096/1024/conc64：**

| Run | Duration (s) | Output tok/s | Peak tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT | P99 TTFT | Mean TPOT (ms) | Median TPOT | P99 TPOT | Mean ITL (ms) | P99 ITL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 225.66 | 1161.70 | 2112 | 5808.51 | 12472.85 | 14063.63 | 23455.99 | 42.92 | 41.58 | 54.40 | 42.92 | 39.80 |
| 2 | 225.79 | 1161.03 | 2112 | 5805.16 | 9712.88 | 9545.03 | 23195.53 | 45.65 | 46.21 | 54.22 | 45.65 | 39.74 |
| 3 | 225.74 | 1161.24 | 2112 | 5806.20 | 10320.79 | 11160.46 | 23121.11 | 45.05 | 44.06 | 54.41 | 45.05 | 40.08 |
| 4 | 225.62 | 1161.89 | 2112 | 5809.43 | 10257.76 | 11148.33 | 23169.43 | 45.08 | 43.98 | 54.47 | 45.08 | 39.93 |

**16384/1024/conc64：**

| Run | Duration (s) | Output tok/s | Peak tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT | P99 TTFT | Mean TPOT (ms) | Median TPOT | P99 TPOT | Mean ITL (ms) | P99 ITL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 545.61 | 480.46 | 1984 | 8167.78 | 24878.60 | 15714.57 | 99456.92 | 108.92 | 117.81 | 131.21 | 108.92 | 3141.94 |
| 2 | 545.73 | 480.35 | 1984 | 8166.02 | 25781.87 | 18843.08 | 99094.13 | 108.07 | 116.30 | 131.24 | 108.07 | 3142.07 |
| 3 | 545.55 | 480.52 | 1984 | 8168.77 | 25446.08 | 15715.03 | 99102.68 | 108.35 | 117.92 | 131.24 | 108.35 | 3142.15 |
| 4 | 544.92 | 481.07 | 2002 | 8178.11 | 24960.14 | 15715.84 | 99189.51 | 108.68 | 117.61 | 131.26 | 108.68 | 3142.13 |

§2.1 引用的基线值同样为去首轮均值：1719.8 / 1161.4 / 480.6 tok/s。
