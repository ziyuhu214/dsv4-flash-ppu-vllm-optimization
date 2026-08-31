# DeepSeek-V4-Flash 在 T-Head PPU 上的 vLLM 推理性能优化

本仓库汇总 DeepSeek-V4-Flash（W8A8-INT8）在 T-Head PPU（ZW810E ×8）上，用**开源社区技术栈**（官方 vLLM + FlagGems + vllm-plugin-FL）进行推理性能优化的全过程资料：启动/压测脚本、deep_gemm 调优脚本与结果、benchmark 数据，以及本篇优化报告。

代码改动以 PR 形式提交在两个配套仓库：

| 仓库 | PR 分支 | 内容 |
|---|---|---|
| [ziyuhu214/vllm-plugin-FL](https://github.com/ziyuhu214/vllm-plugin-FL) | `feat/deepseek-v4-flash-ppu-perf` | DeepSeek-V4 PPU op 移植、int8 GEMM/MoE 路径、CUDA graph 支持、qnorm 头填充快速路径（6 个提交） |
| [ziyuhu214/FlagGems](https://github.com/ziyuhu214/FlagGems) | `fix/deepseek-v4-ppu-fixes` | fp8 编码、sqlite 锁、int64 偏移、qnorm 快速路径四个内核改动（4 个提交） |

**工程细节详见 [`PORTING_REPORT.md`](PORTING_REPORT.md)**（完整移植与优化工程报告：修改清单、算子归属图、性能里程碑、诊断结论存档、遗留事项）。

> ⚠️ 部分代码从 T-Head 厂商 vLLM fork（0.20.1+ppu）逐字移植，厂商授权未确认，**所有仓库保持 private**。

## 1. 背景与目标

T-Head 为 PPU 提供了一个基于 vLLM 0.20.1 的厂商 fork，DeepSeek-V4-Flash 可以直接运行且性能良好。但厂商 fork 版本旧、闭源内核多、难以跟进社区演进。本项目的目标：

1. 先在容器内验证 **T-Head 厂商栈** 的推理性能，作为基线；
2. 参考厂商实现，把 **社区栈**（vllm-0.24.0 + FlagGems v5.3.4 + vllm-plugin-FL）在 PPU 上补齐、调优，尽量逼近厂商基线。

## 2. 软硬件环境

| 项 | 值 |
|---|---|
| 硬件 | T-Head PPU ZW810E × 8（卡 0–7） |
| 模型 | DeepSeek-V4-Flash-0731，W-INT8 per-channel / A-INT8 per-token 量化（ModelScope 下载，见 `scripts/download_dsv4_int8.sh`） |
| 厂商基线栈 | T-Head fork vLLM 0.20.1+ppu（PPUInt8ScaledMMLinearKernel + FlashMLA sparse，闭源内核） |
| 社区优化栈 | vLLM 0.24.0（官方，empty platform 编译）+ FlagGems v5.3.4 + vllm-plugin-FL（flagos-ai main）+ PPU deep_gemm / acext wheel |
| 部署参数 | TP=8，max-model-len 65536，fp8 KV cache（fp8_ds_mla），gpu-util 0.82（短输入可 0.85），batch 32768，无 prefix caching，`VLLM_FL_DSV4_TORCH_COMPILE=1` + FULL_AND_PIECEWISE cudagraph |

启动脚本：厂商基线 `scripts/serve_dsv4_flash_int8.sh`，社区栈 `scripts/serve_dsv4_fl_bench.sh`（两者参数对齐以保证可比性）。

## 3. 性能结果

压测方式：vLLM bench serve，random 数据集，ignore-eos，decode 1024，并发 64，256 条请求，每档取多轮均值（见 `results/*.csv` 与 `results/logs/`）。

### 3.1 T-Head 厂商基线（2026-08-26，`summary_20260826_111911.csv`）

| Prefill | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|
| 1024 | **1719.8** | 3964 | 33.4 |
| 4096 | 1161.4 | 10097 | 45.3 |
| 16384 | 480.7 | 25396 | 108.4 |

### 3.2 社区栈优化演进（prefill 1024 / decode 1024 / conc 64）

| 阶段 | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | 说明 |
|---|---|---|---|---|
| 初次跑通（08-27，`summary_20260827_130315.csv`） | 706.0 | 8001 | 82.9 | 全 eager、triton 通用内核 |
| **最终（08-31，`summary_20260831_145409.csv`，4 轮去首轮）** | **1301.1** | 5339 | 44.0 | 全部优化 + qnorm 头填充快速路径 |

### 3.3 最终三用例总表（vs 厂商基线，4 轮去首轮，256/256 全成功）

| 用例 (prefill/decode/conc) | T-Head 基线 tok/s | 社区栈最终 tok/s | 比例 | 社区栈 TPOT | 社区栈 TTFT |
|---|---|---|---|---|---|
| 1024/1024/64 | 1719.8 | **1301** | 76% | 44.0 ms | 5.3 s |
| 4096/1024/64 | 1161.4 | **702** | 60% | 75.8 ms | 15.8 s |
| 16384/1024/64 | 480.6 | **334** | 70% | 159.2 ms | 32.8 s |

数据：case1 `results/raw_runs_20260831_145409.csv`（util 0.85），case2/3 `results/logs/bench_cases23_final.log`（util 0.82——长输入在 0.85 下触发 inductor 图执行 OOM，降 util 后全部通过）。

**结论：社区栈 case1 从 706 → 1301 output tok/s（+84%），达厂商基线 76%；TPOT 从 82.9ms 降到 44.0ms（厂商 33.4ms）。** 4096 档位比例偏低（60%）主因是 compile 模式下 KV cache 偏小放大了长输入排队（TTFT 差距大于短输入），见 `PORTING_REPORT.md` 第七、八节。

## 4. 优化内容（对应 PR 提交）

### 4.1 vllm-plugin-FL：`feat/deepseek-v4-flash-ppu-perf`

1. **移植 PPU deep_gemm int8 GEMM/MoE 路径**（`ppu_deep_gemm*.py` + `ppu_deepgemm_configs/`）
   厂商 fork 用 PPU deep_gemm wheel 跑 int8 dense/grouped GEMM。移植其包装层，使社区 vLLM 的 MoE 与 dense 层直连 deep_gemm，替代慢的通用 triton 路径。附带用 `tuning/tune_dsv4_deepgemm.py` 对本部署的全部 GEMM 形状（MoE up/gate/down @ 32/33 groups、attention dense @ TP8）自动调优，产出 `DeepSeek-V4-Flash-0731-w8a8-int8-tp8,device_name=PPU-ZW810E` 配置。

2. **DeepSeek-V4 专用 op 与运行时补丁**（`deepseek_v4_*.py`、`patches/deepseek_v4_thead.py`）
   - sparse-attn indexer：上游依赖 NVIDIA `fp8_fp4_mqa_logits`（要求 q/k 同 dtype），PPU 上 indexer Q 为 INT8，改走 deep_gemm 的 `int8_(paged_)mqa_logits`；topk 用 FlagGems `top_k_per_row_*` 替代缺失的 `persistent_topk`。
   - fp8_ds_mla KV cache 的 compress/insert/gather-dequant、o_proj int8、MHC 系列 op。
   - 运行时补丁（幂等）：放开 int8 W8A8 MoE 的 NVIDIA-only 门控到 cuda-alike、FlashMLA sparse attention 接线等。
   - `_C` op 桥接：`dynamic_scaled_int8_quant`、`silu_and_mul` 的 triton/FlagGems fallback；`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` 改造成 0.24 的 9 参 head-padding 契约并注册 Meta impl 以支持 torch.compile。

3. **acext INT8 W8A8 scaled-MM 线性内核**（`quantization/acext_int8_linear.py`）
   移植厂商 `PPUInt8ScaledMMLinearKernel` 的 acext 分支：行主序 int8 权重直入 `acext.int8_gemm`，per-token 动态激活量化。在 thead + acext 可用时优先于 cutlass/triton 被内核 oracle 选中。o_proj 走该路径带来最后一档提升（见 `logs/bench_case1_int8oproj.log`）。

4. **PPU 上启用 CUDA graph 捕获 + 显存核算修复**（platform/worker/model_runner）
   - `support_static_graph_mode()` 加入 thead：否则上游强制 `cudagraph_mode=NONE`，decode 全 eager（这是 706 → 1300 中最大的单项收益之一）。
   - cudagraph 显存估算门控从 `is_cuda()` 放宽到 `is_cuda_alike()`，否则 KV cache 吃掉 graph pool 余量导致捕获 OOM。
   - graph 显存 profiling 纳入 `BreakableCUDAGraphWrapper`（DeepseekV4 使用），并在每次采样前后 `empty_cache()`——eager warmup 的 allocator 缓存曾虚增约 90 GiB 的 graph 显存估计。

5. **qnorm 插入头填充快速路径**（与 FlagGems 第 4 项配套）：padding 头槽位（FlashMLA 要求 64 头 vs TP8 实际 8 头）由 zeros 分配 + 全量 RMSNorm/RoPE 改为 empty 分配 + kernel 内直接写零跳过计算。

6. benchmark 脚本对齐 DSv4 int8 部署参数。

### 4.2 FlagGems：`fix/deepseek-v4-ppu-fixes`

1. **软件 E4M3FN 编码**：`fused_qnorm_rope_kv_insert_kernel` 原用 `x.to(tl.float8e4nv)` 存 fp8 KV cache，PPU 的 Triton 无 fp8e4nv 类型。新增 `_encode_e4m3fn`（RNE 舍入、含 subnormal 路径、位精确）替换全部 7 处存储点。
2. **sqlite 调优缓存锁**：TP=8 时 8 个 worker 共享一个 sqlite 调优缓存，同时调优新 shape 时默认 5s busy timeout 直接报 `database is locked` 杀掉 worker；对 sqlite URL 传 `timeout=120`。
3. **int64 偏移溢出**：`cp_gather_indexer_quant_cache` 的 scale 偏移用 int32 计算，paged cache 超过 ~8k blocks 时溢出、长上下文场景下取错 scale；提升为 int64。
4. **qnorm kernel `num_real_heads` 快速路径**：padding 头槽位走 store-zeros-and-return，跳过 RMSNorm+RoPE。

## 5. 与厂商基线的剩余差距（case1 ~24%）

详细归因见 `PORTING_REPORT.md` 第六、七节，要点：

- **opaque op 边界开销 + 厂商 C++ 融合算子**（qnorm+rope+quant+insert 一体）+ 编译版 custom allreduce——突破需编译 C++ 扩展或 PPU 工具链升级；
- **KV cache 24 vs 37 GiB**：PIECEWISE opaque 边界静态缓冲预留 ~9.1GB，放大长输入排队（4096 档位比例 60% 的主因）；根治靠 mHC 原生进图（等 PPU inductor 成熟，native 实现已留存并验证 1-ulp 等价）；
- 通信（pccl oneshot，decode 中位 15μs）与 indexer decode（0.14ms/层）已近最优。

## 6. 复现步骤

```bash
# 1. 下载 int8 量化模型
bash scripts/download_dsv4_int8.sh

# 2. 安装社区栈：vllm-0.24.0（empty platform）、FlagGems（fix/deepseek-v4-ppu-fixes 分支）、
#    vllm-plugin-FL（feat/deepseek-v4-flash-ppu-perf 分支）、PPU deep_gemm / acext wheel

# 3.（可选）重新调优 deep_gemm 配置
python tuning/tune_dsv4_deepgemm.py

# 4. 启动服务（VLLM_FL_DSV4_TORCH_COMPILE=1 + FULL_AND_PIECEWISE，util 0.82）
bash scripts/serve_dsv4_fl_bench.sh

# 5. 压测：case1 用 vllm-plugin-FL 仓库内 benchmarks/benchmark_throughput_serve.py，
#    case2/3（4096/16384）用 scripts/rerun_cases23.sh
```

## 7. 目录说明

```
scripts/        服务启动、压测、模型下载脚本（thead 基线 + FL 栈）
tuning/         deep_gemm 自动调优脚本与分 shard 输出（tune_out/）
results/        benchmark summary/raw CSV 与最终日志
chat_template/  DeepSeek-V4 chat template（与官方 encoding_dsv4.py 逐字符对齐验证）
```

---
*代码与压测由 Claude Code 完成（2026-08-25 ~ 2026-08-31）。*
