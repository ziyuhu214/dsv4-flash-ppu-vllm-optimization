# DeepSeek-V4-Flash INT8 on T-Head PPU：部署与优化报告

社区栈（官方 vLLM 0.24.0 + vllm-plugin-FL + FlagGems）替代厂商T-Head vllm的移植与调优结果。

- 报告日期：2026-09-04
- 性能数据：`vllm-plugin-FL/benchmark_results/summary_20260904_111410.csv`（当日实测，三用例完整）
- 改动范围：vllm-plugin-FL（7 提交 + 4 文件未提交）、FlagGems（6 提交）；**vLLM 0.24.0 本体零修改**

---

## 1. 运行环境

### 1.1 硬件与驱动

| 项 | 值 |
|---|---|
| 加速卡 | T-Head PPU 真武 ZW810E × 8（16 卡机，本次用卡 0–7） |
| 单卡规格 | 95.6 GiB HBM2e，64 SM，SM 8.0 兼容，L2 64 MB |
| 单卡算力 | BF16 峰值 123 TFLOPS；INT8 = BF16 × 2 = **246 TOPS** |
| 驱动 | PPU-SMI 1.28，Driver 1.3.2-d7f5a2，HGGC 13.0 |
| 主机 | Linux 5.10.134 (al8)，Python 3.12.3 |

### 1.2 软件栈

| 组件 | 版本 | 说明 |
|---|---|---|
| vLLM | `0.24.0+empty`（editable `/workspace/vllm-0.24.0`） | 官方 empty-platform 构建，无编译内核（无 `vllm._C`），**本体零修改** |
| vllm-plugin-FL | `main@f4319bd` + 7 提交 + 4 文件未提交 | 平台插件，所有框架层改动落点 |
| FlagGems | `v5.3.4` + 6 提交 | triton 算子库 |
| PyTorch / Triton | 2.10.0 / `3.6.0+v0.1.0.ppu2.1.0` | PPU 版，全程未改 |
| PPU vendor wheels | `deep_gemm 1.0.0+v0.2.0`、`flash_mla 2.0.0+v0.1.0`、`flash_mla_asllm 1.0.2+dsv4`、`acext 1.0.0`、`deep_ep 1.0.0+v0.2.0` | 闭源二进制，直接调用 |
| compressed-tensors | 0.17.0 | 0.24.0 要求（原 0.14 需升级） |

### 1.3 模型与部署参数

| 项 | 值 |
|---|---|
| 模型 | `T-HEAD/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken` |
| 结构 | 43 层，MoE 256 专家 top-6 + 1 共享专家，hidden 4096，moe_inter 2048，64 注意力头，MLA sparse（`q_lora_rank` 1024） |
| 量化 | W8A8-INT8（权重 per-channel，激活 per-token 动态） |
| 并行 | TP=8（每 rank 8 注意力头、32 routed 专家） |
| KV cache | fp8（`fp8_ds_mla`），**198,758 tokens** |
| 关键参数 | `max-model-len 65536`、`util 0.85`、`max-num-batched-tokens 32768`、无 prefix caching |
| 编译 | `VLLM_FL_DSV4_TORCH_COMPILE=1` + `FULL_AND_PIECEWISE` cudagraph |

> **util 0.85 适用范围已扩大**：原记录中 4096/16384 两档在 0.85 下会触发 inductor 图执行 OOM、必须降至 0.82。本次去掉 Q head padding 后 KV cache 从 172,324 增至 198,758 tokens（+15.3%），三个用例在 0.85 下**全部通过、零 OOM**。此为单次观察，正式固化前建议再确认一次。

### 1.4 基线与对照

| 对照对象 | 说明 |
|---|---|
| **T-Head 厂商栈** | vLLM `0.20.1+ppu` fork，闭源 C++ 融合算子。同硬件，衡量**软件栈完成度**。本次实测复现 1726.54 tok/s / TPOT 33.23 ms，与存档基线 1719.78 / 33.36 一致 |
| **NVIDIA H100** | 8×H100 80GB，CUDA 13.0，Driver 580.105.08，`vllm/vllm-openai:v0.26.0`，FP8 权重模型，TP=8 + fp8 KV。跨硬件，衡量**硬件代际差** |


---

## 2. 性能结果与比对

压测口径统一：`vllm bench serve`，random 数据集，`--ignore-eos`，decode 1024，并发 64，256 请求，4 轮取后 3 轮均值。使用 plugin 自带 `benchmarks/benchmark_throughput_serve.py`，保证同口径。

本次测量环境：16 张卡全部空闲，**无其他租户干扰**。

### 2.1 三用例总表

| 用例 (prefill/decode/conc) | 输出吞吐 | TPOT | TTFT | 压测时长 |
|---|---|---|---|---|
| 1024/1024/64 | **1633.11** tok/s | 35.12 ms | 4.18 s | 160.5 s |
| 4096/1024/64 | **974.86** tok/s | 55.01 ms | 10.89 s | 269.9 s |
| 16384/1024/64 | **411.86** tok/s | 127.96 ms | 26.21 s | 637.1 s |

### 2.2  vs T-Head 厂商栈（同硬件）

| 用例 | T-Head | 优化前 | **本次** | 占 T-Head | 提升 |
|---|---|---|---|---|---|
| 1024/1024 | 1719.8 | 1486.9 (86.5%) | **1633.11** | **95.0%** | +9.8% |
| 4096/1024 | 1161.4 | 702.0 (60.4%) | **974.86** | **83.9%** | +38.9% |
| 16384/1024 | 480.6 | 334.0 (69.5%) | **411.86** | **85.7%** | +23.3% |

TPOT 差距收窄：

| 用例 | T-Head | 优化前 | **本次** | 差距收窄 |
|---|---|---|---|---|
| 1024/1024 | 33.36 ms | 44.00 | **35.12** | **83%** |
| 4096/1024 | 45.26 ms | 75.80 | **55.01** | **68%** |
| 16384/1024 | 108.37 ms | 159.20 | **127.96** | **61%** |

TTFT 已进达到 T-Head 的 **1.03–1.08** 倍：4.18 s vs 3.96 s、10.89 vs 10.10、26.21 vs 25.40。

### 2.3  vs NVIDIA H100（跨硬件）

| 用例 | H100 | **PPU 本次** | PPU/H100 | 优化前 | H100 TPOT | PPU TPOT |
|---|---|---|---|---|---|---|
| 1024/1024 | 3328.9 | **1633.11** | **49.1%** | 44.7% | 18.3 ms | 35.12 ms |
| 4096/1024 | 2530.5 | **974.86** | **38.5%** | 27.7% | 23.4 ms | 55.01 ms |
| 16384/1024 | 1272.4 | **411.86** | **32.4%** | 26.3% | 44.2 ms | 127.96 ms |
| 65536/1024 | 392.5 | 未测 | — | — | 138.3 ms | — |

绝对吞吐为 H100 的 32–49%，与两者标称算力比（1968 TOPS vs 15832 TFLOPS，即 12.4%）相比明显更优——见下节效率口径。

### 2.4 c. 单位算力效率（tok/s per TOPS）

口径：两边均取 **BF16 峰值 × 2** 作为 INT8/FP8 等效算力，对应各自部署的主 GEMM 精度（PPU W8A8-INT8 / H100 FP8）。分母为 8 卡合计：

- PPU：123 TFLOPS × 2 = 246 TOPS/卡 → **1968 TOPS**
- H100 SXM：989.5 TFLOPS × 2 = 1979 TFLOPS/卡 → **15832 TFLOPS**（即官方 FP8 dense 峰值）

| 用例 | PPU tok/s per TOPS | H100 tok/s per TFLOPS | **PPU/H100 效率比** | 优化前效率比 |
|---|---|---|---|---|
| 1024/1024 | **0.830** | 0.210 | **3.95×** | 3.59× |
| 4096/1024 | **0.495** | 0.160 | **3.10×** | 2.23× |
| 16384/1024 | **0.209** | 0.080 | **2.60×** | 2.11× |

PPU 每单位标称算力的实际 token 产出为 H100 的 **2.6–4.0 倍**。

---

## 3. 优化内容


### 3.1 框架层优化

#### F1. CUDA graph 在 thead 平台被禁用 + 显存核算错误

- **算子/组件**：`vllm_fl/worker/worker.py`、`model_runner.py`、`platform.py`（提交 `3474000`）
- **原因（能力缺失 + Bug）**：`support_static_graph_mode` 白名单不含 thead，decode 全程 eager；显存预估门控写死 `is_cuda()`，PPU 走不到；allocator 缓存导致预估虚增约 90 GiB。
- **优化**：白名单加入 thead；`is_cuda()` → `is_cuda_alike()`；将 `BreakableCUDAGraphWrapper` 纳入 profiling 核算；修复缓存虚增。
- **效果**：本项目**单项收益最大**（decode 由全 eager 转为全图）。

#### F2. torch.compile 全图模式

- **算子/组件**：`vllm_fl/patches/deepseek_v4_thead.py`、`ops/deepseek_v4_attn_op.py`、`ops/deepseek_v4_mhc_ops.py`（提交 `026ab85`）
- **原因（性能不足）**：PIECEWISE breakable cudagraph 每尺寸驻留 1.6–2.2 GiB 段间显存，迫使捕获尺寸稀疏；dynamo 无法 trace FlashMLA / indexer / workspace 管理内部。
- **优化**：将 `attention_impl` 与 mHC 包装为 opaque custom op（`vllm::fl_dsv4_attention` 等）置于图外，模型主体 `support_torch_compile` 全图化，注册 `splitting_ops`；`VLLM_FL_DSV4_TORCH_COMPILE=1` 启用。
- **效果**：978 → 1098 tok/s（两端均有日志：`bench_case1_final.log` 977–980、`bench_case1_compile.log` 1097–1099）。mHC 原生进图版本（`deepseek_v4_mhc_native.py`，已验证 1-ulp 等价）保留但未启用——PPU inductor 会触发 `ppu-llc CalledProcessError` 且单图编译约 32 分钟。

#### F3. deep_gemm 调优配置从未加载（两个叠加 Bug）

- **算子/组件**：`vllm_fl/ops/ppu_deep_gemm.py`（未提交）
- **原因（Bug）**：
  1. 加载器用 `current_platform.get_device_name()` 匹配文件名，该函数返回**厂商标签** `'thead'`，而全部 12 个配置文件按**芯片名** `PPU-ZW810E` 命名 → 签名永不匹配，**153 条 DSv4 调优配置从未生效**。
  2. 本项目自产配置的 `config` 字段是 **dict**，11 个厂商文件是 repr **字符串**，加载器无条件调 `ast.literal_eval()` → 对 dict 抛 `ValueError`。仅修 (1) 会导致首次 MoE 调用崩溃。
- **优化**：优先使用驱动上报的芯片名 `torch.cuda.get_device_name(0)`，回退平台标签；新增 `_parse_config()` 兼容 dict/str。加 `VLLM_FL_DEEPGEMM_CONFIGS=0` 开关。
- **效果**：配置项从 0 增至 549 条。

#### F4. 环境与运行时适配（一组小修）

- **算子/组件**：`vllm_fl/patches/deepseek_v4_thead.py` 内多个补丁、`platform.py`；环境层面的包版本
- **原因（能力缺失 / 参数错误）**：官方 0.24.0 与 PPU 环境存在多处不匹配——厂商 vLLM 与官方版无法共存；`compressed-tensors` 0.14 不满足 0.24.0 要求；int8 checkpoint 权重名与上游期望不一致；CuteDSL 在 PPU 不可用却被默认启用；indexer 的 `num_sms` 按 NVIDIA 假设取值；cudagraph 非对称尺寸占用过多显存；`topk_indices_buffer` 未初始化导致稀疏注意力读到脏值。
- **优化**：卸载 T-Head vLLM（完整备份至 `/workspace/vllm-thead-ref`）并安装官方 `0.24.0+empty`；升级 `compressed-tensors` 0.14 → 0.17；新增 int8 权重名映射；禁用 CuteDSL；按 PPU 实际 SM 数（64）修正 indexer `num_sms`；瘦身 cudagraph 非对称捕获尺寸；`topk_indices_buffer` 全量初始化为 −1。

### 3.2 算子层优化

#### K1. `flash_sparse_decode_fwd` / `splitkv_mla_combine`：Q head padding（本轮最大收益）

- **算子**：PPU `flash_mla` 稀疏 decode 内核及其 combine 内核
- **原因（上游缺 PPU 分支）**：上游 `get_padded_num_q_heads()`（`models/deepseek_v4/nvidia/flashmla.py:59`）无条件返回 `64 if num_heads <= 64 else 128`——这是 **NVIDIA FP8 decode 内核**的限制，PPU flash_mla wheel 并无此约束。T-Head fork 有显式分支：
  ```python
  if current_platform.is_ppu():
      self.padded_heads = num_heads   # PPU FlashMLA 支持任意 h_q
  ```
  64 头 / TP8 → 每 rank 仅 8 真实头，被 pad 到 64，内核选中慢变体。profile 实测：
  | 内核 | T-Head | FL（padding） |
  |---|---|---|
  | `flash_sparse_decode_fwd<512,16,...>` | 51.4 µs | — |
  | `flash_sparse_decode_fwd<512,64,...>` | — | **82.3 µs** |
  | `splitkv_mla_combine<512,16>` | 3.8 µs | — |
  | `splitkv_mla_combine<512,64>` | — | **9.9 µs** |

  合计 **+2.12 ms/step**，是 4.9 ms TPOT 差距中的最大单项。
- **优化**：插件内 patch `get_padded_num_q_heads` 返回 `num_heads`（`_patch_no_q_head_padding`）。抽象基类 docstring 明确授权此路径（*"Backends with no padding constraint return num_heads"*）。开关 `VLLM_FL_Q_HEAD_PADDING=1`。
- **改前审计**（全部在 `padded_heads == 8` 时正确退化）：`attn_sink` 参数尺寸恰好匹配（loader 只填前 `n_local_heads` 槽）、`o_padded` 切片退化为 no-op、profile-run 的 `F.pad` 被守卫跳过、FlagGems qnorm wrapper 已有 `q_head_padded == num_heads` 快路径（跳过 `torch.zeros` + `copy_` + 零填充，额外收益）、flash_mla 的 `sched_meta.config.h_q == q.shape[2]` 断言因 tile-scheduler 计划为空、由内核 planner 按实际 q 形状填充而自动一致。
- **效果**：内核变体切换经 profile 确认（`traits<512,64>` → `<512,16>`）；KV cache +15.3%（`o_padded` 由 64 头缩至 8 头）。
- **连带影响**：FlagGems `ae6fd75e`（padding 头槽位跳过 RMSNorm/RoPE 的快速路径）**已无优化对象**——padding 消失后不存在 padding 槽位。该提交此前是在优化本问题的**症状**。

#### K2. `aten::copy_` / `clamp*` / `cat*` / `arange*`：FlagGems triton 覆盖导致 host 侧开销

- **算子**：FlagGems 对上述 aten 算子的 triton 实现
- **原因（性能不足）**：profile 归因显示 decode 有 5.14 ms/step（13.1%）GPU 空转，而**并非** metadata 内核成本（两栈近乎相同且低廉），而是**每次 aten 调用的 host 侧开销**：
  | | T-Head | FL |
  |---|---|---|
  | `aten::copy_` | 0.852 ms / 56 次 | **4.510 ms / 54 次** |
  | `aten::fill_` | 0.165 ms / 28 次 | **1.767 ms / 29 次** |
  | cpu_op 合计 | **4.712 ms/step** | **18.393 ms/step** |

  调用次数相同而单次成本高 5–10 倍：FlagGems 以 triton 内核实现，每次调用付 Python dispatch + JIT 缓存查找 + launch，而 native ATen 直接下发 memcpy/memset。T-Head 基线容器**根本未安装 FlagGems**，故不付此开销。旁证：`clamp_func_min` 在 FL 为 48 次/step，T-Head 为 0。
- **优化**：`vllm_fl/dispatch/config/thead.yaml` 的 `flagos_blacklist` 加入 13 个函数名，令其回落 native ATen（**零代码改动**）。隔离实测（单进程单算子，200 次预热，7×200 取中位）：
  | 算子 | FlagGems | native | 加速 |
  |---|---|---|---|
  | `copy_` | 58.44 µs | 4.24 | **13.8×** |
  | `clamp_tensor` | 73.71 | 4.74 | **15.6×** |
  | `clamp_max` | 73.45 | 5.16 | 14.2× |
  | `clamp_` | 55.06 | 4.19 | 13.1× |
  | `clamp_min` | 69.43 | 6.17 | 11.3× |
  | `cat_out` | 36.07 | 5.43 | 6.6× |
  | `cat` | 41.11 | 6.66 | 6.2× |
  | `arange` / `arange_start` | 25.10 / 25.01 | 9.07 / 9.04 | 2.8× |
- **按数据排除的项**：`zero_` 在 FlagGems 为 15.42 µs、native 为 55.97 µs——**native 反而慢 3.6 倍**（PPU torch 后端 `zero_` 路径较差），故保留在 FlagGems，原设计正确。`zeros`(1.34×)、`zeros_like`(0.82×)、`_to_copy`(0.96×) 收益边际，一并排除。


#### K3. `fused_qnorm_rope_kv_insert`：triton 缺 `fp8e4nv` dtype

- **算子**：FlagGems `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`（提交 `efd9e35e`）
- **原因（能力缺失）**：内核以 `x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)` 写入 fp8 KV cache，PPU/SM80 的 triton 无 `fp8e4nv` dtype，编译失败。
- **优化**：实现 `_encode_e4m3fn` 纯位运算软件编码器（round-to-nearest-even，含 normal 与 subnormal 路径，输入预 clamp 至 ±448），替换全部 7 处写入点。

#### K4. `cp_gather_indexer_k_quant_cache`：int32 溢出导致长上下文崩溃

- **算子**：FlagGems `cp_gather_indexer_k_quant_cache`（提交 `26ea167e`）
- **原因（正确性 Bug）**：`block_id * kv_cache_scale_stride` 以 int32 计算，`stride(0)` 约 26 万 fp32 元素/块，KV cache 超过约 8k 块即溢出，gather 到的 scale 被破坏，长上下文触发 IMA 崩溃。
- **优化**：`block_id` 与 `block_offset` 提升为 int64 后再相乘，与上方数据偏移计算保持一致。

#### K5. `aten::empty` 零填充：3000 次/step 的多余 launch

- **算子**：FlagGems `empty` 覆盖（`c993b3c8`）+ 插件 `empty_int_zero.py`（`f615560`）
- **原因（性能不足 + 隐式依赖）**：FlagGems 的 triton `empty()` 顺带零填充，DeepSeek-V4 decode 每步约 3000 次内核 launch、占 GPU 时间 6%。但直接移除会崩——DSv4 路径部分消费者（KV compressor triton 内核读取 int32 `block_table` / int64 `slot_mapping` / `token_to_req_indices`）**隐式依赖**这个零填充。
- **优化**：FlagGems 侧注销 `aten::empty` 覆盖；插件侧改为**仅对整型/bool dtype** 零填充（`_empty_int_zero`）。
- **效果**：本项目**第二大单项收益**（约 +14%，1301 → 1486.9 tok/s）。


#### K6. dense int8 线性层：上游路径为 fp8-only

- **算子**：`vllm_fl/quantization/acext_int8_linear.py`（`617a867`）、`ops/deepseek_v4_o_proj.py`
- **原因（能力缺失）**：上游 `_o_proj` 为 fp8 专用（`fp8e4nv` triton + deep_gemm `fp8_einsum`），PPU 两者皆无。
- **优化**：移植厂商 `AcextInt8ScaledMMLinearKernel`（行主序权重直入 `acext.int8_gemm`，注册为 thead 平台内核首选）；`_o_proj` 按层分派——int8 `wo_a` 走移植的 inv-RoPE triton + deep_gemm int8 GEMM 路径（初版为 fp32 einsum，后升级为 int8 GEMM），其余回退原实现。

#### K7. MoE / indexer / KV 路径的 PPU 算子桥接

- **算子**：int8 MoE experts、indexer mqa logits、indexer cache/topk ×4、KV 压缩/插入/收集、`moe_align_block_size` 等 4 算子、`dynamic_scaled_int8_quant`、`silu_and_mul`、`qnorm_insert`、`topk_softplus_sqrt`、flashmla ops（逐项见下表）
- **原因（能力缺失）**：主因是 empty 构建**没有 `vllm._C`**（日志中持续出现 `Failed to import from vllm._C`），凡上游走编译内核的算子全部无实现；其次是部分上游路径为 NVIDIA 专用（indexer mqa logits）或准入条件过严（int8 MoE），以及 flashmla ops 的绑定不指向 PPU wheel。
- **优化**：逐一改接 PPU vendor wheel 或 FlagGems triton 实现，缺失者以 torch 回退兜底：
  | 组件 | 实现 | 具体原因 |
  |---|---|---|
  | int8 MoE（routed + shared） | `PPUDeepGemmExperts`（`ppu_deep_gemm_moe.py`） | 上游 int8 MoE 准入过严，且需直连 PPU deep_gemm |
  | indexer mqa logits | deep_gemm `int8_(paged_)mqa_logits` | 上游路径 NVIDIA-only |
  | indexer cache/topk ×4 | FlagGems triton 桥接 | `vllm._C` 不存在 |
  | KV 压缩/插入/收集 | FlagGems `qnorm_insert` + 移植 SM80 compressor / dequant-gather 内核 | 同上 |
  | `moe_align_block_size` 等 4 算子 | FlagGems 桥接 | 同上 |
  | `dynamic_scaled_int8_quant` / `silu_and_mul` / `qnorm_insert`(9 参 schema) | `_C` 算子 triton 回退（`_C_ops_registry.py`） | `vllm._C` 不存在 |
  | `topk_softplus_sqrt` | torch 回退 | 同上 |
  | flashmla ops | 重绑 PPU `flash_mla` pip 包 | 上游绑定不适用 |

#### K8. sqlite 调优缓存并发写锁

- **算子/组件**：FlagGems `utils/models/sql.py`（`a597ec4c`）
- **原因（Bug）**：TP=8 下 8 个 worker 共用同一 sqlite 调优缓存，同时调优新形状时默认 5 s busy timeout 导致 worker 以 `database is locked` 崩溃。
- **优化**：sqlite URL 传入 `connect_args={'timeout': 120}`，写者等待而非退出。

### 3.3 本轮收益归因

| 优化项 | 层级 | TPOT 影响 | 吞吐影响 |
|---|---|---|---|
| K1 去 Q head padding | 算子层 | −2.12 ms/step | 主要贡献 |
| K2 FlagGems 黑名单 | 算子层 | host 侧 18.39 → 目标 4.71 ms/step | 主要贡献 |


case1 TPOT 由 38.21 降至 35.12 ms（−8.1%），吞吐由 1486.9 升至 1633.11（+9.8%）。

**case1 完整演进与证据链**（凡容器内有日志者列出，便于复核）：

| 阶段 | tok/s | 证据 |
|---|---|---|
| 初次跑通（graph 全关） | 102 | ⚠️ 无日志，继承未核实 |
| + F1 CUDA graph 修复 | 706 | ⚠️ 无日志，继承未核实 |
| + K7 deepgemm int8 MoE（vendor kernel） | 978 | `bench_case1_final.log`（977.08–980.19）|
| + F2 torch.compile 全图 | 1098 | `bench_case1_compile.log`（1097.85–1099.38）|
| + K6 o_proj int8 GEMM | 1290 | `bench_case1_int8oproj.log`（1235.99–1290.34）|
| + prefill 调优 | 1301 | `bench_case1_final3.log`（1299.55–1304.10）|
| + K5 empty 定向修复 | 1486.9 | `case1_emptyfix_v2.log`（1484.06–1489.26）|
| **+ K1 去 padding + K2 FlagGems 黑名单** | **1633.11** | `benchmark_results/summary_20260904_111410.csv`（1628.7–1638.9）|

### 3.4 已验证无效或负收益的尝试（避免重复投入）

| 尝试 | 结果 | 原因 |
|---|---|---|
| 收窄 `empty` 零填充 dtype 至 int32/int64 | 减少 365 次 launch / 0.41 ms/step，吞吐 **+0.13%（噪声）** | 该批 1 µs 级内核落在既有空隙中，不在关键路径 |
| 缓存 metadata builder 的 pinned arange | GPU 空转 5.14 → 3.91 ms/step（如预期），吞吐 **−1.1%** | 每步改写 `torch.repeat_interleave` 全局量使 Dynamo guard 失效，代价超过收益。已 gate 在 `VLLM_FL_PIN_ARANGE_CACHE=1`（默认关） |


### 3.5 剩余差距

case1 距 T-Head 仅 5%，但长输入仍差 14–16%。profile 指向：

1. **流并行度**（约 2.26 ms/step）：T-Head 工作分布在 4 条 aux 流（最大单流 7.40 ms，并行系数 1.13×），本栈为 3 条（最大单流 10.59 ms，1.04×）。关键路径取最大流。**机制未查清**——两栈 Python 层均只创建 3 条 aux 流（T-Head `deepseek_v4.py:1388`、0.24.0 `nvidia/model.py:948`），fan-out 逻辑与门限（`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024`）相同，差异出现在 CUDA graph 捕获结构，非 Python 层。
2. **host 侧 `prepare_inputs` 空转**：T-Head 仅 0.790 ms/step（2.7%），本栈优化前 5.142 ms（13.1%）。K2 应已显著改善，但**优化后的实际残余量尚未测量**。
3. **长输入差距更大**说明 prefill 侧另有问题，需单独 profile 长输入定位（本项目 profile 均针对 decode）。

不构成瓶颈（已排除）：
- **MoE 核心**：两栈几乎逐内核相同（up/gate 5.60 vs 5.38 ms，down 5.38 vs 5.37 ms），本栈反快 0.24 ms。
- **通信**：8 个 rank 中位数均约 22 µs、payload 相同。总和差异仅来自毫秒级长尾（allreduce 等慢 rank），且 T-Head 长尾更大（27,545 µs vs 5,162 µs）却整体更快——**将通信计入差距会得出相反结论**。

---

## 4. 复现与部署说明

### 4.1 代码获取

```bash
# vLLM 0.24.0 官方 empty 构建（本体零修改）
# editable 安装于 /workspace/vllm-0.24.0

# 插件：upstream main@f4319bd + 7 提交
cd /workspace/vllm-plugin-FL
git log --oneline f4319bd..HEAD     # ab2cd7d 026ab85 617a867 3474000 9bebd1d 218bf47 f615560
git status --short                  # 4 个未提交文件（本轮改动）

# FlagGems：v5.3.4 + 6 提交
cd /workspace/FlagGems-v5.3.4
git log --oneline v5.3.4..HEAD      # efd9e35e a597ec4c 26ea167e ae6fd75e 8d3179b0 c993b3c8
```



### 4.2 部署命令

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_FL_DSV4_TORCH_COMPILE=1
export VLLM_FL_EMPTY_INT_ZERO=1          # 必须为 1，=0 会在 graph capture 阶段 IMA 崩溃

vllm serve /mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken \
  --tensor-parallel-size 8 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 32768 \
  --chat-template /workspace/pytorch/chat_template/chat_template_deepseekv4.jinja \
  --trust-remote-code \
  --no-enable-log-requests \
  --no-enable-prefix-caching \
  --compilation-config.cudagraph_mode FULL_AND_PIECEWISE \
  --compilation-config.cudagraph_capture_sizes '[1,2,4,8,12,16,24,32,40,48,56,64,96,128,192,256,320,384,448,512]' \
  --compilation-config.compile_sizes '[1,2,4]' \
  --port 8011
```

封装脚本：`/workspace/pytorch/serve_opt_cards8_15.sh`（`CARDS=` 可换卡组，默认 8–15）。

FlagGems 黑名单由 `thead.yaml` 提供，**无需设置环境变量**。

### 4.3 关键运维要点

**启动就绪判据**——不要轮询 `/health`：引擎未就绪时它已返回 200，据此判断会把残留的孤儿服务误判为新服务已就绪。使用日志中的 `Application startup complete`：

```bash
bash /workspace/pytorch/wait_ready.sh <serve_log> 900
# 输出 KV cache 大小、padding patch 状态、配置加载数、致命错误数
```

**关闭服务**——不要用 `pkill -f`（本机与其他容器共享），也不要 grep 日志取 worker PID（本构建同时记为 `Worker pid=` 与 `Worker_TP<n> pid=`，只 grep 后者会漏杀 8 个 worker，导致 90 GB/卡 泄漏，下次启动死于 `Free memory on device (6.64/95.62 GiB)`）。按**进程树**枚举并轮询驱动确认显存回收：

```bash
bash /workspace/pytorch/stop_server.sh
```

**⚠️ 切换 padding 配置必须清 torch.compile 缓存**：AOT 缓存不将 `padded_heads` 纳入 key，两种配置命中同一哈希。8 头缓存喂 64 头 q 必崩：

```
AssertionError: output buffer shape [32768, 8, 512] must match q shape [32768, 64, 512]
```

```bash
mv /root/.cache/vllm/torch_compile_cache /root/.cache/vllm/torch_compile_cache.bak
# 首次启动需额外 4–5 分钟重编译
```

**关键开关**：

| 环境变量 | 作用 |
|---|---|
| `VLLM_FL_Q_HEAD_PADDING=1` | 恢复上游 Q head padding（回退 K1） |
| `VLLM_FL_FLAGOS_BLACKLIST=__none__` | 关闭 FlagGems 黑名单（回退 K2；env 优先于 yaml） |
| `VLLM_FL_DEEPGEMM_CONFIGS=0` | 禁用 deepgemm 调优配置加载（回退 F3） |
| `VLLM_FL_DISABLE_DEEPGEMM_MOE=1` | MoE 回退 triton 实现 |
| 不设 `VLLM_FL_DSV4_TORCH_COMPILE` | 回退 breakable cudagraph |

### 4.4 压测复现

```bash
# 三用例标准压测（plugin 自带 harness，产出 benchmark_results/summary_*.csv）
cd /workspace/vllm-plugin-FL
python3 benchmarks/benchmark_throughput_serve.py          # 默认 3 用例，4 轮跳首轮
python3 benchmarks/benchmark_throughput_serve.py --enable-all   # 全部 10 用例

# 单用例（case1）
bash /workspace/pytorch/bench_case1.sh <logfile>
```



```bash
bash /workspace/pytorch/ab_boots.sh <tag> off on off on
```

同主机其他租户会竞争 host CPU，而 decode 有相当时间花在 host 侧 `prepare_inputs`，务必记录压测期间其他卡的利用率。

### 4.5 正确性验证

```bash
# native ATen 回落算子的数值等价性（28 项：非连续 strided 源、跨 dtype 拷贝、
# 0 元素张量、张量形式 clamp 边界、全部 in-place 变体、cat.out、带 step 的 arange）
CUDA_VISIBLE_DEVICES=0 MODE=off python3 /workspace/pytorch/verify_native_ops.py

# 端到端生成（temperature 0）：算术 17*23=391、速度题 120 km/h、
# 多轮记忆 42*2=84、折扣反推 50、Python 一行、中文张量并行解释、前 10 个质数
# 全部与优化前逐字一致
```

**已知缺口**：压测使用 random + `ignore-eos`，不验证输出质量；上述端到端检查为小样本探针。正式发布前建议补一次**真实 prompt 集的精度回归**。

### 4.6 Profiling

vLLM 0.24.0 **已移除 `VLLM_TORCH_PROFILER_DIR`**（设置它只会打印 "Unknown vLLM environment variable"，profile 路由不注册，`/start_profile` 返回 404）。改用：

```bash
--profiler-config.profiler torch \
--profiler-config.torch_profiler_dir /abs/path \
--profiler-config.torch_profiler_with_stack false \
--profiler-config.ignore_frontend true \
--profiler-config.max_iterations 12       # 限定 12 个 engine step，自动停止并 flush
```

```bash
bash /workspace/pytorch/capture_decode_profile.sh      # 稳态 decode 采样
python3 /workspace/pytorch/analyze_timeline.py  <trace.gz> --iters 12   # busy-union / 空转
python3 /workspace/pytorch/analyze_gaps.py      <trace.gz> --iters 12   # 空转归因
python3 /workspace/pytorch/final_attribution.py <thead.gz> <fl.gz>      # 两栈逐项对比
```

**⚠️ 不要在 `max_iterations` 之外再调 `/stop_profile`**：会写出重复 trace 并**拆掉 profiler 子系统**，此后 `/start_profile` 无法连接（curl 000）。

**分析口径**：decode 运行在 3–4 条并发流上，**必须用 busy-union 而非 kernel 时间总和**（总和高估约 4–13%）。通信内核时长包含等待慢 rank 的成分，比较时应看中位数而非总和。

### 4.7 T-Head 基线复现（对照用）

厂商栈已卸载但完整备份，可与现栈并存运行，无需改动当前安装：

```bash
# /workspace/vllm-thead-ref 本身即 vllm/ 包目录（含编译内核 _C.abi3.so / _moe_C.abi3.so）
mkdir -p /tmp/thead_stack && ln -sfn /workspace/vllm-thead-ref /tmp/thead_stack/vllm
export PYTHONPATH=/tmp/thead_stack
export VLLM_PLUGINS=          # 必须清空，否则 vllm-plugin-fl 会注入 0.20.1
python3 -m vllm.entrypoints.cli.main serve <model> ...   # 不能用 vllm 命令（属 0.24.0）
```

脚本：`/workspace/pytorch/serve_thead_profile.sh`。实测复现 1726.54 tok/s / TPOT 33.23 ms。其 0.20.1 fork 已带同样的 `--profiler-config.*` 机制，故两栈可用完全一致的采样方法对比。

---

