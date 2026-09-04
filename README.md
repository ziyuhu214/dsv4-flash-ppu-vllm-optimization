# DeepSeek-V4-Flash 在 T-Head PPU 上的 vLLM 推理性能优化 — 项目报告

DeepSeek-V4-Flash（W8A8-INT8）在 T-Head PPU（真武 ZW810E ×8）上，用**开源社区技术栈**（官方 vLLM 0.24.0 + FlagGems + vllm-plugin-FL）替代厂商定制栈的推理性能优化项目。

代码改动以 PR 形式提交在两个配套仓库：

| 仓库 | PR 分支 | 内容 |
|---|---|---|
| [ziyuhu214/vllm-plugin-FL](https://github.com/ziyuhu214/vllm-plugin-FL/pull/1) | `feat/deepseek-v4-flash-ppu-perf` | DeepSeek-V4 PPU op 移植、int8 GEMM/MoE 路径、CUDA graph、empty 定向修复、去 Q head padding、FlagGems 黑名单等（12 个提交） |
| [ziyuhu214/FlagGems](https://github.com/ziyuhu214/FlagGems/pull/1) | `fix/deepseek-v4-ppu-fixes` | fp8 编码、sqlite 锁、int64 偏移、aten::empty 接管移除等内核改动（6 个提交） |

**主报告：[`DEPLOYMENT_AND_OPTIMIZATION_REPORT.md`](DEPLOYMENT_AND_OPTIMIZATION_REPORT.md)** —— 部署与优化的完整技术报告（环境、逐项优化的原因与证据、收益归因、无效尝试清单、运维要点、profiling 方法）。

过程存档另见 [`PORTING_REPORT.md`](PORTING_REPORT.md)（移植过程、算子归属图）与 [`MODIFICATIONS_REPORT.md`](MODIFICATIONS_REPORT.md)（逐 commit 核查）。**三份报告中数据不一致处，以本 README 与主报告为准**（均为 09-04 同批实测）。

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
| 部署参数 | TP=8，max-model-len 65536，fp8 KV cache（fp8_ds_mla，**198,758 tokens**），util **0.85**（三用例均通过），batch 32768，无 prefix caching，`VLLM_FL_DSV4_TORCH_COMPILE=1` + FULL_AND_PIECEWISE cudagraph |

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

全部为 09-04 同批实测（`results/summary_20260904_111410.csv`），测量期间 16 张卡无其他租户干扰：

| 用例 (prefill/decode/conc) | T-Head 栈 tok/s | 社区栈（本项目最终） | 比例 | 社区栈 TPOT | 社区栈 TTFT |
|---|---|---|---|---|---|
| 1024/1024/64 | 1719.8 | **1633.11** | **95.0%** | 35.12 ms | 4.18 s |
| 4096/1024/64 | 1161.4 | **974.86** | **83.9%** | 55.01 ms | 10.89 s |
| 16384/1024/64 | 480.6 | **411.86** | **85.7%** | 127.96 ms | 26.21 s |

TPOT 差距相对上一版收窄 61–83%；TTFT 已进入 T-Head 的 1.03–1.08 倍（4.18 vs 3.96 s、10.89 vs 10.10、26.21 vs 25.40）。

优化演进（case1）：初次跑通 706 → 1301 → 1486.9 → **1633.11 tok/s（累计 +131%）**；TPOT 82.9 → 44.0 → 38.0 → **35.12 ms**。本轮（09-04）增益来自去 Q head padding + FlagGems 黑名单，见 §3.1 第 8–9 项。

> **长输入读数偏保守**：case1 四轮平坦（1638.9 / 1636.5 / 1634.1 / 1628.7），但 4096 与 16384 两档在四轮内单调爬升（16384：370.9 → 394.5 → 417.5 → 423.6），harness 的跳首轮未能完全去除长输入预热趋势，均值低估稳态（16384 后两轮均值 420.6 vs 报出 411.86）。基线由同一 harness 产出，故比例同口径、结论不受影响。基线数据 `results/raw_runs_20260826_111911.csv`（本次另复现 1726.54 tok/s，与存档 1719.78 一致）。

### 2.2 vs H100（跨硬件，衡量硬件代际差）

H100 数据（8×H100 80GB，vLLM 0.26.0，FP8 模型）：

| 用例 | H100 tok/s | PPU 社区栈 tok/s | PPU/H100 | 上一版 | H100 TPOT | PPU TPOT |
|---|---|---|---|---|---|---|
| 1024/1024/64 | 3328.9 | **1633.11** | **49.1%** | 44.7% | 18.3 ms | 35.12 ms |
| 4096/1024/64 | 2530.5 | **974.86** | **38.5%** | 27.7% | 23.4 ms | 55.01 ms |
| 16384/1024/64 | 1272.4 | **411.86** | **32.4%** | 26.3% | 44.2 ms | 127.96 ms |
| 65536/1024/64 | 392.5 | —（未测） | — | — | 138.3 ms | — |

绝对吞吐为 H100 的 32–49%，而两者标称算力比仅 12.4%（1968 TOPS vs 15832 TFLOPS）——见下节效率口径。

> ⚠️ H100 数据为外部引入（另一台机器所测），本容器内无对应 CSV/日志。

### 2.3 单位算力效率（tok/s per TOPS，8 卡合计算力为分母）

统一口径：两边都取 **BF16 峰值 ×2** 作为 INT8/FP8 等效算力——PPU 123 TFLOPS ×2 = 246 TOPS/卡（8 卡 1968T），H100 SXM BF16 989.5 TFLOPS ×2 = 1979 TFLOPS/卡（8 卡 15832T，即官方 FP8 dense 峰值）。与两边部署的主 GEMM 精度（PPU W8A8-INT8 / H100 FP8）对应。

| 用例 | PPU tok/s/TOPS | H100 tok/s/TFLOPS | PPU/H100 效率比 | 上一版效率比 |
|---|---|---|---|---|
| 1024/1024/64 | **0.830** | 0.210 | **3.95×** | 3.59× |
| 4096/1024/64 | **0.495** | 0.160 | **3.10×** | 2.23× |
| 16384/1024/64 | **0.209** | 0.080 | **2.60×** | 2.11× |

PPU 每单位标称算力的实际 token 产出为 H100 的 **2.6–4.0 倍**（调优前约 1.7 倍）。这既反映 PPU 架构在此负载上的实际利用率较高，也说明 H100 的标称 FP8 峰值在 MoE decode 这类**权重加载受限**负载下难以兑现。

## 3. 优化内容

改动全部落在 vllm-plugin-FL 与 FlagGems（vLLM 0.24.0 本体未改动），共 18 个提交。每项的原因、证据与实测收益详见 [`DEPLOYMENT_AND_OPTIMIZATION_REPORT.md`](DEPLOYMENT_AND_OPTIMIZATION_REPORT.md) §3。

### 3.1 vllm-plugin-FL（[PR #1](https://github.com/ziyuhu214/vllm-plugin-FL/pull/1)，12 提交）

1. **移植 PPU deep_gemm int8 GEMM/MoE 路径**（`ppu_deep_gemm*.py` + 调优配置）：厂商包装层移植，MoE 与 dense 层直连 PPU deep_gemm wheel；用 `tuning/tune_dsv4_deepgemm.py` 对本部署全部 GEMM 形状自动调优，产出 DSv4-tp8 专属 153 条配置（decode M 1–1024 + prefill M 1536–32768）。
2. **DeepSeek-V4 专用 op 与运行时补丁**（15 个补丁 + 8 个 op 文件）：int8 sparse-attn indexer（deep_gemm `int8_(paged_)mqa_logits` 替代 NVIDIA-only 路径）、fp8_ds_mla KV cache compress/insert/gather、o_proj int8、mHC opaque op 化、`_C` 算子 triton/FlagGems 桥接、torch.compile 全图模式（`VLLM_FL_DSV4_TORCH_COMPILE=1`）。
3. **acext INT8 W8A8 scaled-MM 线性内核**：厂商 kernel 类移植，行主序权重直入 `acext.int8_gemm`，thead 平台内核 oracle 首选。
4. **thead 启用 CUDA graph + 显存核算修复**：`support_static_graph_mode` 白名单、估算门控 `is_cuda()`→`is_cuda_alike()`、BreakableCUDAGraphWrapper 纳入 profiling、修复 allocator 缓存虚增 ~90 GiB 的估算 bug。单项收益最大（decode 从全 eager 到全图）。
5. **qnorm 插入头填充快速路径**：padding 头槽位（FlashMLA 64 头要求 vs TP8 实际 8 头）跳过 RMSNorm/RoPE。
6. **empty 隐式清零的定向修复**（`f615560`，**本轮收益最大：+14.3%**）：FlagGems 移除 `aten::empty` 接管后，暴露出 DeepSeek-V4 路径中隐式依赖清零的消费方。定位到真正的根因是 `DeepseekV4Model.topk_indices_buffer`——上游用 `torch.empty` 分配、每步只写当前 token 的行，而 **CUDA graph replay 会读取整个捕获范围**，padding 行的内存垃圾成为越界 int32 索引，导致 sparse attention gather 触发 IMA。改法：显式 `fill_(-1)`（`-1` 是 indexer 自身的「无索引」哨兵）+ qnorm padding buffer 改回 `torch.zeros` + 新增 `ops/empty_int_zero.py` 作为兜底（只对整数/bool dtype 清零，浮点大激活缓冲完全跳过 kernel）。
7. benchmark 脚本对齐。
8. **去掉 PPU 上的 Q head padding**（`cd2a71b`，**本轮最大收益**）：上游 `get_padded_num_q_heads()` 无条件返回 `64 if num_heads <= 64`，这是 **NVIDIA FP8 decode 内核**的限制，PPU flash_mla wheel 并无此约束（T-Head fork 有 `is_ppu()` 显式分支）。DSv4 共 64 头、TP8 下每 rank 仅 8 真实头，被 pad 到 64 会让 flash_mla 选中慢变体：profile 实测 `flash_sparse_decode_fwd<512,64>` 82.3 µs vs `<512,16>` 51.4 µs、combine 9.9 vs 3.8 µs，合计 **+2.12 ms/step**，是 4.9 ms TPOT 差距中的最大单项。改前已逐项审计 5 个消费方在 `padded_heads == 8` 时均正确退化。连带效果：`o_padded` 由 64 头缩至 8 头，**KV cache 172,324 → 198,758 tokens（+15.3%）**，长输入不再需要降 util 到 0.82。逃生开关 `VLLM_FL_Q_HEAD_PADDING=1`。
9. **FlagGems 算子黑名单 13 项**（`3c78240`，本轮第二大收益，**零代码改动**）：`copy_` / `clamp*` / `cat*` / `arange*` 这类廉价搬运算子，FlagGems 的 triton 实现每次调用的 host 侧开销（Python dispatch + JIT 缓存查找 + launch）远超算子本身，而 native ATen 直接下发 memcpy。decode 时这部分 host 开销不与计算重叠，直接表现为 GPU 空转：cpu_op 合计 18.39 ms/step，而 T-Head 栈（**根本没装 FlagGems**）仅 4.71 ms/step——`aten::copy_` 调用次数相同（54 vs 56）但单次成本高 5–10 倍。隔离实测 native 快 2.8–15.6 倍。**按数据保留了 FlagGems 更快的项**：`zero_` 上 native 反而慢 3.6 倍（PPU 后端 `zero_` 路径差），`zeros`/`zeros_like`/`_to_copy` 收益边际。正确性由 `verification/verify_native_ops.py` 28/28 覆盖。⚠️ 黑名单键匹配的是注册**函数的 `__name__`**（非 aten 算子名），点号写法会**静默失效**——文件内早期注释此处写反了，已更正。
10. **修复 deepgemm 调优配置从未加载**（`85350aa`，两个叠加 Bug）：① 加载器用 `current_platform.get_device_name()` 匹配文件名，该函数返回厂商标签 `thead`，而 12 个配置文件按芯片名 `PPU-ZW810E` 命名 → 签名永不匹配，**153 条 DSv4 调优配置一直是死代码**；② 只修 ① 会在首次 MoE 调用崩溃——本项目自产配置的 `config` 字段是 dict，11 个厂商文件是 repr 字符串，加载器无条件调 `ast.literal_eval()`。配置项 0 → 549。**但受控 A/B（4 次交替启动）实测吞吐效应 −0.05%，即无收益**：decode 时 MoE GEMM 受专家权重加载限制（batch 64、top-6、每 rank 32 专家 ≈ 1.5 token/专家，down-proj 退化为 batched GEMV），tile 调优无法削减搬权重时间。**此项按正确性保留**，不计入性能收益——原记录中「+deepgemm 调优 config → 978 tok/s」的收益应全部归于换用 vendor kernel。
11. **empty() 包装开销优化 + dtype 旋钮**（`6bde0d4`）：该覆盖挂在全进程 `aten::empty` 上，直接计算连续 strides 而非分配 meta 张量读 `.stride()`（host 侧 1.07 → 更低，隔离实测 int32 9.98 → 7.41 µs）。新增 `VLLM_FL_EMPTY_ZERO_DTYPES={all|index}`，收窄到 int32/int64 可减少 365 次 `zeros_kernel` launch（1.19 → 0.78 ms/step），但**实测吞吐仅 +0.13%（噪声内）**，默认保持 `all`。
12. **decode metadata 的 pinned arange 缓存**（`71bff45`，**默认关闭的负收益实验**）：两个 metadata builder 每步各调一次 `pin_memory()`（每次约 480 µs 的全新 pinned 分配）。缓存后 GPU 空转如预期从 5.14 降到 3.91 ms/step，但吞吐反降 1.1%（1500.9 → 1484.8，3 轮一致）——每步改写 `torch.repeat_interleave` 全局量使 Dynamo guard 失效，代价超过收益。保留为已测量的死路存档，`VLLM_FL_PIN_ARANGE_CACHE=1` 可开启。

### 3.2 FlagGems（[PR #1](https://github.com/ziyuhu214/FlagGems/pull/1)，6 提交）

1. **软件 E4M3FN 编码器**：PPU Triton 无 `fp8e4nv` 类型，纯位运算 RNE 实现（含 subnormal），替换 fp8 KV cache 全部 7 处存储点。
2. **sqlite 调优缓存 busy timeout** 5s→120s：修复 TP=8 并发调优 `database is locked` 崩溃。
3. **int64 scale 偏移**（防御性加固）：`cp_gather_indexer_quant_cache` 的 scale 偏移原在 int32 下计算，长上下文下有溢出隐患；按本部署规模（约 2.7k 块）尚未触发，属预防性修复。
4. **qnorm kernel `num_real_heads` 快速路径**（与 plugin 第 5 项配套）。⚠️ 09-04 去掉 Q head padding（§3.1 第 8 项）后**已无优化对象**——不再存在 padding 槽位。该提交此前优化的是本问题的**症状**，保留不影响正确性。
5. **empty_kernel 依赖警告注释**（09-01，已被下一条取代）。
6. **移除 `aten::empty` 接管**（`c993b3c8`，对齐上游 PR #5438）：该 triton kernel 顺带清零，代价是 DeepSeek-V4 decode **每步约 3000 次 kernel 启动**。改由 plugin 侧定向修复承接（见 §3.1 第 6 项）。

## 4. Profiling 结果与剩余优化方向

分析工具在 `profiling/`（采样脚本 + 空转归因 + 两栈逐项对比），摘要存档 `results/profiles/profiler_out_0.txt`。方法与陷阱（0.24.0 已移除 `VLLM_TORCH_PROFILER_DIR`、必须用 busy-union 而非内核时间总和等）见主报告 §4.6。

### 4.1 Decode 每步 GPU 时间分解（09-01 采样，decode 53.6 ms/step）

> 下表为 **09-01 的采样**，早于本轮两项主要优化（去 padding、FlagGems 黑名单）。当时 TPOT 44 ms，现为 35.12 ms。表中前三项与 T-Head 同源的判断仍然成立，`empty_kernel` 与 attention 变体两行已被修复消除。

| 算子 | ms/step | 占比 | 与 T-Head 对比判断 |
|---|---|---|---|
| deep_gemm int8 GEMM（MoE up/down） | 6.6 | 12.4% | 同款内核，无差距 |
| deep_gemm batched_gemvt（dense 小批量） | 5.3 | 9.9% | 同款，无差距 |
| flash sparse decode attention | 4.7 | 8.8% | 同款 wheel，无差距 |
| **empty_kernel（3003 次/步）** | 5.8 | 5.2% | **纯 FL 栈损耗 → 已消除**（结论 1） |
| aiu cutlass GEMM（acext 线性层） | ~3.0 | 5.6% | 同款，无差距 |
| hc_prenorm GEMM（mHC 内 tilelang） | 2.7+1.3 | 7.4% | 同款内核 |
| 通信（twoShot+RING） | 3.8 | 7.2% | pccl 相同；T-Head 或有编译版 custom allreduce 略优 |
| mhc_pre/post tilelang | 3.0 | 5.6% | 同款内核，但**调用次数 ×2**（结论 3） |
| copy_kernel（649 次/步） | 1.2 | 2.3% | opaque op 边界拷贝，FL 栈损耗 |
| per_token_quant_int8（368 次/步） | 1.1 | 2.1% | T-Head 用 C++ 融合量化并进 GEMM 前处理，我们是独立 kernel |

### 4.2 核心结论：差距不在「大算子」，在「粘合部分」

重计算算子（GEMM / attention / MoE，合计约 60%）与 T-Head 完全同源，**这部分没有差距**。剩余差距全部来自 FL 栈的粘合层：

重计算算子与 T-Head 完全同源，**这部分没有差距**——profile 逐内核对比：MoE up/gate 5.60 vs 5.38 ms、down 5.38 vs 5.37 ms（本栈反快 0.24 ms）；通信 8 个 rank 中位数均约 22 µs、payload 相同。差距全部来自 FL 栈的粘合层，且已消除三项主要来源：

**✅ 已消除**

1. **empty_kernel（−5.81 ms/step）**：FlagGems 把 `torch.empty` 换成「写零」triton kernel，每步 3003 次启动。最初判断「删不掉」（下游隐式依赖清零），次日定位真正的依赖方并显式初始化（`topk_indices_buffer.fill_(-1)`）后成功移除。**机制自洽**：profile 测得 5.81 ms/step，实测 TPOT 降 5.95 ms（44.00 → 38.04），相差不到 3%。
2. **attention 慢变体（−2.12 ms/step）**：Q head padding 使 flash_mla 选中 `<512,64>` 而非 `<512,16>`，见 §3.1 第 8 项。
3. **FlagGems 廉价算子的 host 开销**：cpu_op 合计 18.39 ms/step（T-Head 4.71），黑名单后回落 native，见 §3.1 第 9 项。

**剩余差距**（case1 距 T-Head 仅 5%，长输入仍差 14–16%）

1. **流并行度（约 2.26 ms/step）**：T-Head 工作分布在 4 条 aux 流（最大单流 7.40 ms，并行系数 1.13×），本栈为 3 条（最大单流 10.59 ms，1.04×），关键路径取最大流。**机制未查清**——两栈 Python 层均只创建 3 条 aux 流，fan-out 逻辑与门限相同，差异出现在 CUDA graph 捕获结构而非 Python 层。
2. **host 侧 `prepare_inputs` 空转**：T-Head 仅 0.790 ms/step（2.7%），本栈优化前 5.142 ms（13.1%）。§3.1 第 9 项应已显著改善，但**优化后的残余量尚未测量**。
3. **长输入差距更大**说明 prefill 侧另有问题——本项目所有 profile 均针对 decode，需单独采样长输入定位。
4. **mHC 每步跑两遍**（预估可收回约 1 ms/step）：`mhc_pre_big_fuse_with_norm` 8514 次 / `mhc_post` 8429 次，而同层参照算子 `fused_qnorm_rope_kv_insert` 恰为 4257 次（43 层 × 99 步）= 1.00×，即 mHC 精确地 2.00×——`fused_post_pre` 融合路径没被用上，opaque 包装走了拆开路径。

**已验证的死路**（避免重复投入，详见主报告 §3.4）：收窄 empty 零填充 dtype（+0.13%，噪声内）、缓存 metadata 的 pinned arange（GPU 空转如期下降但吞吐 −1.1%，Dynamo guard 失效）、153 条 deepgemm 调优配置（−0.05%，MoE decode 是权重加载受限而非计算受限）。

KV cache 现状：**198,758 tokens**（去 Q head padding 后 +15.3%），三用例已可统一在 util 0.85 下通过，`PORTING_REPORT.md` §七记录的「24 vs 37 GiB、cg 预留 9.1 GB」与「长输入须降 util 0.82」均已过时。

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

# 4. 启动服务（util 0.85 + FULL_AND_PIECEWISE；VLLM_FL_EMPTY_INT_ZERO 必须为 1）
bash scripts/serve_opt_cards8_15.sh            # CARDS= 可换卡组
bash scripts/wait_ready.sh <serve_log> 900     # 就绪判据（勿轮询 /health）

# 5. 压测（plugin 自带 harness，产出 benchmark_results/summary_*.csv）
cd /workspace/vllm-plugin-FL && python3 benchmarks/benchmark_throughput_serve.py

# 6. 关闭服务（勿用 pkill -f）
bash scripts/stop_server.sh
```

**运维要点**（完整版见主报告 §4.3）：

- **就绪判据**用日志里的 `Application startup complete`，不要轮询 `/health`——引擎未就绪时它已返回 200，会把残留的孤儿服务误判为新服务。
- **关服务**用 `stop_server.sh` 按进程树枚举并轮询驱动确认显存回收。`pkill -f` 会误杀同主机其他容器；只 grep `Worker_TP<n> pid=` 会漏杀 8 个 worker，导致 90 GB/卡 泄漏，下次启动死于 `Free memory on device (6.64/95.62 GiB)`。
- **切换 padding 配置必须清 torch.compile 缓存**：AOT 缓存不把 `padded_heads` 纳入 key，两种配置命中同一哈希，8 头缓存喂 64 头 q 必崩（`output buffer shape [32768, 8, 512] must match q shape [32768, 64, 512]`）。`mv /root/.cache/vllm/torch_compile_cache{,.bak}`，首启多花 4–5 分钟。
- **测量方差**：同配置不同次启动差异约 0.9%，轮内 0.16–0.74%。**低于 1.5% 的效应不应视为有效**，用 `scripts/ab_boots.sh <tag> off on off on` 多次启动交替验证。同主机其他租户会竞争 host CPU，而 decode 有相当时间花在 host 侧 `prepare_inputs`。

**逃生开关**：

| 环境变量 | 作用 |
|---|---|
| `VLLM_FL_Q_HEAD_PADDING=1` | 恢复上游 Q head padding |
| `VLLM_FL_FLAGOS_BLACKLIST=__none__` | 关闭 FlagGems 黑名单（env 优先于 yaml） |
| `VLLM_FL_DEEPGEMM_CONFIGS=0` | 禁用 deepgemm 调优配置加载 |
| `VLLM_FL_DISABLE_DEEPGEMM_MOE=1` | MoE 回退 triton 实现 |
| 不设 `VLLM_FL_DSV4_TORCH_COMPILE` | 回退 breakable cudagraph |

**正确性验证**：`CUDA_VISIBLE_DEVICES=0 MODE=off python3 verification/verify_native_ops.py`（28 项 native 回落算子的数值等价性）。⚠️ 压测用 random + `ignore-eos`，只验证吞吐与稳定性；端到端仅做了 temperature 0 的小样本探针，**正式发布前建议补真实 prompt 集的精度回归**。

## 附：目录说明

```
DEPLOYMENT_AND_OPTIMIZATION_REPORT.md  主报告：部署与优化完整技术报告
PORTING_REPORT.md        移植过程存档（算子归属图、早期诊断）
MODIFICATIONS_REPORT.md  逐 commit 核查报告（每条改动的原因与证据出处）
scripts/            服务启动（FL 栈 / thead 基线 / profile）、压测、A-B、就绪与停机脚本
profiling/          profile 采样与分析（timeline busy-union、空转归因、两栈逐项对比）
verification/       native 回落算子的数值等价性测试（28 项）
tuning/             deep_gemm 自动调优脚本与分 shard 输出（tune_out/）
results/            benchmark summary/raw CSV、各阶段日志、profiler 摘要（profiles/）
chat_template/      DeepSeek-V4 chat template（与官方 encoding_dsv4.py 逐字符对齐验证）
```

---
*代码与压测由 Claude Code 完成（2026-08-25 ~ 2026-09-04）。*

## 附录：原始数据

逐轮原始数据，按时间倒序。**A.0 为当前最优配置的三用例完整数据（09-04）**，§2 各表均取自此；A.1 为上一版（09-02，含 empty 修复）；A.2–A.4 为 08-31 三用例；A.5 为 T-Head 厂商栈基线。

### A.0 三用例最终数据（09-04，当前最优）

来源：`results/raw_runs_20260904_111410.csv`、`results/summary_20260904_111410.csv`。对应代码为 plugin 12 提交（本轮新增 `cd2a71b` 去 Q head padding、`3c78240` FlagGems 黑名单、`85350aa` deepgemm 配置加载、`6bde0d4` empty 包装优化）+ FlagGems 6 提交。**三用例统一 util 0.85，全部 256/256 成功**，压测期间 16 张卡无其他租户干扰。

**1024/1024/conc64** — 去首轮均值 **1633.11 tok/s / TPOT 35.12 ms**：

| Run | Duration (s) | Output tok/s | Peak tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|---|---|
| 1（跳过） | 159.95 | 1638.93 | 2012 | 4180.79 | 34.98 |
| 2 | 160.18 | 1636.51 | 1984 | 4178.86 | 35.04 |
| 3 | 160.42 | 1634.13 | 1984 | 4192.70 | 35.09 |
| 4 | 160.95 | 1628.70 | 1984 | 4173.34 | 35.24 |

**4096/1024/conc64** — 去首轮均值 **974.86 tok/s / TPOT 55.01 ms**：

| Run | Duration (s) | Output tok/s | Peak tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|---|---|
| 1（跳过） | 343.35 | 763.49 | 1792 | 15469.26 | 66.80 |
| 2 | 292.84 | 895.18 | 1728 | 12411.40 | 58.66 |
| 3 | 258.05 | 1015.88 | 1728 | 10136.34 | 53.11 |
| 4 | 258.65 | 1013.51 | 1728 | 10133.08 | 53.27 |

**16384/1024/conc64** — 去首轮均值 **411.86 tok/s / TPOT 127.96 ms**：

| Run | Duration (s) | Output tok/s | Peak tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|---|---|
| 1（跳过） | 706.78 | 370.90 | 1728 | 27384.58 | 142.11 |
| 2 | 664.54 | 394.48 | 1728 | 26390.53 | 131.11 |
| 3 | 627.92 | 417.48 | 1728 | 25991.65 | 127.64 |
| 4 | 618.82 | 423.62 | 1728 | 26253.56 | 125.13 |

⚠️ **长输入读数偏保守**：case1 四轮平坦（极差 0.6%），但 4096 与 16384 两档在四轮内单调爬升——跳首轮未能完全去除长输入的预热趋势，均值低估稳态（16384 后两轮均值 420.6 vs 报出 411.86；4096 后两轮 1014.7 vs 974.86）。基线由同一 harness 产出，故 §2.1 的比例同口径可比，但长输入**绝对值应视为下界**。

### A.1 Case1 + empty 定向修复：1024/1024/conc64/256 prompts（util 0.85，上一版）

来源：`results/logs/case1_emptyfix_v2.log`（2026-09-02 16:38）。对应代码为 plugin `f615560` + FlagGems `c993b3c8`。4 轮全部 256/256 成功：

| Run | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|---|
| 1（warmup） | 1336.84 | 2673.69 | 7475.99 | 40.59 |
| 2 | 1484.06 | 2968.11 | 5135.23 | 38.13 |
| 3 | 1489.26 | 2978.53 | 5138.00 | 37.97 |
| 4 | 1487.31 | 2974.61 | 5142.58 | 38.03 |

去首轮均值：**1486.88 tok/s / TPOT 38.04 ms**，轮间方差极小（±0.2%）。相对 A.2（1301.1）**+14.3%**，达当时 T-Head 基线的 86.5%。

### A.2 Case1：1024/1024/conc64/256 prompts（util 0.85）

来源：`results/raw_runs_20260831_145409.csv`。4 轮全部 256/256 成功：

| Run | Duration (s) | Output tok/s | Peak tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT | P99 TTFT | Mean TPOT (ms) | Median TPOT | P99 TPOT | Mean ITL (ms) | P99 ITL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 201.51 | 1300.88 | 1600 | 2601.76 | 5306.21 | 4412.90 | 7533.35 | 44.04 | 44.46 | 48.02 | 44.04 | 48.12 |
| 2 | 200.94 | 1304.57 | 1600 | 2609.14 | 5290.56 | 4375.84 | 7490.08 | 43.91 | 44.65 | 48.21 | 43.91 | 48.19 |
| 3 | 200.97 | 1304.42 | 1600 | 2608.83 | 5291.60 | 4391.35 | 7507.17 | 43.92 | 44.50 | 48.03 | 43.92 | 48.16 |
| 4 | 202.52 | 1294.40 | 1600 | 2588.80 | 5435.88 | 4384.44 | 7500.00 | 44.16 | 45.09 | 48.44 | 44.16 | 48.38 |

同一 CSV 中 4096/16384 档位在 util 0.85 下失败（inductor 图执行 OOM，服务中断），因此这两档降 util 至 0.82 后用 `scripts/rerun_cases23.sh` 重跑，即 A.3/A.4。（09-04 去 Q head padding 后 KV cache 增大，三用例已可在 0.85 下全部通过，见 A.0。）

### A.3 Case2：4096/1024/conc64/256 prompts（util 0.82）

来源：`results/logs/bench_cases23_final.log`。4 轮全部 256/256 成功：

| Run | Duration (s) | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | P99 TTFT | Mean TPOT (ms) | P99 TPOT |
|---|---|---|---|---|---|---|---|---|
| 1 | 380.97 | 0.67 | 688.09 | 3440.44 | 16476.35 | 36476.07 | 76.15 | 102.41 |
| 2 | 406.46 | 0.63 | 644.94 | 3224.72 | 16390.73 | 32858.03 | 81.12 | 99.70 |
| 3 | 318.02 | 0.80 | 824.31 | 4121.54 | 12960.68 | 32796.06 | 65.00 | 76.69 |
| 4 | 412.12 | 0.62 | 636.09 | 3180.43 | 17990.40 | 38905.37 | 81.34 | 105.41 |

### A.4 Case3：16384/1024/conc64/256 prompts（util 0.82）

来源同 A.3。4 轮全部 256/256 成功：

| Run | Duration (s) | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | P99 TTFT | Mean TPOT (ms) | P99 TPOT |
|---|---|---|---|---|---|---|---|---|
| 1 | 902.74 | 0.28 | 290.39 | 4936.58 | 37183.91 | 163123.86 | 173.83 | 199.65 |
| 2 | 799.83 | 0.32 | 327.75 | 5571.74 | 33067.73 | 143241.26 | 162.66 | 193.35 |
| 3 | 776.49 | 0.33 | 337.60 | 5739.26 | 32681.06 | 143228.59 | 157.65 | 186.96 |
| 4 | 775.23 | 0.33 | 338.15 | 5748.54 | 32684.00 | 143247.92 | 157.34 | 186.46 |

汇总口径说明：§2 各表引用的值均为**4 轮去首轮取均值**（跳首轮以去预热）——现全部取自 A.0（1633.11 / 974.86 / 411.86）。A.1–A.4 为历史各阶段数据，保留以便复核演进。

### A.5 T-Head 厂商栈基线（2026-08-26，同硬件同压测参数）

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

§2.1 引用的基线值同样为去首轮均值：1719.8 / 1161.4 / 480.6 tok/s。（09-04 另在同容器复现 1726.54 tok/s / TPOT 33.23 ms，与本表一致。）
