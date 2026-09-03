# 两个插件（vllm-plugin-FL / FlagGems）修改与优化报告

**核查日期**：2026-09-02 · **核查方式**：逐个 commit 读 diff + 运行时探针 + 原始压测日志复算
**基线**：T-Head 定制 vLLM 0.20.1+ppu 栈 · **硬件**：8× PPU-ZW810E 96GB，TP=8
**模型**：DeepSeek-V4-Flash-0731 W-INT8 per-channel / A-INT8 per-token

> 本报告的每条结论都标注了证据出处（commit / 文件:行 / 日志文件 / 运行时探针）。
> 与既有 `PORTING_REPORT.md`、`README.md` 不一致处，均以本次实测为准并显式标出。

---

## 0. 环境与改动范围（已核实）

| 项 | 实测值 | 核查方式 |
|---|---|---|
| vLLM 本体 | `0.24.0+empty`，editable `/workspace/vllm-0.24.0` | `git status` 无 diff、仅 1 个 commit、无非 build 未跟踪文件 → **零修改成立** |
| vllm-plugin-FL | 分支 `feat/deepseek-v4-flash-ppu-perf`，上游 `f4319bd` 之上 **7 个 commit** | `git log f4319bd..HEAD` |
| FlagGems | 分支 `fix/deepseek-v4-ppu-fixes`，上游 `f7c55cb2` 之上 **6 个 commit** | `git log f7c55cb2..HEAD` |
| PPU wheel（全程未动） | torch 2.10.0 / triton 3.6.0+ppu / deep_gemm 1.0.0+ppu / flash_mla 2.0.0+ppu / acext 1.0.0 | `pip list` |
| 依赖升级 | compressed-tensors 0.17.0（0.24.0 要求） | `pip list` |

**文档陈旧提示**：`README.md` 写「plugin 6 提交 / FlagGems 5 提交」，`PORTING_REPORT.md` 写「5 / 4」。
实际为 **7 / 6**。最新一对 commit（`f615560` + `c993b3c8`，均 2026-09-02 16:45:23）两份文档都未收录，
且**尚未推送**（远端仍在 `218bf47` / `8d3179b0`），故 README 引用的两个 PR 不含这对修改。

**关于 commit 时间**：Aug-31 的 6 个 commit 全部在 67 秒内写入（`026ab85` 12:15:50 → `9bebd1d` 12:16:08）。
即 git 历史是收尾时按逻辑重新切分的结果，**不是开发时序**。真实时序证据在 `/workspace/port_attempt*.log`
（Aug 26）、`moe_port_test*.log`（Aug 27）、`bench_case1_*.log`（Aug 28–31）。这一点影响对「里程碑」的解读，见 §3。

---

## 一、算子部分

### 1.1 FlagGems（6 个 commit）

#### ① `efd9e35e` 软件 E4M3FN 编码器 —— 原因：**编译期直接报错，跑不起来**

- 文件：`src/flag_gems/fused/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py`
- 原实现 `x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)` 在 PPU triton 上抛：
  ```
  ValueError: type fp8e4nv not supported in this architecture.
              The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')
  ```
  证据：`/workspace/port_attempt13.log`（真实报错原文，非推测）。
- 改法：新增 `_encode_e4m3fn`，纯整数位运算实现 RNE（round-to-nearest-even），含 normal 与 subnormal
  两条路径、mantissa 进位溢出处理、subnormal→normal 的 carry；输入要求预 clamp 到 ±448。
- 替换点数：diff 中新增 8 处 `_encode_e4m3fn(`，删除 7 处 `float8e4nv`。即 **1 个函数定义 + 7 个存储点**，
  与 commit message 的「seven store sites」一致。
- 性质：**构建/兼容性阻塞修复**（不是精度问题——目标就是与硬件 cast 位级一致）。
- ⚠️ 待确认项：编码器未见 NaN/Inf 分支（依赖调用方预 clamp），也未做「硬件支持 fp8e4nv 时走原生路径」的
  能力分支。在支持 fp8e4nv 的后端上这会是纯软件开销。容器内无位级对比测试产物，**「位精确」未被独立验证**。

#### ② `a597ec4c` sqlite busy timeout 5s→120s —— 原因：**TP=8 并发崩溃**

- 文件：`src/flag_gems/utils/models/sql.py:46`
- 8 个 TP worker 共用同一份 tuning cache DB 文件，同时 tune 新 shape 时默认 5s busy timeout 直接
  `database is locked` 杀 worker。改为仅对 `sqlite` 前缀 URL 传 `connect_args={"timeout": 120}`。
- 性质：**并发崩溃修复**（基础设施类，不影响数值）。

#### ③ `26ea167e` int64 scale 偏移 —— 原因：**int32 溢出（正确性加固）**

- 文件：`src/flag_gems/fused/cp_gather_indexer_k_quant_cache.py:87-95`
- `block_id * kv_cache_scale_stride + block_offset * num_quant_blocks` 原在 int32 下计算；
  改为 `block_id.to(tl.int64)` / `block_offset.to(tl.int64)`，与上方 data-offset 的算法保持一致。
- ⚠️ **修正 commit message**：message 称 stride(0) 约 26 万 fp32 元素/块、超过约 8k 块即溢出。
  该 stride 是运行时 `k_cache_scale.stride(0)`（`cp_gather_indexer_k_quant_cache.py:169` 传入），
  容器内**无法确认 26 万这个数值**。而本部署 KV cache 为 172,329 token，
  block_size 64/128 分别只有约 2,692 / 1,346 块。
  → 结论：这是**正确的防御性加固**，但按当前压测规模推算，**并非当时正在触发的 bug**；
  「KV cache >8K 块时 IMA 崩溃」的表述缺少证据支撑，建议改为「长上下文下的溢出隐患」。

#### ④ `ae6fd75e` qnorm kernel `num_real_heads` 快速路径 —— 原因：**性能（无用计算）**

- FlashMLA 要求 64 头，TP8 实际只有 8 个真实头。新增 `num_real_heads: tl.constexpr = -1`，
  当 `num_real_heads >= 0 and (not is_kv) and slot_idx >= num_real_heads` 时直接
  `tl.store(零)` 并 return，跳过 load + RMSNorm + RoPE 全部计算。
- 默认 `-1` 关闭，向后兼容；由 plugin 侧 `_C_ops_registry.py:117` 传入实参。
- ⚠️ 语义变化：padding 槽位从「对 0 做 RMSNorm/RoPE 得到的值」变为「直接写 0」。两者在数学上
  应当一致（`0/sqrt(eps)` 仍为 0），但不再是逐位同一条计算路径。
- 实测收益：见 §3，**case1 上未测出可辨识增益**。

#### ⑤⑥ `8d3179b0` + `c993b3c8` empty() 零填充 —— 一对**互相矛盾**的 commit，需注意最终状态

这是本项目最有价值的一条发现，也是唯一一处「文档结论被后续实测推翻」的地方。

- `8d3179b0`（09-01）在 `ops/empty.py:86` 加注释警告：**不要**删除 zero-store kernel，
  「看起来是 6% 的白拿收益，但下游隐式依赖它，删了会在 dsv4 compressor state cache 触发 IMA」。
- `c993b3c8`（09-02，次日）**恰好做了被警告的事**：在 `src/flag_gems/__init__.py:422` 把
  `("empty.memory_format", empty)` 注释掉，对齐上游 PR #5438，不再接管 `aten::empty`。
- ⚠️ **`8d3179b0` 的警告注释现仍留在 `ops/empty.py` 中，已经过时且会误导后人**——
  它警告的操作在同一分支上已经被执行了。建议清理或改写为「已由 plugin 侧定向修复取代」。
- 解决方式不是「继续全量清零」，而是**定位真正的依赖方并逐个显式初始化**（见 §1.2 ④）。

### 1.2 vllm-plugin-FL 算子侧

#### ① `ab2cd7d` 移植 PPU deep_gemm int8 GEMM/MoE 通路 —— 原因：**能力缺失 + 性能**

- 新增 `ppu_deep_gemm.py`（887 行）/ `ppu_deep_gemm_utils.py`（765 行）/ `ppu_deep_gemm_moe.py`（705 行），
  自 T-Head vendor fork 移植；11 份厂商预调优 config + 1 份本项目新调优的 DSv4-tp8 config。
- 触发原因（真实报错，`/workspace/port_attempt1.log`）：
  `NotImplementedError: No Int8 MoE backend supports the deployment configuration.`
- **本项目自调优 config 的量化收益（本次新测，两份文档均无此数据）**：
  文件 `DeepSeek-V4-Flash-0731-w8a8-int8-tp8,device_name=PPU-ZW810E-deepgemm_configs.json`
  含 **153 条**，覆盖 26 个 M 值（1 → 32768，decode 段 1–1024 + prefill 段 1536–32768）。
  每条都带 `baseline_time_ms`（deep_gemm 默认启发式）与 `time_ms`（调优后），可直接复算：

  | 指标 | 值 |
  |---|---|
  | 中位加速比 | **1.090×** |
  | 平均 | 1.100× |
  | 最大 | 1.509×（M=128, K=4096, N=4096, 44.9→29.8 ms） |
  | 最小 | 1.011× |
  | 快于默认的比例 | **153/153（100%）** |
  | decode 段（M≤1024，n=75）中位 | 1.104× |
  | prefill 段（M>1024，n=78）中位 | 1.082× |

  即调优是**全面正收益、无回退**，GEMM 层面中位约 9%。

#### ② `026ab85` DeepSeek-V4 专用算子集（8 个文件）+ 运行时补丁包

实现方式已逐个核实（`@triton.jit` 计数 + vendor 调用 + 自定义 op 注册）：

| 文件 | 行数 | triton kernel | 依赖/性质 | 缺失原因 |
|---|---|---|---|---|
| `deepseek_v4_ppu_indexer.py` | 460 | 0 | 包装 PPU deep_gemm `int8_(paged_)mqa_logits`，注册 `fl_ppu_sparse_attn_indexer` | 上游 indexer 走 NVIDIA-only `fp8_fp4_mqa_logits` |
| `deepseek_v4_compress_cache.py` | 463 | 2 | 自写 SM80 kernel，复用 FlagGems 的 `_encode_e4m3fn` | fp8_ds_mla KV cache compress/insert 在 PPU 无实现 |
| `deepseek_v4_dequant_gather.py` | 224 | 2 | 注册 `fl_deepseek_v4_dequant_gather_sm80` | 同上（SM80 路径） |
| `deepseek_v4_indexer_q.py` | 199 | 2 | 注册 `fl_indexer_q_rope_int8` | indexer Q 的 int8 量化 |
| `deepseek_v4_o_proj.py` | 397 | 1 | inv-RoPE triton + `deep_gemm.gemm_int8_int8_bf16_nt` | o_proj int8 通路 |
| `deepseek_v4_mhc_ops.py` | 186 | 0 | opaque op 包装（`fl_mhc_fused_post_pre` 等） | 让 tilelang kernel 能进 torch.compile 图 |
| `deepseek_v4_mhc_native.py` | 169 | 0 | 纯 torch 原生等价实现（备用，未启用） | 留待 PPU inductor 成熟 |
| `deepseek_v4_attn_op.py` | 59 | 0 | 注册 `fl_dsv4_attention` | 同 mhc，图边界 |

**⚠️ 修正 README 的一处描述**：README §性能里程碑把「+ o_proj int8 GEMM」列为一次代码改动带来的提升。
实测 `git show 026ab85:vllm_fl/ops/deepseek_v4_o_proj.py` 显示 **int8 GEMM 路径在该文件首次提交时就已存在**
（`use_int8_gemm` 在第 355 行，`deep_gemm.gemm_int8_int8_bf16_nt` 在第 299 行）。
两条路径（int8 GEMM / fp32 einsum）是**共存的运行时分支**，选择条件为
`n_local_groups == 1 and R % 128 == 0 and 权重形状匹配 and weight_scale 存在`，否则回退 fp32 einsum。
里程碑数字本身是真实测得的（见 §3），但它对应的是开发过程中的状态变化，**在压缩后的 git 历史里已不可见**。

#### ③ `_C_ops_registry.py` / `_C_ops_schemas.py` —— 原因：**`vllm._C` 不存在**

empty 构建没有编译 C++ 扩展，运行时反复出现
`Failed to import from vllm._C: ModuleNotFoundError`（`vllm_dsv4_fl_bench_serve.log`）。
为 `dynamic_scaled_int8_quant`、`silu_and_mul`、`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`
提供 triton/FlagGems 回退，并把 qnorm 算子改造成 0.24.0 的 9 参 head-padding 契约 + Meta 实现（供 torch.compile）。

#### ④ `f615560` empty 零填充的定向修复 —— 原因：**上游移除隐式清零后暴露的真实崩溃**

配合 FlagGems `c993b3c8`。三处改动：
1. 新增 `ops/empty_int_zero.py`：只对**整数/bool** dtype 的 `empty()` 结果清零，浮点大激活缓冲完全跳过 kernel。
   通过 `torch.library.Library("aten","IMPL").impl("empty.memory_format", ..., "CUDA")` 注册。
2. `_patch_topk_indices_buffer_init`：把 `DeepseekV4Model.topk_indices_buffer` 显式 `fill_(-1)`。
   **这是真正的根因**，patch docstring 说得很具体（`deepseek_v4_thead.py:762-776`）：
   上游用 `torch.empty` 分配，每步只写当前 token 的行（`buffer[:num_tokens] = -1`），
   padding 行/未用列保留内存原值；**CUDA graph replay 会读取整个捕获范围** →
   垃圾 int32 索引 → sparse attention 越界 gather → IMA。`-1` 是 indexer 自身的「无索引」哨兵值。
3. 把 qnorm padding buffer 从 `torch.empty` **改回** `torch.zeros`
   （即部分回退 `218bf47`），理由写在 `_C_ops_registry.py:98-101`：不能依赖任何隐式清零；
   但 kernel 内的 `num_real_heads` 快速路径保留——「that was the actual win」。

⚠️ 遗留：`deepseek_v4_compress_cache.py:403-425` 留有调试探针
（`VLLM_FL_DEBUG_COMPRESS_INPUTS=1` 时打印 block_table/slot_mapping 极值），提交前建议清理。

---

## 二、框架部分

### ① `3474000` 启用 CUDA graph + 修复显存核算 —— **单项收益最大**

三处改动，因果链已逐条核实：

1. **`vllm_fl/platform.py:339`** — `support_static_graph_mode()` 白名单加 `"thead"`。
   不加则上游强制 `cudagraph_mode=NONE`，**decode 全程 eager**。这是最初「跑通但极慢」的根因。
2. **`vllm_fl/worker/worker.py:523`** — `current_platform.is_cuda()` → `is_cuda_alike()`。
   因果链：OOT vendor（thead）的 `is_cuda()` 为 False → 跳过 cudagraph 显存预估 →
   `available_kv_cache_memory = requested - non_kv_cache - 0` → KV cache 吃掉本该留给 graph pool 的余量 →
   **capture 阶段 OOM**。
3. **`vllm_fl/worker/model_runner.py:6578`** — 两个子问题：
   - `BreakableCUDAGraphWrapper._all_instances`（DeepseekV4 用的就是它）原先未纳入 profiling pool，
     导致其 graph 显存既没被测量也没被回收，KV cache 定尺后真实 capture 才 OOM。同时补上
     `BreakableCUDAGraphWrapper.clear_all_graphs()`。
   - 两次读数前都插入 `torch.accelerator.empty_cache()`：eager warmup 的激活值滞留在 torch allocator
     缓存里，会把 graph delta 虚增（注释记录 **PPU 上观测到约 90 GiB**）。graph pool 显存不可回收，
     所以 empty_cache 之后的 delta 才能隔离出真实值。

   附带把预估日志从 `logger.debug` 提到 `warning` 并加 `[fl-debug]` 明细，现网日志可见：
   ```
   [fl-debug] requested=81.28 GiB, non_kv=47.20 GiB (weights=36.51, peak_act=9.94,
              non_torch=0.75), cg_est=3.35 GiB -> kv_avail=30.73 GiB
   ```

### ② `617a867` acext INT8 W8A8 scaled-MM 线性内核 —— 原因：**性能（厂商内核优先）**

- 新增 `vllm_fl/quantization/acext_int8_linear.py`（141 行），移植 vendor 的 `PPUInt8ScaledMMLinearKernel`。
- 注册方式（`quant_linear.py:51-61`）：`_POSSIBLE_INT8_KERNELS[PlatformEnum.OOT].insert(0, ...)`，
  即插到 kernel oracle 首位，**外层包 `try/except ImportError`**，acext 缺失时静默跳过。
- `can_implement()` 只在 `not c.input_symmetric` 时返回 False（非对称量化回退 triton）；
  本部署是对称量化，故走 acext。
- 权重按 **row-major [N, K] 原样**喂给 `acext.int8_gemm`（省掉转置/拷贝）；
  激活走 `_C::dynamic_scaled_int8_quant` 桥接做 per-token 动态量化。
- 运行时验证：profile 中 `int8_gemm_ops::int8_gemm` 708 次、775.5 ms（7.05%），确认生效。

### ③ `218bf47` / `9bebd1d`

- `218bf47`：qnorm padding buffer `zeros`→`empty` + 传 `num_real_heads`。后被 `f615560` 部分回退（见 §1.2 ④）。
- `9bebd1d`：仅改 benchmark 脚本（指向 DSv4 int8 checkpoint、端口 8011、去掉 65536 档），无功能影响。

### ④ 补丁包总览 —— **18 个补丁，不是文档写的 15 个**

`vllm_fl/patches/deepseek_v4_thead.py`（823 行），入口 `apply_deepseek_v4_thead_patches()`（第 799 行），
由 `register_model()` 调用，开头 `if vendor_name != "thead": return` 全局门控，各补丁另有
`_fl_*_patch` 标志位保证幂等。实际调用 18 个（第 806–823 行）：

| # | 补丁 | 归类 |
|---|---|---|
| 1 | `_patch_int8_moe_quant_scheme` | 能力准入放宽 |
| 2 | `_patch_int8_moe_deepgemm_backend` | 性能（deep_gemm MoE） |
| 3 | `_patch_flashmla_ops` | 重绑 PPU flash_mla wheel |
| 4 | `_patch_int8_o_proj` | 性能 |
| 5 | `_patch_topk_softplus_sqrt` | torch 回退（能力缺失） |
| 6 | `_patch_moe_align_block_size` | FlagGems 桥接 |
| 7 | `_patch_sparse_indexer_ops` | 能力缺失（NVIDIA-only） |
| 8 | `_patch_indexer_q_quant` | 能力缺失 |
| 9 | `_patch_sparse_indexer_forward` | PPU int8 分发 |
| 10 | `_patch_compressor_cache_insert` | SM80 kernel 移植 |
| 11 | `_patch_indexer_num_sms` | 正确性（PPU 64 SM 修正） |
| 12 | `_patch_asymmetric_capture_sizes` | 显存（PIECEWISE 瘦身） |
| 13 | `_patch_topk_indices_buffer_init` | **崩溃修复（IMA）** |
| 14 | `_patch_empty_int_zero` | 崩溃修复兜底 |
| 15 | `_patch_torch_compile_model` | 性能（全图模式） |
| 16 | `_patch_int8_weights_mapper` | int8 权重名映射 |
| 17 | `_patch_disable_cutedsl` | 禁 CuteDSL（PPU 不支持） |
| 18 | `_patch_dequant_gather` | 能力缺失 |

环境开关共 3 个：`VLLM_FL_DISABLE_DEEPGEMM_MOE`（回退 triton MoE）、
`VLLM_FL_DSV4_TORCH_COMPILE`（全图模式）、`VLLM_USE_BREAKABLE_CUDAGRAPH`。
另有 `VLLM_FL_EMPTY_INT_ZERO`（默认 "1"）、`VLLM_FL_EMPTY_AUDIT`、`VLLM_FL_DEBUG_COMPRESS_INPUTS`。

---

## 三、性能：实测数据与文档的差异

压测口径：`vllm bench serve`，random + ignore-eos，1024/1024，并发 64，256 条，4 轮去首轮。
基线与优化栈使用同一套 serve 参数（TP8 / max-len 65536 / fp8 KV / util 0.85 / batch 32768 / 无 prefix caching）。

### 3.1 **关键发现：最新一轮优化未被任何文档记录**

| 配置 | Output tok/s | Mean TPOT | 占基线 | 数据来源 |
|---|---|---|---|---|
| T-Head 基线 | **1719.8** | 33.4 ms | 100% | `results/raw_runs_20260826_111911.csv` |
| FL 栈（两份文档的「最终」值） | 1301.1 | 44.0 ms | 75.7% | `results/summary_20260831_145409.csv`（1301.13 / 44.0，吻合） |
| **FL 栈 + empty 修复（09-02）** | **1486.9** | **38.04 ms** | **86.5%** | `/workspace/pytorch/case1_emptyfix_v2.log` |

- 后者三轮（去首轮）：1484.06 / 1489.26 / 1487.31 tok/s；TPOT 38.13 / 37.97 / 38.03 ms，方差很小。
- 相对文档值 **+14.3%**，与基线差距由 24.3% 收窄到 **13.5%**。
- 可比性已核实：同一 `serve_dsv4_fl_bench.sh`（util 0.85、`FULL_AND_PIECEWISE`、同 capture sizes）、
  同一 `bench_case1.sh`；HEAD 两个 commit 的时间戳（16:45:23）晚于该压测完成时间（16:38:14），
  且工作树干净 → **该数据对应当前 HEAD 代码**。

### 3.2 机制自洽性验证

profile（`results/profiles/profiler_out_0.txt`，rank0，99 个 decode step）中
`empty_kernel` 297,343 次调用 = **3003.5 次/步**、575.213 ms = **5.81 ms/步**、占 self CUDA 5.23%。
实测 TPOT 下降 **5.95 ms/步**（44.00 → 38.04）。两者相差不到 3%，**机制得到独立印证**。

⚠️ README §4.1 表格写 `empty_kernel` 为「3.4 ms/step、6.3%」。调用次数（3007/步）与该文件一致，
但 ms/步 与占比无法从这份 profile 复算出来；能解释实测增益的是 **5.81 ms/步**这个值。

### 3.3 里程碑（从原始日志复算，非文档转述）

| 阶段 | 日志 | 时间 | 各轮 tok/s | TPOT |
|---|---|---|---|---|
| torch.compile 全图 | `bench_case1_compile.log` | 08-28 17:36 | 1042 / 1099 / 1098 / 1098 | 51.7–54.1 |
| o_proj int8 阶段 | `bench_case1_int8oproj.log` | 08-31 11:20 | 1290 / 1236 / 1290 | 44.1–44.8 |
| final3 | `bench_case1_final3.log` | 08-31 12:05 | 1206 / 1304 / 1300 | 43.8–46.4 |
| qnorm padding 快速路径 | `bench_case1_padopt.log` | 08-31 14:46 | 1189 / 1228 / 1303 | 44.0–46.4 |
| **empty 修复** | `case1_emptyfix_v2.log` | 09-02 16:38 | 1337 / 1484 / 1489 / 1487 | **38.0** |

⚠️ **qnorm padding 快速路径（plugin `218bf47` + FlagGems `ae6fd75e`）在 case1 上没有测出可辨识增益**：
稳态值 1303 vs 前一阶段 1300/1290，落在轮间方差内。两份文档把它列为一项优化，
但**没有支撑其收益的实测数据**。它的实际价值可能体现在其他形状上，或仅是为 `f615560` 铺路。

### 3.4 KV cache（文档数据已过时）

| | tokens | 可用 KV | cudagraph 预留 |
|---|---|---|---|
| T-Head | 203,303 | 36.73 GiB | — |
| FL（当前实测） | 172,329 | 30.73 GiB | 3.35 GiB |

即 **84.8% 的基线容量**。`PORTING_REPORT.md` §七写「KV cache 24 vs 37 GiB」、「cg 预留 9.1GB」，
两个数字都已被后续优化改善（30.73 / 3.35），该遗留事项的严重性应下调。

### 3.5 mHC 重复计算（文档结论已验证成立）

以 43 层 × 99 步 = 4257 为「每层每步一次」的基准：

| kernel | 调用数 | 倍数 |
|---|---|---|
| `fused_qnorm_rope_kv_insert`（参照） | 4,257 | 1.00× |
| `flash_sparse_decode`（参照） | 4,214 | 0.99× |
| `mhc_pre_big_fuse_with_norm` | 8,514 | **2.00×** |
| `mhc_post_tilelang` | 8,429 | 1.98× |

参照算子精确落在 1.00×，mHC 精确落在 2.00× → **「mHC 每步被调用两遍」的结论数值上成立**，
是仍未收割的优化项（预估约 1 ms/步）。

---

## 四、⚠️ 需要注意的运行时状态问题

**当前运行的服务（pid 562731）带 `VLLM_FL_EMPTY_INT_ZERO=0`**（读 `/proc/562731/environ` 得到）。
后果链：`empty_int_zero.register()` 在 `os.environ.get("VLLM_FL_EMPTY_INT_ZERO","1") != "1"` 处返回 False
（`empty_int_zero.py:116`）→ 整数清零兜底**未安装**；而 FlagGems 侧 `aten::empty` 接管也已注释掉
→ **当前配置下完全没有任何 empty 清零**。

日志侧印证：`topk_indices_buffer fully initialized to -1` 出现 8 次（每 worker 一次），
而 `aten::empty int-dtype zero-fill installed` **一次都没出现**，与上述推断一致。

含义有两面：
- **正面**：1486.9 tok/s 是在「零兜底」下取得的，说明 `topk_indices_buffer.fill_(-1)` +
  qnorm 改回 `torch.zeros` 这两处**定向修复已经覆盖了真实依赖点**，兜底并非必需。§1.1 ⑤ 的
  「删不掉」结论确实被推翻了。
- **风险**：该结论目前只有「一次 4 轮压测未崩」的支撑。同日早先两次尝试都失败过——
  `case1_fix_cards8_15.log`（15:34，约 499 tok/s、TPOT 121 ms）与
  `case1_emptyfix_cards0_7.log`（16:19，第 1 轮 499 tok/s，第 2 轮 **0 条成功**，即崩溃）。
  从 499→1487 的跨度说明中间还发生了别的状态变化，而**产生 1487 的那次 serve 日志已在 16:46 被覆盖**，
  无法回溯其确切环境。建议：把 `VLLM_FL_EMPTY_INT_ZERO` 的取值显式写进 serve 脚本，
  并在两种取值下各跑一次长稳测试 + 精度校验后再定稿。

---

## 五、无法在本容器内验证的声明

以下出现在既有文档中，但容器内**找不到支撑数据**，报告对外发布时应补齐或标注：

1. **H100 对照数据**（3328.9 / 2530.5 / 1272.4 tok/s 等）：全库 grep 只命中 `README.md` 自身，
   无任何 CSV/日志。应为另一台机器所测，属于外部引入数据。
2. **精度/正确性结论**（「多轮/数学/中英输出全对，与 T-Head 数值行为一致」）：
   未找到测试脚本或输出产物。且所有压测用的是 random + `ignore-eos` 数据，
   **只能验证「不崩 + 吞吐」，无法验证输出正确性**。§4 的新配置尤其需要一次真实精度回归。
3. **chat template「与官方 `encoding_dsv4.py` 逐字符对齐验证」**：template 文件存在
   （`chat_template/chat_template_deepseekv4.jinja`），但容器内无 `encoding_dsv4.py`，也无对齐测试产物。
4. **`26ea167e` 的 stride ≈26 万**：见 §1.1 ③。

---

## 六、建议的后续动作（按性价比排序）

1. **推送 `f615560` / `c993b3c8`** 并更新两个 PR——当前 PR 不含收益最大的那次修复（+14.3%）。
2. **确定 `VLLM_FL_EMPTY_INT_ZERO` 的最终取值**，写进 serve 脚本，两种取值各做一次长稳 + 精度回归。
3. **补一次精度回归**（非 ignore-eos 的真实 prompt 集），覆盖新配置。
4. **更新 `README.md` / `PORTING_REPORT.md`**：commit 数（7/6）、最终性能（1486.9 / 86.5% / TPOT 38.0）、
   KV cache（30.73 GiB / cg 3.35 GiB）、`empty_kernel` 结论反转、补丁数（18）。
5. **清理**：`ops/empty.py` 中已过时的警告注释；`deepseek_v4_compress_cache.py` 的调试探针。
6. **mHC 重复调用**（§3.5，已数值确认）：让 `fl_mhc_fused_post_pre` 真正走融合路径，预估约 1 ms/步。
7. 复核 `26ea167e` 的实际 stride，修正 commit message 表述。
8. 给 `_encode_e4m3fn` 增加「硬件支持 fp8e4nv 时走原生 cast」的能力分支，便于上游合并。
