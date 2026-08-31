# DeepSeek-V4-Flash INT8 × 官方 vLLM 0.24.0 × vllm-plugin-FL × FlagGems 移植与优化工程报告

日期：2026-08-31 · 硬件：16× PPU-ZW810E 96GB（使用卡 0–7，TP=8）
模型：`T-HEAD/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken`（W8A8-INT8, 43 层 MoE 256 专家, MLA sparse 注意力）

## 一、目标与结果

**目标**：在官方 vLLM 0.24.0（empty 构建，无编译内核）+ vllm-plugin-FL(main) + FlagGems v5.3.4 上跑通
DeepSeek-V4-Flash int8 推理，缺失能力从 T-Head 定制版 vLLM (0.20.1+ppu) 移植，尽量只改 plugin/FlagGems。

**结果**：跑通并正确（多轮/数学/中英输出全对，与 T-Head 版数值行为一致），性能达 T-Head 基线的约 74%：

| Case1 (1024/1024/c64/256) | 输出吞吐 | Mean TPOT | Mean TTFT |
|---|---|---|---|
| T-Head 定制栈（基线） | 1720 tok/s | 33.4 ms | 4.0 s |
| FL 栈·最初跑通 | 706 tok/s (41%) | 82.9 ms | 8.0 s |
| **FL 栈·最终** | **~1270 tok/s (74%)** | **44 ms** | **5.4 s** |

vLLM 0.24.0 本体**零修改**。所有改动落在 vllm-plugin-FL（5 个 commit）与 FlagGems（4 个 commit）。

## 二、修改清单

### vllm-plugin-FL（github.com/flagos-ai/vllm-plugin-FL @ main + 5 commits）

1. `ab2cd7d` 移植 T-Head PPU deep_gemm int8 路径：
   - `vllm_fl/ops/ppu_deep_gemm.py` / `ppu_deep_gemm_utils.py` / `ppu_deep_gemm_moe.py`（PPUDeepGemmExperts int8 MoE）
   - `ppu_deepgemm_configs/`：T-Head 预调优 11 份 + 本项目调优 DSv4-tp8 专属 153 条（decode M 1–1024 + prefill M 1536–32768）
2. `026ab85` DeepSeek-V4 PPU 运行时补丁包 `vllm_fl/patches/deepseek_v4_thead.py`（15 个补丁）+ 5 个 ops 文件：
   - int8 MoE 准入放宽、flashmla 重绑 PPU pip 包、int8 o_proj（fp32 einsum→后升级 int8 GEMM）、
     topk_softplus torch 回退、moe_align/indexer 4 算子桥接 FlagGems、SparseAttnIndexer PPU int8 分发、
     SM80 compressor/dequant-gather kernel 移植、indexer num_sms 修正、int8 权重名映射、禁 CuteDSL、
     `_C` 算子回退（dynamic_scaled_int8_quant/silu_and_mul/qnorm_insert 9 参 schema）
   - torch.compile 全图模式（`VLLM_FL_DSV4_TORCH_COMPILE=1`）：mHC/attention opaque op 化、探测函数冻结、
     splitting_ops 注册
3. `617a867` acext int8 W8A8 线性内核 `vllm_fl/quantization/acext_int8_linear.py`（行主序权重直用，OOT 内核首位）
4. `3474000` CUDA graph 修复：`support_static_graph_mode` 白名单加 thead；graph 显存预估 `is_cuda()`→`is_cuda_alike()`；
   预估纳入 BreakableCUDAGraphWrapper；PIECEWISE 非对称尺寸瘦身
5. `218bf47` qnorm 插入头填充快速路径（zeros→empty + kernel 内零填充）

### FlagGems（github.com/flagos-ai/FlagGems @ v5.3.4 + 4 commits）

1. `efd9e35e` 软件版 E4M3FN 编码器（纯位运算 RNE），替换 PPU triton 编不过的 `tl.float8e4nv` ×7 处
2. `a597ec4c` sqlite 调优缓存 busy timeout 5s→120s（8 TP worker 并发写锁崩溃）
3. `26ea167e` **正确性修复**：`cp_gather_indexer_quant_cache` scale 偏移 int32 溢出（KV cache >8K 块时 IMA 崩溃）
4. `ae6fd75e` qnorm kernel `num_real_heads` 快速路径

### 环境级操作

- 卸载 T-Head vLLM（完整备份 `/workspace/vllm-thead-ref`），装官方 `vllm-0.24.0+empty`（editable, `/workspace/vllm-0.24.0`）
- `compressed-tensors` 0.14→0.17（0.24.0 要求）；PPU 版 torch/triton/deep_gemm/flash_mla/acext 全程未动
- 自制 DeepSeek-V4 chat template `/workspace/pytorch/chat_template/chat_template_deepseekv4.jinja`
  （与官方 encoding_dsv4.py 逐字符对齐验证）

## 三、标准启动命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_FL_DSV4_TORCH_COMPILE=1 \
vllm serve /mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken \
  --tensor-parallel-size 8 --max-model-len 65536 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.82 --max-num-batched-tokens 32768  # 0.85 可跑短输入；长输入需 0.82 \
  --chat-template /workspace/pytorch/chat_template/chat_template_deepseekv4.jinja \
  --trust-remote-code --no-enable-log-requests --no-enable-prefix-caching \
  --compilation-config.cudagraph_mode FULL_AND_PIECEWISE \
  --compilation-config.cudagraph_capture_sizes [1,2,4,8,12,16,24,32,40,48,56,64,96,128,192,256,320,384,448,512] \
  --compilation-config.compile_sizes [1,2,4] \
  --port 8011
```
脚本：`/workspace/pytorch/serve_dsv4_fl_bench.sh`。逃生开关：`VLLM_FL_DISABLE_DEEPGEMM_MOE=1`（回退 triton MoE）、
`VLLM_FL_DSV4_TORCH_COMPILE` 不设（回退 breakable cudagraph）。

## 四、算子归属图（运行时验证）

| 组件 | 实现 | 来源 |
|---|---|---|
| int8 MoE (routed+shared) | PPUDeepGemmExperts + 153 条调优 config | T-Head 移植 |
| dense int8 线性 (wq_b/wo_b/wkv…) | AcextInt8ScaledMMLinearKernel | acext wheel + 本项目内核类 |
| o_proj | inv-RoPE triton + deep_gemm int8 GEMM | T-Head kernel + 本项目 GEMM 化 |
| MLA sparse attention | flash_mla_sparse_fwd | PPU flash_mla wheel |
| indexer mqa logits | int8_(paged_)mqa_logits | PPU deep_gemm wheel |
| indexer cache/topk ×4 | FlagGems triton | FlagGems |
| KV 插入/压缩/收集 | FlagGems qnorm_insert + SM80 移植 kernel | FlagGems + T-Head 移植 |
| mHC | tilelang kernel（opaque op 包装） | 官方 vLLM + 本项目包装 |
| MoE 路由 | torch 回退 | 官方 vLLM |
| 通信 | PYNCCL/pccl（内部自动 oneshot） | PPU pccl |

## 五、关键性能里程碑

| 阶段 | Case1 tok/s | 关键改动 |
|---|---|---|
| 初次跑通（graph 全关，bug：平台白名单漏 thead） | 102 (A口径) | — |
| graph 修复 FULL_DECODE_ONLY | 706 (B口径) | 白名单+估算修复 |
| + deepgemm int8 MoE + 调优 config | 978 | MoE 换 vendor 内核 |
| + torch.compile 全图 (FULL_AND_PIECEWISE) | 1098 | opaque op 化整套 |
| + o_proj int8 GEMM + prefill 调优 | **~1270** | 当前 |

## 六、诊断结论存档（避免后人重走弯路）

- **breakable cudagraph 的 PIECEWISE 每尺寸约 1.6–2.2 GiB 段间驻留** → 尺寸必须稀疏；compile 管线图无此成本但 PPU inductor 有以下限制
- **PPU inductor 限制**：mHC 全原生进图触发 ppu-llc CalledProcessError + 32 分钟/图编译 → 保持 opaque（native 实现留存 `deepseek_v4_mhc_native.py`，1-ulp 等价已验证，工具链升级后可切换）
- **通信已近最优**：pccl 自动 oneshot，decode 中位 15μs；CUSTOM allreduce 需编译 vllm._C（越出约束）
- **indexer decode 已近最优**：0.14ms/层（logits 0.055 + topk 0.085），占 TPOT 14%
- **profile 假象教训**：轻负载 trace 的通信占比 51% 是 warmup 离群值+低占用等待，中位数才可信

## 七、遗留事项与后续方向

1. **KV cache 24 vs 37 GiB**：cg 预留 9.1GB（PIECEWISE opaque 边界静态缓冲）——根治靠 mHC 原生进图（等 PPU inductor 成熟）或手写单 triton kernel（预期收益 <4%，工程 1–2 天，性价比低暂缓）
2. **最后 26% 吞吐差距构成**：opaque op 边界开销 + T-Head C++ 融合算子（qnorm+rope+quant+insert 一体）+
   编译版 custom allreduce。突破需编译 C++ 扩展或工具链升级
3. compile 模式首次启动慢（inductor 编译 range 图约 4 分钟 + 捕获），有 inductor 缓存后约 4–5 分钟
4. 孤儿进程风险：杀服务须连带 `VLLM::Worker`（本项目发生过 3 次占卡事故），推荐
   `ps aux | grep -E 'VLLM|vllm serve' | awk '{print $2}' | xargs kill -9`

## 八、数据存档

- 基线（T-Head 栈）：`benchmark_results/summary_20260826_111911.csv`
- 中间各阶段：`bench_case1_compile.log` / `bench_case1_int8oproj.log` / `bench_case1_final3.log`
- 最终完整 3 用例：case1 `raw_runs_20260831_145409.csv`（util 0.85），case2/3 `bench_cases23_final.log`（util 0.82）
- profile trace 分析结论见第六节；原始 trace 已清理（单份 300MB+）

### 最终三用例总表（4 轮去首轮，256/256 全成功）

| 用例 | T-Head out tok/s | FL 最终 | 比例 | FL TPOT | FL TTFT |
|---|---|---|---|---|---|
| 1024/1024/c64 | 1719.8 | 1301 | **76%** | 44.0 ms | 5.3 s |
| 4096/1024/c64 | 1161.4 | 702 | **60%** | 75.8 ms | 15.8 s |
| 16384/1024/c64 | 480.6 | 334 | **70%** | 159.2 ms | 32.8 s |

注：case2/3 在 util 0.85 下长输入触发 inductor 图执行 OOM，降至 0.82 后全部通过——
compile 模式对长 prefill 的中间显存峰值更敏感（第七节遗留事项 1 的另一表现）。
KV cache 减小也放大了长输入用例的排队（TTFT 差距大于短输入），此为 4096 用例比例偏低（60%）的主因。
