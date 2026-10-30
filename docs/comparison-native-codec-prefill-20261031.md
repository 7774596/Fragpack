# 独立 codec、gather＋FA3 与 fused prefill 对照

2026-10-31，在 `feat/native-attention-external-codec` 分支完成。基础提交 `9edec04`；
GPU core 与上轮 codec 实验相同，本轮仅新增
[对照 benchmark](../benchmarks/native_codec_prefill_bench.py)，没有改 attention 实现或第三方源码。

**原生页上的 FA3 读取比 gather＋FA3 更快；从同一外部归档恢复后再读，总耗时差距明显缩小。**
独立 codec 需要先解压成 raw GPU KV，原生页占用更多 HBM；冷数据是否卸载由缓存策略决定，
两条路线都可以卸载。当前 fused prefill 在这些 GPU Graph 对照中没有胜过 gather＋FA3。

## 三条路径

| 路径 | GPU 持久表示 | 读取过程 | 接入边界 |
| --- | --- | --- | --- |
| Native | 原生 BF16 页 | 外部恢复时一次解压；随后原生 FA3 | 缓存传输/connector，无需换 attention backend |
| Gather | 压缩记录 | 每层 gather 成临时 BF16 workspace，再 FA3 | 自定义 backend 适配压缩存储、gather 和页表；FA3 kernel 不改 |
| Fused | 压缩记录 | 当前自动 split-KV fused prefill，kernel 内解压并计算 | 自定义 backend 和 attention kernel |

两种解压都能产生相同 BF16 KV。区别是是否把还原结果作为原生 KV 长期保存。
workspace 本身不是必须定制 backend 的原因，GPU 持久 KV 布局不符合原生 backend 的预期才是。
本轮没有把独立 codec 接到实际 vLLM connector 或模型服务。

## 数据与协议

- GPU：物理卡 2，H800 PCIe，114 SM，UUID
  `GPU-fb9880e5-6155-b080-837a-948d5fd08b4c`。每阶段前检查空闲；主测和复测各 10 次
  占用采样只看到本次进程。没有调整时钟或停止他人任务。
- 环境：Python 3.10.20、PyTorch 2.13.0+cu129、vLLM 0.30.0+cu129、CUDA 12.9；
  CPU/内存绑定 NUMA 0。FA3 显式 `fa_version=3`、`num_splits=0`。
- 48 层，4 KV heads、32 Q heads、head_dim 128，BF16，128-token 物理页。
  各层顺序执行，三臂使用同一 KV、码表、Q 和逻辑页顺序。
- KV 来自历史 CPU Transformers capture，模型为
  Qwen3-30B-A3B-Instruct-2507-FP8 权重反量化至 BF16；内容为 OpenHands 轨迹。
  KV 不是 FP8，不是 AgentX，也不是 GPU serving capture。
- 原始 dump 为 uint16 `[32,48,2,128,4,128]`，SHA256
  `1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322`。
  页 `0:8` 校准，评估使用从页 8 开始的 4/16 个真实块，不复制页扩充上下文。
- Query 使用种子 314 生成各层独立随机 BF16 Q。C 为总 KV 长度，包含右对齐 causal 的
  Q-token 后缀；所有 KV 已准备好。未计 QKV 投影、新 KV 写入/编码、MLP 或层间依赖。
  **这是 48 层只读 attention 回放，不是完整模型 forward 或 TTFT。**
- 六组 `C={512,2048} × Q={16,128,512}`；每模式每臂预热 3 次、正式 30 次。
  顺序覆盖全部六种排列；第二个独立进程反转顺序、重复相同协议。未筛除样本。
- 驻留组同时测 eager 完整 host wall 和捕获全部 48 层的 CUDA Graph GPU 时间。
  外部恢复组从同一份 pinned-host FPKB 起步，逐臂实际 H2D 到独立接收区，恢复、
  status D2H、等待并确认成功后才执行 attention；同时保留 graph/eager 两种读取方式。
  不含归档生成、编码、D2H 保存、RPC、网络或 SSD。主计时与分阶段诊断分开。

Fused 自动 splits：C512 的 Q16/128/512 为 `4/1/1`，C2048 为 `14/1/1`。
没有用未优化的 unsplit 短 query 代替当前实现。

## 数据已在 GPU 上

主测中位数，单位 **ms**，为一个完整 48 层 CUDA Graph 的 GPU 时间。
Native 的一次恢复不在此窗口；Gather 每次读取的解压在窗口内。

| C / Q | Native FA3 | Gather＋FA3 | Fused prefill | Native 相对 Gather 降低 |
| --- | ---: | ---: | ---: | ---: |
| 512 / 16 | 0.732 | 0.963 | 0.991 | 24.01% |
| 512 / 128 | 0.789 | 1.029 | 1.784 | 23.26% |
| 512 / 512 | 0.933 | 1.154 | 4.004 | 19.13% |
| 2048 / 16 | 0.826 | 1.153 | 1.414 | 28.41% |
| 2048 / 128 | 1.273 | 1.609 | 5.133 | 20.85% |
| 2048 / 512 | 2.788 | 3.158 | 14.109 | 11.71% |

逆序复测中 Native 相对 Gather 降低 11.87%–27.70%，六组排序不变。
这说明原生页就绪后省去了 gather 的重复工作，不能解释为外部恢复本身更快。

## 相同外部数据：恢复完成后再读取

主测中位数，单位 **ms**，为 H2D＋恢复＋状态检查＋48 层 graph attention 的完整
host wall。三臂传输字节相同：C512 为 35,471,360 B，C2048 为 141,905,920 B，
另有 40/136 B offsets；恢复状态检查在 attention 前完成。

| C / Q | Native FA3 | Gather＋FA3 | Fused prefill | Native 相对 Gather 降低 |
| --- | ---: | ---: | ---: | ---: |
| 512 / 16 | 1.543 | 1.685 | 1.729 | 8.40% |
| 512 / 128 | 1.613 | 1.770 | 2.515 | 8.86% |
| 512 / 512 | 1.757 | 1.876 | 4.745 | 6.30% |
| 2048 / 16 | 3.887 | 3.939 | 4.192 | 1.33% |
| 2048 / 128 | 4.330 | 4.383 | 7.902 | 1.21% |
| 2048 / 512 | 5.851 | 5.932 | 16.889 | 1.36% |

逆序复测降低 1.24%–8.69%，六组排序不变。较大上下文中的收益很小，
本次不支持“独立 codec 的召回显著加速”或“召回一定更慢”的概括。

C2048/Q128 的独立阶段诊断，中位数 **μs**：

| 阶段 | Native | Gather |
| --- | ---: | ---: |
| H2D payload＋offsets | 2577.5 | 2576.4 |
| 解码恢复 / 导入压缩记录 | 454.0 | 176.3 |
| 后续 48 层读取 | 1275.4 | 1610.2 |

Native 提前做解压，恢复阶段比旧 import 多约 278 μs；后续读取少约 335 μs，
大部分互相抵消。上表来自三个独立诊断样本，不能相加替代主测完整中位数。
旧 import 主要校验 header、几何、码表身份和长度；新 decode 还检查目标映射与
记录内部 payload。这里比较的是两个现有公共接口的完整成本，不是纯解压算法消融。

## Eager 的排序为何不同

主测完整 host wall，**ms**；每格按 Native / Gather / Fused 排列：

| C / Q | 驻留 eager | 外部恢复＋eager |
| --- | --- | --- |
| 512 / 16 | 1.339 / 2.095 / 1.138 | 2.177 / 2.841 / 1.890 |
| 512 / 128 | 1.342 / 2.091 / 1.855 | 2.188 / 2.853 / 2.599 |
| 512 / 512 | 1.359 / 2.107 / 4.102 | 2.196 / 2.849 / 4.850 |
| 2048 / 16 | 1.725 / 2.516 / 1.549 | 4.802 / 5.275 / 4.342 |
| 2048 / 128 | 1.726 / 2.481 / 5.217 | 4.825 / 5.291 / 7.998 |
| 2048 / 512 | 2.940 / 3.377 / 14.195 | 6.010 / 6.158 / 16.982 |

短 Q 的 eager Fused 可以更快，但 Graph 中排序反转。不同 Python/API 提交路径会
影响 eager 完整耗时；不能据此说 Fused 的 GPU 计算优于 FA3。没有单独做 CPU profiler，
不能把所有差额精确归因于某个调用，也不能把局部 Graph 结果当成完整模型 Graph 支持。

## HBM 容量与工作区

有效 C2048、48 层 KV：原生 BF16 为 **201.327 MB**，压缩 arena 预算为
**146.276 MB**。实际测试各加两个未使用的保护页，对应数组分配为 226.492 / 164.561 MB。

| 额外分配 | 本次 C2048 |
| --- | ---: |
| 旧 store staging / scratch / overflow | 12.583 / 1.049 / 0.033 MB |
| 旧 store metadata＋码表 | 0.230 MB |
| Gather BF16 workspace，跨层复用 | 4.194 MB |
| 当前 NativeCodec 全部自有缓冲 | 402.900 MB |
| 每臂压缩接收 payload | 141.906 MB |

NativeCodec 仍分配 201.327 MB 编码 record 槽和 201.392 MB 输出 blob；虽然恢复后读取
不用它们，本次没有释放，不能从预算中隐去。共享 pinned payload 另占 141.906 MB；
Q128 的查询和每臂输出各占 50.332 MB，均不是 KV。三臂共存的 allocator 数值不是
任一方案独立部署的峰值。

因此独立 codec 的主要价值是保留原生计算路径与缩小接入边界，而不是 GPU 驻留容量。
如果继续这条路线，编码/解码工作区拆分与按需池化值得优先处理。冷 KV 是否卸载、
何时卸载仍由引擎和外部缓存策略决定，本轮没有改变该策略。

## 验证与证据

- 主测和逆序复测六组配置全部完成。全部选中 KV 与原始 capture 逐位一致；
  未选原生页及旧 arena 保护页未被修改；每次外部恢复都检查 status。
- Native 和 Gather 的 FA3 输出逐位一致；Fused 通过 `atol=rtol=0.02`，
  最大绝对误差 0.03125，不是逐位一致。KV 无损不意味着不同 attention kernel
  的浮点归约结果逐位一致；本轮没有做模型精度评估。
- 两次运行的源码、输入和查询哈希一致。benchmark SHA256：
  `522bf98cb96b64b78307df02d6a080efbbbd0ba487accde436dced547075a525`。
- 仓库外证据目录 `20261031-native-vs-gather-fa3` 保留 `primary.json`、`repeat.json`、
  功能 smoke、逐样本数据、GPU 占用采样、执行命令、日志与源码。smoke 不混入主表。
- 未执行模型服务、真实新 suffix 生命周期、网络/SSD、并发容量压力或端到端吞吐。

复现时先确认 GPU 独占，使用相同依赖环境并按设备拓扑绑定 NUMA：

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" TORCH_EXTENSIONS_DIR=/path/to/data/jit-cache \
  numactl --cpunodebind=0 --membind=0 \
  python -m benchmarks.native_codec_prefill_bench \
  --dump /path/to/captured-kv.npy --out /path/to/data/comparison.json \
  --contexts 512,2048 --queries 16,128,512 --rounds 30 --warmup 3 --cold-eager
```

逆序复测增加 `--order reverse`，用 `--input-description` 记录真实数据来源。
