# 独立 codec 优化总览（2026-10-31）

初始的大显存差距主要来自过量预分配的编码/接收工作区；本轮通过角色拆分和有界复用
缩小了这部分开销，并降低了原生页恢复成本。产品边界保持为：
**外部归档可以压缩，恢复后的活跃 KV 仍以原生 BF16 驻留，继续使用 stock attention。**
它不具备压缩 HBM 路径的驻留容量收益；这里也尚未验证完整模型服务接入。

下表仅汇总 **GPU2（H800 PCIe）** 的分阶段实测。各行使用独立对照，不能把速度比相乘。
完整配置、正反序结果、负收益、校验和来源哈希均保留在链接报告中。

| 阶段 | 已验证的结论与边界 | 报告 |
|---|---|---|
| 工作区角色拆分与有界复用 | 固定 32K KV、Q=128，reader 张量峰值 **13.006 → 6.561 → 3.387 GB**：先移除编码工作区，再把 K=256 缩至 4。与同 K 的 packed Raw 仅差 37,888 B。另行分配的 K=4 writer pool 为 **0.101 GB**。 | [内存与 pool](optimization-native-codec-memory-20261031.md) |
| 先测别名检查 | 基线 N=256 时，P=128 的重复页检查增量约 7 μs，P=16 约 58 μs。相对完整恢复的理想化可消除量分别不足 0.1% / 0.8%；这是诊断估算，不是删去校验后的实测加速。 | [阶段 profiling](profiling-native-codec-stages-20261031.md) |
| 合并解码写回 | 含校验的 **GPU 常驻恢复为 1.707–1.979×**；另测真实 H2D、策略与 48 层读取后，完整 trace wall 仅下降 **5.48%–8.15%**。容量、命中和传输量未变。每 CTA 共享内存增加 8,176 B，没有新增全局 BF16 staging。 | [GPU 恢复](optimization-native-codec-coalesced-20261031.md) · [完整回放](benchmark-native-codec-coalesced-budget-20261031.md) |
| Graph 提交与每请求检查 | 同一分配内，Native 的配对速度比为 K=1 **1.097–1.100×**、K=4 **1.022–1.027×**；仍保留全部 status 检查。该实现没有传输/解码重叠，Graph 使用固定地址与已核验归档的完整 receive 容量视图。 | [配对调度](benchmark-native-codec-paired-dispatch-20261031.md) |
| 32K 会话、固定总预算 | 三会话工作集在 8 GB 下，Raw/Native 只能保留两会话，压缩 HBM 可保留三会话，miss 为 **10 对 3**；12 GB 下都能保留三个。Q=128、12 GB 时 Native 约 404 ms，Gather 456–468 ms；Fused 的 Q=128 慢例仍约 1.54 s。 | [长上下文回放](benchmark-native-codec-long-context-20261031.md) |
| 双槽流水化 | 长上下文 Q=1，同一双槽分配内 Native 为 **1.111–1.117×**；但 0.5 GB 短会话场景从四会话降至三个，实际总耗时反增 **33.3%–33.8%**。Q=128 未显示部署净收益，不能设为默认。 | [收益、容量代价与 DMA 时间线](benchmark-native-codec-copy-pipeline-20261031.md) |
| 原生 Raw Direct 基线 | 去掉 receive/codec 提高容量，但不保证更快。0.45 GB、1K 短会话、K=4 时，Native 为 **47.358 / 48.602 ms**，Raw Direct 为 **61.295 / 44.523 ms**（F/R），优势方向翻转。 | [直传基线](benchmark-native-codec-raw-direct-20261031.md) |
| 恢复粒度：后续独立对照 | 0.45 GB 下 K=1/2/4 的 Native 容量为 **34/33/31 blocks**；K=2 恢复第四个可保留会话，比 K=4 少 **25.5%–25.9%** wall，比同期 Raw Direct 少 **16.3%–16.8%**。0.5 GB 下为 37/37/35 blocks，miss 都为 11；K=2/K=4 快慢随顺序翻转，没有统一最优 K。 | [K=1/2/4 完整结果](benchmark-native-codec-raw-direct-20261031.md#后续迭代减小-native-receive-粒度) |

**GPU4（H800）另行完成 [32 项独立复测](benchmark-native-codec-gpu4-replication-20261031.md)，
共 384 条 trace。** 0.45 GB 下 Native K=2 保留四会话、K=4 仅三个，耗时降低
26.2%–26.8%；32K/8 GB/Q1 时，压缩 HBM 保留三会话、Native 两个，miss 为 3 对 10。
32K/12 GB/Q128 时各路径均仅三次 miss：Native 为 327.0–327.2 ms，Raw Direct 为
364.3–365.4 ms，Gather 为 376.3–377.7 ms，Fused 为 878.0–880.4 ms。两张 GPU 的样本
分开保存，不汇成一个速度比；Q 与预算同时变化，也不是单变量 Q 消融。Raw Direct、K
及 GPU4 的复现设置见[可移植配置包与命令](../benchmarks/cases/native_codec_20261031/README.md)。

内存表测的是 Torch GPU 张量峰值，包含 KV、接收区、codec、Graph，以及回放保留的
48 层 Q/output；后者是测试 fixture，不是实际服务的常驻分配模型。模型权重、驱动和
CUDA context 不在其中。reader 的剩余大头是 **3.221 GB 原生 BF16 KV**。reader 使用
一份串行复用工作区；公共 pool 的跨流行为另经 GPU 测试。0.101 GB writer pool 与 reader
分别构造、测量和释放，不能据此声称已经验证读写并存峰值或编码吞吐。
在这组输入上，剩余的原生 BF16 与压缩 HBM 容量差主要来自 **KV 表示选择**，不是目标页
别名检查；后者负责映射合法性，其诊断成本已在表中单独列出。

性能矩阵使用历史 Qwen3-30B-A3B/OpenHands **BF16 KV** 捕获，经重复构造较长的不可变
会话，执行 48 层只读 attention。它不是新的真实 32K 模型执行，也没有生成增长、完整
模型计算、并发 serving 或 TTFT 测量。固定预算回放计入各路径实际发生的 pinned H2D、
状态处理、策略与读取等待，但编码/归档准备在计时外。早期 Raw 是 packed H2D＋scatter；
[初始容量矩阵](benchmark-native-codec-fixed-budget-20261031.md)及其
[顺序补充对照](benchmark-native-codec-budget-controls-20261031.md)保留原始波动，不能与后续源码混作配对结果。

Raw Direct 的独立功能/profile 检查确认：三种连续/碎片映射均逐位恢复，没有新增
GPU 张量暂存，也没有恢复区间内的 kernel 或 D2D；该诊断不用于性能计时。上述 K
迭代与主基线分别报告，保留方向翻转，不能用后续较快结果替换早期慢例。

下一步应按实际预算和整会话容量门槛选择 K、槽数，再在真实 connector 路径衡量
编码、迁出和召回的净成本。独立 Nsight 诊断中，双槽虽覆盖约 87.5% 的 decode 时间，
全部恢复 kernel 仅覆盖约 9% 的 payload DMA 时间；更多槽还可能挤掉可复用会话。
优先保留原生 Raw Direct 对照与完整内存核算，不默认启用双槽。
