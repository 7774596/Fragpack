# 32K 多会话固定预算回放（2026-10-31）

**8 GB 下，压缩 HBM 能保留三个会话，Raw/Native 只能保留两个；12 GB 下四条路径都能保留三个。** Q=1 时 Fused 最快；Q=128 时，8 GB 下 Gather 最快，12 GB 下 Native 最快。融合读取并非所有 query 长度都占优。

本轮全部八个独立进程完成：32 个配置、96 条计时 trace、1,920 次请求。完整正反序结果与慢例均保留，未筛选或合并成一个总体加速比。

## 配置与计时范围

单卡 H800 PCIe（GPU 2，UUID `GPU-fb9880e5-6155-b080-837a-948d5fd08b4c`）；PyTorch 2.13.0+cu129、vLLM 0.30.0+cu129。48 层、BF16 KV、GQA 32:4、head dimension 128、原生页和压缩逻辑块均为 128 tokens。三个会话各含 32,768 tokens；每次读取的 query token 数 Q∈{1,128}，GPU tensor 预算∈{8,12} GB。每批恢复 4 blocks，工作区与 receive 按该上限预分配。

每个进程依次测四条路径，每条先暖机一条 trace，再记录三条。每条 trace 重置逻辑 LRU，串行执行同一序列：

```text
0,0,1,1,2,2, 0,1,2,0,1,2, 0,0,0,2,2,2,0,2
```

前六次含首次载入与立即复用，中间六次扫描工作集，最后八次集中复用。会话和 KV 不增长；这里的集中复用不代表并发请求。输入是同一历史 Qwen3-30B-A3B/OpenHands CPU BF16 捕获：0:8 校准、8:32 评估池，经旋转/重复扩到 32K，**不是新的真实 32K 模型捕获**。本组输入未触发 raw fallback。

| 路径 | 外部归档与 GPU 驻留 | 读取方式 |
|---|---|---|
| Raw | raw 归档 → packed H2D＋已验证 scatter → 原生 BF16 页 | FA3 |
| Native | 压缩归档 → H2D＋独立 codec → 原生 BF16 页 | FA3 |
| Gather | 压缩归档 → H2D＋import → 压缩 HBM | 每层 gather 到 BF16 workspace，再调用 FA3 |
| Fused | 压缩归档 → H2D＋import → 压缩 HBM | Q=1 融合 decode；Q=128 融合 prefill |

Native 使用 coalesced scatter 候选。统一 `graph/request` 设置在长矩阵运行前，按[已完成的配对实验](benchmark-native-codec-paired-dispatch-20261031.md)中 Native 的四个 K=4 结果选定，再用于所有路径。这里分别实测每条路径的容量，不沿用配对实验的 superset 分配。

耗时是整条 20 请求 trace 的 host wall：包括 LRU、元数据、实际 H2D、恢复、状态处理、48 层读取和等待。编码与归档准备在计时外；计时中的 D2H 只返回状态。这不是编解码往返、完整模型推理、TTFT 或 serving 测试。Raw 是统一接口的 packed-transfer＋scatter 对照，尚未测直接向原生页写入 H2D 的优化基线。

## 容量、命中与传输

下表是实测可分配 blocks；每个会话需要 256 blocks，三个共需 768。正反序容量完全一致，所有设置都实际验证了再增加一个 block 会超过预算。

| Q | 预算（GB） | Raw | Native | Gather | Fused |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 627 | 627 | 860 | 864 |
| 1 | 12 | 947 | 947 | 1293 | 1304 |
| 128 | 8 | 619 | 619 | 842 | 853 |
| 128 | 12 | 939 | 939 | 1282 | 1293 |

同一设置下，Raw 与 Native 的容量、逐请求 slots、命中和驱逐完全一致。Native 在外部压缩，恢复后仍以 BF16 驻留，因而此路径不扩大 GPU 可保留的会话数。Gather/Fused 的压缩 HBM 跨过了 8 GB 下的三会话门槛。

以下每条 trace 的行为在 Q=1/128 和正反序中一致；payload 是实际传输字节，不含很小的元数据。

| 预算（GB） | 路径 | 命中 / 20 | Miss | 驱逐 | Payload H2D（GB） | Restore chunks / host waits |
|---:|---|---:|---:|---:|---:|---:|
| 8 | Raw | 10 | 10 | 8 | 32.212 | 640 / 10 |
| 8 | Native | 10 | 10 | 8 | 22.704 | 640 / 10 |
| 8 | Gather | 17 | 3 | 0 | 6.811 | 192 / 3 |
| 8 | Fused | 17 | 3 | 0 | 6.811 | 192 / 3 |
| 12 | Raw | 17 | 3 | 0 | 9.664 | 192 / 3 |
| 12 | Native | 17 | 3 | 0 | 6.811 | 192 / 3 |
| 12 | Gather | 17 | 3 | 0 | 6.811 | 192 / 3 |
| 12 | Fused | 17 | 3 | 0 | 6.811 | 192 / 3 |

压缩归档总量为 raw 的 70.48%。8 GB 下 Native 仍载入十次；Gather/Fused 只载入三次。12 GB 下四条路径均只在首次访问时载入三个会话，额外 HBM 容量在这条 trace 中没有进一步减少 miss。

## 全部耗时

单位 ms，数值为同一进程三条 trace 的中位数。F 的路径顺序为 Raw→Native→Gather→Fused；R 完全反向。运行顺序对每个 Q 均为 8GB/F、12GB/F、12GB/R、8GB/R。

| Q | 预算（GB） | 顺序 | Raw | Native | Gather | Fused |
|---:|---:|:---:|---:|---:|---:|---:|
| 1 | 8 | F | 739.821 | 571.807 | 284.948 | 195.294 |
| 1 | 12 | F | 254.052 | 217.777 | 279.151 | 194.035 |
| 1 | 12 | R | 275.418 | 200.897 | 264.682 | 194.202 |
| 1 | 8 | R | 810.719 | 562.250 | 268.979 | 208.397 |
| 128 | 8 | F | 950.446 | 829.146 | 472.438 | 1557.373 |
| 128 | 12 | F | 466.208 | 403.548 | 467.529 | 1544.163 |
| 128 | 12 | R | 464.433 | 403.985 | 455.614 | 1550.434 |
| 128 | 8 | R | 1050.719 | 758.313 | 452.304 | 1548.391 |

- **8 GB 容量受压：** Q=1，Fused 为 195–208 ms，Native 为 562–572 ms；Q=128，Gather 为 452–472 ms，Native 为 758–829 ms。压缩驻留减少了反复载入。
- **12 GB 三会话均可保留：** Native 在两个 Q 下都快于 Gather；Q=128 约 404 ms，对应 Gather 456–468 ms。Q=1 仍是 Fused 最快，约 194 ms。
- **保留慢例：** Q=128 的 Fused 为 1,544–1,557 ms；即使 miss 已降到三次，融合读取仍明显慢于 FA3 路径。
- Native 在八个进程中均快于 Raw，中位耗时降幅为 12.8%–30.6%；正反序幅度存在明显差异。两次有序进程与进程内三条 trace 不构成独立同分布样本，不给出置信区间或稳定收益承诺。

下面仅用 12 GB、Q=128、F 的实际中位 trace 拆分墙钟区间。各行分项来自同一条 trace，可以相加；这些仍包含 host 提交和等待，不是独立 GPU kernel 时间。

| 路径 | 策略＋恢复（ms） | 元数据＋读取（ms） | Step 外循环（ms） | 合计（ms） |
|---|---:|---:|---:|---:|
| Raw | 215.432 | 250.709 | 0.067 | 466.208 |
| Native | 150.941 | 252.536 | 0.070 | 403.548 |
| Gather | 162.944 | 304.514 | 0.072 | 467.529 |
| Fused | 148.916 | 1395.173 | 0.074 | 1544.163 |

在同为三次载入的对照里，Fused 的额外耗时主要记录在读取墙钟区间。下一步若优化 Q=128 的融合路径，应先对这一读取阶段做 GPU profiling；本轮不能把差值进一步归因到具体 kernel。

## 内存与验证边界

预算覆盖评估器的 KV 池、receive、codec/store 辅助空间、48 层 Q/output 和 graph 分配。以下取 8 GB、Q=128、F；KV 区是整个池的逻辑分配，峰值是计时 trace 中实际 PyTorch tensor 分配。allocator 舍入使分项之和与实测峰值不同。

| 路径 | KV/arena 分配（GB） | Receive（MB） | Codec 辅助（MB） | Store 辅助（MB） | Gather workspace（MB） | 实测峰值（GB） |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 7.789 | 50.332 | 0.000016 | 0.000 | 0.000 | 7.940 |
| Native | 7.789 | 50.348 | 0.021520 | 0.000 | 0.000 | 7.940 |
| Gather | 7.698 | 50.348 | 0.000000 | 17.873 | 67.109 | 7.934 |
| Fused | 7.798 | 50.348 | 0.000000 | 17.926 | 0.000 | 7.967 |

Q=128 时所有路径另有相同的 Q＋output 100.663 MB；Fused 在这个设置中的额外 fused workspace 为 0。Native 的 `decode` workspace 不分配编码 scratch 或输出 blob。表中不含模型权重、CUDA context、allocator reserved 或独立正确性验证临时峰值，不能当作完整部署峰值。

实验同时保留 raw 归档 9.664 GB 和压缩归档 6.811 GB，共 16.475 GB；这是对照与验证的 host 数据，不是单一路径部署需求。每个新进程启动前均通过 32 GB 的全局/cgroup 可用内存 gate，最小记录余量为 566.573 GB。该 gate 没有测量进程 host 峰值，也不单独验证 NUMA 节点余量。

全部 32 个配置完成 poison 后的 48 层 KV 位值检查；Raw/Native/Gather 的 attention 输出与源数据 FA3 逐位一致。Fused 的最大绝对误差为 Q=1 的 0.0078125、Q=128 的 0.03125，通过既定 `atol=rtol=0.02`，但输出不逐位相同。每个配置另通过两种 payload/map、inactive canary 及 `[4,4]` 同 graph/status 连续复用的早期错误保留探针。

所有实际 fit 和计时峰值符合预算；每个路径释放后 allocated 为 0。八个进程的 source 前后快照一致，独占 GPU 监控均通过；NUMA 绑定不等于独占 CPU 或 PCIe。没有丢弃某次反序或较慢 trace。

[完整 CSV](data/native-codec-long-context-20261031.csv) 保存全部 96 条 trace 的原始 μs 时间、同 trace 墙钟分段、命中/字节/等待计数、容量、峰值和来源哈希；原始 JSON 保留 1,920 条请求明细。

## 来源

实验从隔离候选 checkout 执行，记录 HEAD 为 `7b552c56666c087514bd5944fdba176aac839ca6`；下列文件哈希标识实际候选内容，不能只用该 HEAD 复现。

| 标识 | SHA256 |
|---|---|
| 源文件聚合 digest | `83ed172c9d6e8abaf202cb3fb7d2bd0cde124e2df57f5b4808d768b3c3d49b6e` |
| `native_codec.cu` | `e024cbfe6b8a9e280c24a153f75b11b76d12df0e4610c25fcd4cd9ea18980175` |
| `native_codec_budget_bench.py` | `39d56438794b8d5e27446014431703b8687c030d9b9832c9c35fef1c02bcc701` |
| 输入捕获 | `1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322` |
| 评估池 | `90d340536c304f6a0395b10f5c0abe9f73204ac2095b140b1c725485c175cb51` |
| 码表 | `8cd2b8979b12101119104f1ed92652ce938752e49a4e2d1f5c69de47ff9d0a30` |
| 统一设置选择记录 | `b72744a37c143dcbee3c512d12952af4a8870b608beee2068f4894c223143b1f` |
| 最终 protocol | `f14983ade53e441ab77e48d8148c2655bb4bb740496a5d6c606df6ca951d3b92` |
| 执行 manifest | `87181830c5e67b503559b78479d3a4622e5553206c95dd73c2b86c932e61a916` |

| 原始结果文件 | SHA256 |
|---|---|
| `long-c32768-w3-b8000000000-q1-k4-forward.json` | `0ba40e39260fa6f6b0d07066c96ee9f9007facaed0dfc03fcbe21c32274a5054` |
| `long-c32768-w3-b12000000000-q1-k4-forward.json` | `efa9518bc141b92b5763ee10a8d8b5ba1683f00318961e0bc68131bafccd36b2` |
| `long-c32768-w3-b12000000000-q1-k4-reverse.json` | `bd82694e29262240fe09557add940b167a8b41cf448b4cfca0686679ea2eed76` |
| `long-c32768-w3-b8000000000-q1-k4-reverse.json` | `ea8751899e54ca989c880e491b95295796140a598ce42bc500335d7bbeda691d` |
| `long-c32768-w3-b8000000000-q128-k4-forward.json` | `76bd4c79b6f1be5c31ef36635a84d9f9f1326d86fc68cd174c28102e7390aa2a` |
| `long-c32768-w3-b12000000000-q128-k4-forward.json` | `06c5475a81959cc0aaeba943e643c8a89707a35d2aac1ca8d40ce75f42d6b13a` |
| `long-c32768-w3-b12000000000-q128-k4-reverse.json` | `56867a4b88ba64a15d27b80037ed00509601e44fcd14de5c0130886d21647cd3` |
| `long-c32768-w3-b8000000000-q128-k4-reverse.json` | `5c5d2faa1d5e3aa54f9bdbdf9f77aea3415af76106d409ae4a9a3625830adfcf` |
