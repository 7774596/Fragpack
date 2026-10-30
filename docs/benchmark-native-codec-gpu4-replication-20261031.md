# GPU4 四路径固定预算复测（2026-10-31）

这是[GPU2 RawDirect 与恢复粒度实验](benchmark-native-codec-raw-direct-20261031.md)的独立硬件复测。GPU4 与 GPU2 的样本分别保存；本文所有表格只含 GPU4。

## 配置与计时

- GPU4：NVIDIA H800，UUID `GPU-910de65d-2e94-591b-557e-4b29047bf2fc`，CPU/内存绑定 NUMA1；PyTorch 2.13.0+cu129、vLLM 0.30.0+cu129。每进程启动前检查至少 32 GB 主机内存余量，运行期间检查 GPU 独占。
- 48 层 BF16 KV，GQA 32:4，D=128，P=128；历史 capture 的 0:8 校准，8:32 重复/轮换构造不可变历史。长上下文是扩展后的合成负载，未采集新的 32K KV 分布。
- RawDirect 从 pinned 原生布局直接复制至原生页；Native 从压缩归档恢复原生页；二者均调用 stock FA3。Gather 与 Fused 保持 GPU KV 压缩：前者逐层 gather 后调用 FA3，后者直接融合读取。Q=1 的 Fused 使用实际 decode kernel。
- 压缩路径使用单 receive 与 graph restore/request wait，Native 使用 decode workspace；RawDirect 使用公开连续复制和一次请求完成等待。离线编码与 Raw 布局转换不计时。每条 trace 从空逻辑 LRU 开始，计入策略、实际 pinned H2D、恢复/完成等待和 48 层读取；无新 KV 写入或服务端模型计算。
- 两个独立进程：F 按 A→D、RawDirect→Native→Gather→Fused；R 同时逆转配置和路径。各路径/配置预热 1 条、计时 12 条 trace，进程间比较不视作配对轮次或独立服务样本。

| 配置 | 上下文×会话 | Q | 预算 GB | 恢复块数 K | 每 trace 请求数 |
|---|---:|---:|---:|---:|---:|
| A | 1024×5 | 1 | 0.45 | 2 | 24 |
| B | 1024×5 | 1 | 0.45 | 4 | 24 |
| C | 32768×3 | 1 | 8 | 4 | 20 |
| D | 32768×3 | 128 | 12 | 4 | 20 |

## 完整耗时与容量

单位为每 trace 主机 wall time 的中位数，ms；F/R 均保留。正的 Native 降幅表示比同进程 RawDirect 更快。

| 配置/进程 | RawDirect | Native | Gather+FA3 | Fused | Native 较 RawDirect 降幅 |
|---|---:|---:|---:|---:|---:|
| A/F | 37.714 | 31.264 | 27.154 | 19.596 | +17.1% |
| A/R | 37.844 | 31.480 | 27.422 | 19.578 | +16.8% |
| B/F | 37.886 | 42.699 | 26.771 | 19.301 | -12.7% |
| B/R | 37.794 | 42.641 | 27.160 | 19.149 | -12.8% |
| C/F | 619.066 | 495.035 | 215.872 | 166.604 | +20.0% |
| C/R | 632.300 | 493.080 | 216.225 | 166.733 | +22.0% |
| D/F | 364.312 | 327.159 | 377.707 | 877.958 | +10.2% |
| D/R | 365.421 | 327.048 | 376.263 | 880.442 | +10.5% |

容量与 miss 数来自实际分配峰值和完整 trace。每格为“blocks / miss”；审计核对 F/R 一致。

| 配置 | RawDirect | Native | Gather+FA3 | Fused |
|---|---:|---:|---:|---:|
| A | 35 / 11 | 33 / 11 | 44 / 5 | 44 / 5 |
| B | 35 / 11 | 31 / 19 | 41 / 5 | 41 / 5 |
| C | 635 / 10 | 627 / 10 | 860 / 3 | 864 / 3 |
| D | 945 / 3 | 939 / 3 | 1282 / 3 | 1293 / 3 |

Native 与 RawDirect 的容量差包含 receive、codec、metadata 和 graph allocations；Gather/Fused 的容量还包含压缩 arena、overflow、staging 和各自 attention workspace。实际构造、最大请求 cold restore/read 与计时峰值均纳入绝对 tensor 预算，并验证下一个 block 的不可行边界；allocator reserved 与 CUDA context 不在该 tensor 上限内。不能用逻辑字节线性推算跨 allocator 台阶的容量。

## 观察

- **恢复缓冲会改变容量收益。** A/B 中 Native 的 K=2 保留 33 blocks、4 个完整会话，K=4 保留 31 blocks、3 个会话；miss 从 19 降至 11，trace 耗时降低 26.2%–26.8%。K=2 比 RawDirect 快 16.8%–17.1%，K=4 则两轮都更慢。
- **GPU 常驻压缩在容量不足时仍有明显优势。** A/B 的 Gather 与 Fused 可完整保留 5 个会话，仅首次 5 次 miss；Native K=2 仍只有 4 个会话容量。C 的 Native 只能保留 2 个 32K 会话，Gather/Fused 可保留全部 3 个，miss 为 10 对 3，累计压缩 H2D 为 22.704 对 6.811 GB。C 的耗时差同时包含历史保留和读取执行成本，不能当作单次解码速度差。
- **容量足够、Q=128 时 Native 的读取路径更有利。** D 的四条路径都只有首次 3 次 miss；Native 较 RawDirect 降低 10.2%–10.5%，较 Gather 降低 13.1%–13.4%，较当前 Fused 实现降低 62.7%–62.9%。Q=1/C 与 Q=128/D 同时改变预算和驻留状态，二者不是单变量 Q 消融。

这些数据描述当前 trace、预算和设备上的路径差异。GPU 独占不隔离共享 CPU、内存或传输资源；F/R 耗时变化未做因果归因，也不据此推导硬件常数吞吐或默认最优配置。

## 验证与证据

本组 32/32 个路径/配置/进程项通过；共 384 条计时 trace、8448 次请求。全部成功项逐块 poison 后检查 48 层 KV 位；RawDirect/Native/Gather attention 与 stock FA3 位一致，Fused 通过元素级 `atol=rtol=0.02`，记录最大绝对误差 0.03125。该误差不要求单独小于 atol，因为逐元素门限同时包含 rtol。

每条路径保留改变 payload/map 的恢复检查、未用页 canary、错误阻止 attention 的检查；RawDirect 检查 CPU 非法映射无写入，压缩路径检查错误 status 在后续成功 chunk 后仍保留。全部复测输入、码表、Q 与原始 FA3 reference 的哈希逐进程一致；独立审计重建 LRU、物理页和传输字节，核对 source 前后不变、NUMA/GPU/主机内存门禁、fit/计时峰值及释放后 allocated=0。

完整逐 trace 标量在 [CSV](data/native-codec-gpu4-replication-20261031.csv)，保留总 wall、恢复/读取 wall 分区、miss/eviction、传输字节、copy 提交、status、容量、组件字节及峰值。分区是主机 wall，不能当作 GPU 独立阶段归因。

复现代码对应 `74ea52e` 中 RawDirect 功能；实际运行保留冻结候选的 HEAD 和源码哈希。没有全模型、在线调度、新 KV 编码/保存、SSD/网络、AgentX 或 serving TTFT/TPOT 结论。

| 文件 | SHA256 |
|---|---|
| `raw-direct-gpu4-forward-r1.json` | `085f218357715a9289e10094a83163f49c64625a4123336356eddd63b23d2c25` |
| `raw-direct-gpu4-reverse-r1.json` | `e4e63d8f8ff8250b5ba18f880efa67c9fb5359a1a72235c27704b4f794ef32bc` |
| `gpu4-replication-independent-audit.json` | `52d129d0a056d2342950f52a4e483d85982317fee666a79c62478d01d2df2fb2` |
| `analyze_gpu4_replication.py` | `f981a578561acfd194982ca31698267fc60a79dac760e146ac62c6645ff6acfa` |
| `benchmarks/native_codec_budget_bench.py` | `ace561513f5610b57cd9dcbbca482d70f6023b05ae311b88cbda052c3d659370` |
| `benchmarks/native_codec_raw_direct.py` | `ccb9f8adc5e2eac816b39e3fa6e4920812006a0c7d2d13f9d7458622e2a5b135` |
| `fragpack/ops/native_codec.cu` | `e024cbfe6b8a9e280c24a153f75b11b76d12df0e4610c25fcd4cd9ea18980175` |

CSV SHA256：`8346106c740a80b194e74f2a9a7a2926cb024dee80d0c4da1d90efc4161c578a`。所有 GPU2 历史数据与报告保持原样。
