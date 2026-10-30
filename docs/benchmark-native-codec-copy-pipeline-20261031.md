# 双槽 H2D / restore 流水化实测（2026-10-31）

**长上下文 Q=1 有配对收益，但本轮不支持把双槽设为默认。** Native 在同一双槽分配内的加速比为 1.111–1.117×；0.5 GB 短会话场景却因第二槽挤掉一个会话，部署耗时增加约 33%。Q=128 的配对结果波动较大，也未体现部署层面的净收益。

本轮四个主进程全部完成：48 个路径/配置/进程组合、2,304 条计时 trace，共 49,152 次请求。正反序及慢例全部保留。

## 实验与控制

H800 PCIe，GPU UUID `GPU-fb9880e5-6155-b080-837a-948d5fd08b4c`；PyTorch 2.13.0+cu129、vLLM 0.30.0+cu129。48 层、BF16 KV、GQA 32:4、D=128、128 tokens/block、每批 K=4。输入沿用历史 CPU 捕获，0:8 校准、8:32 评估池，经旋转/重复构造不可变会话；不是新的真实 32K 捕获。本组无 raw fallback。

| 场景 | 会话与预算 | 每条 trace |
|---|---|---|
| A：短会话 | 5×1,024 tokens，0.5 GB，Q=1 | 24 次；四会话局部复用＋第五会话干扰 |
| B：长上下文 decode | 3×32,768 tokens，8 GB，Q=1 | 20 次；首次载入、工作集扫描及集中复用 |
| C：长上下文 prefill read | 3×32,768 tokens，12 GB，Q=128 | 与 B 相同的会话序列 |

Raw 采用 packed H2D＋validated scatter；Native 采用压缩 H2D＋独立 codec，二者恢复到原生 BF16 页并调用 FA3。Gather/Fused 保持压缩 HBM：前者每层 gather 后调用 FA3，后者 Q=1 用融合 decode、Q=128 用融合 prefill。本轮没有改变这些 attention 路径。压缩归档大小约为 Raw 的 70.48%。

[原型](design/native-codec-copy-pipeline.md)增加一个有界 receive/map/offset 槽；恢复与状态 D2H 仍在一个有序 stream 上。第二 copy stream 通过每槽 ready/done 事件建立依赖，完整请求只等待一次并检查全部状态，然后允许 attention。

- **O / original：** 直接调用原单槽提交路径，但保留第二槽分配；用于排除额外事件包装的净收益对照。
- **S / serial-double：** 交替两个槽，copy 与 restore 同 stream；包含双槽协调成本。
- **P / overlap：** 相同两个槽和事件，copy 使用独立 stream。

O/S/P 在每个路径内共用相同地址、容量、live bytes、归档和请求轨迹。每进程 12 个轮转并镜像的 A/B/C/C/B/A round，每 round 每模式两条 trace。另有真正只分配一个槽的独立基线，每进程 24 条 trace。进程顺序为单槽 F→双槽 F→双槽 R→单槽 R；R 反转路径与初始模式顺序，各场景顺序保留。每条 trace 清空逻辑 LRU，KV 内容不增长。

计时为整个串行 trace 的 host wall，覆盖策略、实际 pinned H2D、恢复、状态处理、48 层读取及等待。编码和归档准备在计时外；D2H 只传状态。没有 outbound KV 编码/写回、网络/SSD、完整模型层或 serving/TTFT 测量。Raw 是 packed 基线，不代表直接向原生页 H2D 的最佳实现。

## 同分配的配对结果：全部四条路径

下表每格为 **F / R** 的 round ratio 中位数。每个 ratio 先用同一 round 的两条 trace 求模式均值，再相除；大于 1 表示分母更快。O/P 是净收益，S/P 分离同等双槽协调下的收益；O/S 小于 1 表示协调增加耗时。它们不是各模式总体中位数相除。

| 场景 | 路径 | O/P | S/P | O/S |
|---|---|---:|---:|---:|
| A | Raw | 1.0165 / 1.0237 | 1.0273 / 1.0267 | 0.9893 / 0.9921 |
| A | Native | 1.0178 / 1.0120 | 1.0334 / 1.0338 | 0.9853 / 0.9840 |
| A | Gather | 1.0042 / 1.0002 | 1.0114 / 1.0093 | 0.9931 / 0.9966 |
| A | Fused | 1.0053 / 1.0080 | 1.0144 / 1.0096 | 0.9907 / 0.9922 |
| B | Raw | 1.0864 / 1.0808 | 1.0866 / 1.0816 | 0.9998 / 1.0005 |
| B | Native | 1.1171 / 1.1112 | 1.1204 / 1.1125 | 0.9970 / 0.9993 |
| B | Gather | 1.0561 / 1.0555 | 1.0586 / 1.0591 | 0.9976 / 0.9975 |
| B | Fused | 1.0794 / 1.0756 | 1.0831 / 1.0767 | 0.9966 / 0.9984 |
| C | Raw | 1.0410 / 1.0372 | 1.0399 / 1.0378 | 0.9971 / 0.9973 |
| C | Native | 1.0430 / 1.0072 | 1.0436 / 1.0270 | 0.9967 / 1.0116 |
| C | Gather | 1.0294 / 1.0332 | 1.0274 / 1.0323 | 1.0005 / 1.0007 |
| C | Fused | 1.0081 / 1.0008 | 1.0072 / 1.0117 | 0.9995 / 0.9958 |

Native 的 B 场景为 1.111–1.117×，等价于配对耗时降低约 10.0%–10.5%；并非降低 11.7%。该场景的两次进程中所有 round 均为正收益。A 的净收益仅 1.2%–1.8%；第二槽容量代价尚未计入这些配对数字。

C 的 Native O/P 为 1.0430× / 1.0072×，反序 12 个 round 范围达 0.9329–1.1844×，包含变慢。Fused C 仍约 1.53 s/trace，流水化没有消除其长 query 读取慢例。各 round 的正负结果均保留；不据此给置信区间、稳定加速或默认启用结论。

## 真正部署：容量与净结果

以下是实际 fit 的单槽→双槽 blocks，正反序一致。预算包括 KV、codec/store、receive、元数据、Q/output 和 graph 分配；表中差值不能用第二槽字节数线性推算。

| 场景 | Raw | Native | Gather | Fused |
|---|---:|---:|---:|---:|
| A | 35→31 | 35→31 | 47→41 | 47→41 |
| B | 627→627 | 627→627 | 860→853 | 864→861 |
| C | 939→937 | 939→937 | 1282→1281 | 1293→1288 |

Native 第二槽增加逻辑存储 50,348,088 B；实际分配按 allocator 档位变化。例如 B 的单/双槽都能容纳 627 blocks：实测峰值分别为 7,941,064,704 / 7,991,001,088 B；增加到 628 后分别为 8,003,979,264 / 8,053,915,648 B，均越过 8 GB。这里的相同容量有实际 next-block 测量支持。

下表专列 Native。耗时为独立进程内 trace 中位数，单位 ms，格式 F / R；不是同 round 配对比较。

| 场景 | 单槽→双槽整会话数 | Hit / Miss / Eviction：单槽→双槽 | 单槽 wall | 双槽 P wall |
|---|---:|---|---:|---:|
| A | 4→3 | 13/11/7→5/19/16 | 33.039 / 34.901 | 44.210 / 46.532 |
| B | 2→2 | 10/10/8→10/10/8 | 573.075 / 524.344 | 467.242 / 509.774 |
| C | 3→3 | 17/3/0→17/3/0 | 392.945 / 393.527 | 399.369 / 399.370 |

A 少保留一个会话，miss 从 11 增至 19，压缩 H2D 从 0.780 GB 增至 1.348 GB；双槽总耗时反而增加 33.3%–33.8%。B 的 miss 与 22.704 GB payload 不变，适合继续研究流水化。C 仍可保留三个会话、每 trace 仅三次载入；独立单槽约 393 ms，双槽 P 约 399 ms，没有部署层面的净收益。

跨进程绝对耗时有变化，例如 B 的 Native 单槽 F/R 为 573/524 ms，双槽 O 为 522/564 ms。GPU 独占及 NUMA 绑定不隔离 CPU、host memory 或 PCIe；这里不把漂移归因到某个未测量的资源。同分配调度收益应优先看上述配对比值，部署比较同时看实际容量和命中变化。

## 独立 Nsight 诊断：确认发生了什么重叠

另行捕获 Native、C4096/Q1、K=4、48 层、1 GB 的六段 O/S/P/P/S/O；三个会话按 `0,1,2,0,1,2` 访问，全部 miss。该进程受 profiler 影响，以下只解释 GPU 时间线，不替代主性能结果。

O/S 的 payload H2D 与 restore kernel 交集均为零。P 的两个片段分别在 copy stream 45 与 restore stream 7 上，每段恰有 42/48 次 decode 与下一 chunk H2D 相交，对应六个请求各七次可重叠机会。

| P 片段 | H2D∩decode（ms） | Decode 时间覆盖 | H2D∩全部 restore（ms） | Payload H2D 时间覆盖 |
|---|---:|---:|---:|---:|
| 第一次 | 2.295408 | 87.517% | 3.059231 | 9.030% |
| 第二次 | 2.294540 | 87.496% | 3.023388 | 9.184% |

这里的全部 restore 包含 decode、header/record 校验和 status clear，不含 attention。约 87.5% 的 decode 时间确实与传输重叠，但这些 restore 区间只覆盖约 9% 的 payload DMA 时间，大部分传输仍暴露在关键路径上；这些比例不能直接当作 serving 加速。

六个闭合 NVTX 片段均与结果账本精确对齐：每段 48 次 payload H2D、1,702,813,696 B；108 次 metadata H2D、3,480 B；48 次 status D2H、768 B；1,056 个 kernels。全 trace 的 1,224 个 memcpy 和 6,336 个 kernels 均唯一归属于某个片段。Nsight 同时提示 NVTX/CUDA events 可能未完全收集，并出现一条 “No NVTX events collected” 诊断，原样保留；上述指定事件计数已核对完整，不据此推断所有 CUDA API/event 类别完整。

## 正确性与保留证据

功能 smoke 覆盖 K=1、K=4、K=16、不同上下文尾块和两槽起始相位，14 个 admitted 配置×3 模式共 42 个模式检查通过。K=4 的 `[4,4,4]` 探针实际重用同槽、同 count 的 status 存储，四条路径均保留首个错误、禁止 attention，并核验后续合法恢复与 inactive canary。C4096、0.5 GB 的双槽 Raw/Native 明确保留为 **2 个 infeasible**，未用单槽结果替代。

主结果的 48 个组合全部通过：逐块 poison 后 48 层 KV 位精确比较；Raw/Native/Gather 的 attention 输出与源数据 FA3 逐位一致。Fused 最大绝对误差 0.03125，通过预设 `atol=rtol=0.02`。O/S/P 的地址、live bytes、逐请求 slots、命中、驱逐和传输计数保持一致。每个实际 setup/timed peak 均符合预算，next-block 边界验证通过，释放后 allocated 为零。

四个主进程源文件前后哈希一致、GPU 独占监控通过；启动时均通过 32 GB host/cgroup headroom gate，最小记录余量 565.353 GB。该 gate 不是进程 host 峰值或单个 NUMA 节点余量。Raw 与压缩历史同时留在 host 中服务于对照，不是单一路径部署的 RAM 需求。

[完整 CSV](data/native-codec-copy-pipeline-20261031.csv) 保存全部 2,304 条主 trace 的原始 μs、同 trace 墙钟分段、计数、物理容量、实际峰值与结果文件哈希；原始 JSON 保存 49,152 次请求明细。独立审计覆盖 smoke 与主结果共 62 passed＋2 infeasible，未删除波动、负收益或失败边界。

## 来源

隔离候选记录 HEAD `9c3fc6d620a6185f511570c1997f98e254f8ba09`。原单槽评估器和 core 未修改；候选由以下源码哈希识别，不能只凭 HEAD 复现。原型后续以 `eaa0006a601ae8f609d01f3bd8ae1105e167ca4a` 提升到仓库，可作为代码复现起点，仍需对照实际实验哈希。

| 文件/证据 | SHA256 |
|---|---|
| `native_codec.cu` | `e024cbfe6b8a9e280c24a153f75b11b76d12df0e4610c25fcd4cd9ea18980175` |
| `native_codec_budget_bench.py` | `39d56438794b8d5e27446014431703b8687c030d9b9832c9c35fef1c02bcc701` |
| `native_codec_pipeline.py` | `12bdf5f5036c92e719d60f085c0e40ff18a79b72d5245daec088dac6cdac0f71` |
| `native_codec_pipeline_ab_bench.py` | `1c22b18ba9c4725e69b461a0133e5153be7fc9a58548e38e4a0b895e6e27416c` |
| `输入捕获` | `1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322` |
| `评估池` | `90d340536c304f6a0395b10f5c0abe9f73204ac2095b140b1c725485c175cb51` |
| `码表` | `8cd2b8979b12101119104f1ed92652ce938752e49a4e2d1f5c69de47ff9d0a30` |
| `独立审计 JSON` | `1aa221837a0b2fb7c2af33d2234613b4742083f8c14d792222f3098c76bc1884` |
| `主结果 CSV` | `a37182f014b7b75cdfc8635d098f014c30c9638365211094b3b09e305a96a0fe` |

| 原始结果文件 | SHA256 |
|---|---|
| `pipeline-smoke-r1.json` | `16c564a489f380e1a814231d1e44306f1240756d53133440bdd9504eacb3e5b1` |
| `pipeline-boundary-r1.json` | `aa3abe1ab782efe9229aacb21fb39a22e3124e2c88c965a9859712089211ec7a` |
| `pipeline-single-main-forward-r1.json` | `95234e44b97d38890e0cc027246899150fa5a42ed82fac0dd6c69d1ebd9bad70` |
| `pipeline-double-main-forward-r1.json` | `cd1fa8143186d7222c40423319a403ab478a31cf1b9802202e506f3ff136d297` |
| `pipeline-double-main-reverse-r1.json` | `91a4795e751f30531a2557e5bc801f8a0b8692093628a591c23aedd3276ab2a6` |
| `pipeline-single-main-reverse-r1.json` | `022e2d00e3155ae83eae4f52c70e6e27c57e37f6c8a654fd10882421abacfb65` |

| 独立 profiler 证据 | SHA256 |
|---|---|
| `pipeline-profile-r1.nsys-rep` | `e153ac4fb46b4aeffafa48ee6bc4865a2e58bfbd3055d25dfcf9dcd37ff28ac7` |
| `pipeline-profile-r1.sqlite` | `503eec6485155a1509cd5381fa596f158e62ab71c1a0a7f621a943a6732f7dc3` |
| `pipeline-profile-r1-analysis.json` | `1a96b67dc65f65292d7dcc00e9a37b1e1f64f08d652708261995fabad88347eb` |
| `pipeline-profile-r1-ranges.csv` | `a0d30234a42359e699f0f918e0442795ed0867412f54a8bcbc034bbea37a028c` |
| `pipeline-profile-r1-overlap-pairs.csv` | `4a03025639a356db99a1c045111b1fe9bff7333b1b28cb072c9a792ea593d789` |
| `pipeline-profile-r1.sql` | `65ffd77874c54a4f0fbd4e0ab23ad331f17d814637096901af18494c242775e4` |
