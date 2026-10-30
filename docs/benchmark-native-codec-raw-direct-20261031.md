# 原生 Raw 直传基线与 Native 恢复粒度（2026-10-31）

**去掉 Raw 的 receive/scatter 能腾出容量，但并不保证更快；Native 的恢复粒度需要同时考虑容量和提交次数。** 在 0.45 GB 短会话场景，Native 从 K=4 改为 K=2 后恢复了第四个可保留会话，耗时降低 25.5%–25.9%，并比同期 RawDirect 低 16.3%–16.8%。0.5 GB 下 K=2/K=4 的快慢随运行顺序改变，不能宣布一个统一最优值。

主基线四个进程、后续粒度两个进程全部完成，共 56 个路径/配置/进程组合、672 条计时 trace、15,360 次请求。两组分开列示，保留原始慢例与方向翻转。

## 配置与可比边界

H800 PCIe，UUID `GPU-fb9880e5-6155-b080-837a-948d5fd08b4c`；PyTorch 2.13.0+cu129、vLLM 0.30.0+cu129。48 层、BF16 KV、GQA 32:4、D=128、128 tokens/block。历史 CPU 捕获 0:8 校准、8:32 评估，经旋转/重复生成不可变会话；本组无 raw fallback，也不是新的真实 32K 捕获。

| 路径 | 外部归档 → GPU 原生 BF16 页 | GPU 恢复辅助空间 |
|---|---|---|
| RawPacked | packed raw H2D → validated GPU scatter | K=4 receive＋status |
| RawDirect | 预先排为 layer/KV/block/token/head/dim 的 pinned raw → 连续页段直接 H2D | 无 receive、codec 或 restore graph |
| Native | 压缩 H2D → validated 独立 codec | 有界 receive＋decode metadata/codebook |

三者使用相同 stock FA3；无 attention backend 改动。RawDirect 在 CPU 完整验证目标映射与布局，逐层/KV/连续段提交 copy，完成一次请求后等待；实际归档拥有者保留至完成。另两条路径统一 `graph/request`，整请求检查全部 GPU status 后才允许 attention。主基线 K=4；后续 Native K∈{1,2,4}，K 表示一次恢复的 block 数，不是 attention 参数。

Raw 的两种布局 payload 字节数与逻辑 KV 完全相同，归档 hash 因布局改变；跨路径的压缩归档、capture indices、Q 和源数据 FA3 reference hash 均核对一致。Native 在 packed/direct 两进程组之间是同一代码与输入的时间对照。RawDirect 的离线布局准备与 Native 编码均不计时。

耗时是串行会话 LRU 回放的整条 host wall：策略、目标准备、实际 pinned H2D、恢复、状态处理、48 层读取及等待。每个组合暖机一条，再记录 12 条 trace；每条重置逻辑 LRU。不是 outbound KV 往返、模型 serving、TTFT、AgentX 或 SSD/network 测量。GPU tensor 预算包含实际 KV 池和所有工作区、Q/output、graph 分配，容量通过实际 fit 与 next-block 边界核验。

## 主基线：K=4 全部结果

A/B 均为五个 1,024-token 会话、Q=1、24 次四会话局部复用＋第五会话干扰，预算分别 0.45/0.5 GB。C 为三个 32,768-token 会话、Q=1、8 GB；D 同样三个长会话、Q=128、12 GB，均为 20 次首次载入、工作集扫描与集中复用。

| 场景 | RawPacked blocks / miss | RawDirect blocks / miss | Native blocks / miss |
|---|---:|---:|---:|
| A | 31 / 19 | 35 / 11 | 31 / 19 |
| B | 35 / 11 | 37 / 11 | 35 / 11 |
| C | 627 / 10 | 635 / 10 | 627 / 10 |
| D | 939 / 3 | 944 / 3 | 939 / 3 |

运行顺序为 Packed-F→Direct-F→Direct-R→Packed-R；R 同时反转场景和路径顺序。下表单位 ms，均为进程内 12 条 trace 中位数；“Native 降幅”是 `1−Native/Raw`，负数表示 Native 更慢。两条路径依次执行，**不是 round 内配对测量**。

| 场景 | 归档组 / 顺序 | Raw wall | Native wall | Native 降幅 |
|---|---|---:|---:|---:|
| A | Packed-F | 56.518 | 51.556 | +8.8% |
| A | Direct-F | 61.295 | 47.358 | +22.7% |
| A | Direct-R | 44.523 | 48.602 | -9.2% |
| A | Packed-R | 54.908 | 44.926 | +18.2% |
| B | Packed-F | 40.916 | 34.018 | +16.9% |
| B | Direct-F | 40.376 | 35.605 | +11.8% |
| B | Direct-R | 42.754 | 34.516 | +19.3% |
| B | Packed-R | 38.379 | 32.672 | +14.9% |
| C | Packed-F | 781.219 | 585.054 | +25.1% |
| C | Direct-F | 816.233 | 640.446 | +21.5% |
| C | Direct-R | 721.662 | 628.938 | +12.8% |
| C | Packed-R | 697.630 | 525.337 | +24.7% |
| D | Packed-F | 475.936 | 404.365 | +15.0% |
| D | Direct-F | 478.512 | 409.748 | +14.4% |
| D | Direct-R | 466.275 | 427.450 | +8.3% |
| D | Packed-R | 471.938 | 414.776 | +12.1% |

A 的 RawDirect 容量从三会话增至四会话，miss 从 19 降为 11；Native K=4 仍只有三会话。Direct-F 中 Native 为 47.358 ms、RawDirect 为 61.295 ms；Direct-R 却为 48.602 / 44.523 ms，方向翻转。该场景不能仅凭前一次慢 RawDirect 结果宣布优势。

B/C/D 中 RawDirect 增加的 blocks 没有跨过整会话容量门槛，miss 不变。同期 Native 相比 RawDirect 的中位耗时降幅分别为 11.8%–19.3%、12.8%–21.5%、8.3%–14.4%；这些是本组有序进程观测，不是稳定加速保证。RawDirect 相比 RawPacked 也没有一致的时间优势。

时间漂移主要可定位到记录的“策略＋恢复”墙钟区间，不能进一步归因到未测量的 CPU/PCIe 原因：例如 C 的四次 Native 中位 trace 为 585/640/629/525 ms，读取区间中位值约 47.4–47.6 ms。分项中位数取自不同 trace，不应相加还原总中位数。

RawDirect 的 restore metadata H2D、status D2H 和 codec chunks 均为零；读取阶段的 block table/length H2D 仍然存在。A/B 每 trace 有 11 个连续 run、1,056 次公开 copy 调用；C 为 10 / 960，D 为 3 / 288。Native K=4 相应有 38/22/640/192 个 restore chunks。RawDirect 所有主 trace 的实际映射均连续；碎片映射单独作功能和 profiler 验证。

## 后续迭代：减小 Native receive 粒度

该矩阵在主基线运行后单独预注册，只改配置。仍用 A/B 的相同 KV、Q、24 请求轨迹，每个 K 单元重新测一次同期 RawDirect；RawDirect 的计时恢复实现不依赖 K，这些重复值是时间对照。F 按预算递增、K1→K2→K4、Raw→Native；R 完全反向。每组合仍为 12 条 trace。

| 预算（GB） | K | Native receive（MB） | Native blocks / miss / chunks | RawDirect blocks / miss |
|---:|---:|---:|---:|---:|
| 0.45 | 1 | 12.587 | 34 / 11 / 88 | 35 / 11 |
| 0.45 | 2 | 25.174 | 33 / 11 / 44 | 35 / 11 |
| 0.45 | 4 | 50.348 | 31 / 19 / 38 | 35 / 11 |
| 0.5 | 1 | 12.587 | 37 / 11 / 88 | 37 / 11 |
| 0.5 | 2 | 25.174 | 37 / 11 / 44 | 37 / 11 |
| 0.5 | 4 | 50.348 | 35 / 11 / 22 | 37 / 11 |

全部实际容量在 F/R 中一致。0.5 GB 的 K=1 实测为 37 blocks，低于预注册粗估的 38；本表只用实际 allocation/graph/cold-read peak。K=1/2/4 的 codec 逻辑空间为 12,292/15,368/21,520 B，receive 才是更大的辅助占用。

下表每格是 **F / R**，wall 单位 ms；降幅相对同一单元的 RawDirect，未将不同 K 或进程样本混合。

| 预算（GB） | K | RawDirect wall | Native wall | Native 降幅 |
|---:|---:|---:|---:|---:|
| 0.45 | 1 | 40.255 / 40.255 | 34.675 / 34.750 | +13.9% / +13.7% |
| 0.45 | 2 | 40.126 / 40.202 | 33.599 / 33.439 | +16.3% / +16.8% |
| 0.45 | 4 | 40.355 / 40.262 | 45.070 / 45.125 | -11.7% / -12.1% |
| 0.5 | 1 | 40.206 / 40.262 | 34.787 / 34.649 | +13.5% / +13.9% |
| 0.5 | 2 | 40.206 / 40.268 | 33.445 / 33.386 | +16.8% / +17.1% |
| 0.5 | 4 | 40.183 / 40.161 | 32.770 / 34.625 | +18.4% / +13.8% |

0.45 GB 下，K=2 将 receive 从 50.348 MB 缩至 25.174 MB，保留会话数由三增至四，miss 19→11、payload 1.348→0.780 GB；两个顺序均比 K=4 少约 25.5%–25.9% wall，并比同期 RawDirect 少 16.3%–16.8%。K=1 也恢复第四个会话，但每 trace 88 次 chunk 恢复，比 K=2 的 44 次更多，本组 wall 稍高。

0.5 GB 下各 K 都可保留四个会话，miss 和 payload 相同。K=4 正序比 K=2 快，反序却更慢；缩小 batch 会同时改变 transfer/launch 数、kernel 批量与峰值，不能把差值单独解释为 Python 开销。可把 K=2 作为紧预算场景的候选，继续按实际容量门槛选择，而不是统一缩小或增大 K。

后续 K=4 的 0.45 GB 结果为 Native 45.070/45.125 ms、RawDirect 40.355/40.262 ms，两个顺序都更慢；这与其额外 miss 一致。原主基线中 61.295 ms 的慢 RawDirect 仍完整保留。也保留本矩阵的尾部慢样本，例如正序 A/K4 的 RawDirect 最大 58.888 ms、Native 最大 67.605 ms；不筛选或把嵌套 trace 当作独立服务样本计算置信区间。

## 正确性、内存与独立 profiler

29 项新增 CPU 测试覆盖 BF16 特殊位、复制布局、全局重复/越界/非法映射、晚层几何错误、源/目标别名拒绝和提交异常后的 stream drain；提升前全量 CPU 297 项通过。GPU 独立 contract 用任意 BF16 bit patterns 验证 48 层、五块、非零连续/逆序碎片/一般碎片映射：位值精确、未用页不变、调用者 stream 恢复；四类非法映射拒绝且零写入。

GPU smoke 的四个路径配置，以及两组主结果的 56 个配置全部通过单独 poison/restore 位检查与源 FA3 输出逐位比较。Native 的映射/status error gate 也通过。每次 fit、计时 peak 和 next-block 边界符合预算，退出后 allocated=0；源文件前后哈希和 GPU 独占监控通过。六个主进程均通过 32 GB host/cgroup headroom gate。NUMA 绑定不代表 host/PCIe 或 CPU 独占，未测 host 峰值。

独立 Nsight contract 的三个闭合范围只观察到预期 pinned H2D 与完成事件，没有 kernel、D2D/D2H、allocation 或 launch API：

| 映射 | 连续 runs | 实际 H2D 次数 | Payload 字节 | 新增 GPU tensor 存储 |
|---|---:|---:|---:|---:|
| 非零连续 | 1 | 96 | 62,914,560 | 0 |
| 逆序碎片 | 5 | 480 | 62,914,560 | 0 |
| 一般碎片 | 3 | 288 | 62,914,560 | 0 |

DMA/API correlation 与功能账本逐项一致，DMA 和完成事件均在 runtime stream 13；不将验证 stream 7 错当作无 GPU 活动的 caller stream。144 个 poison kernels 和 144 次验证 D2H 均在恢复范围外。live/peak 都为 113,246,208 B，支持这些恢复没有额外 GPU tensor staging/codec/scatter，不约束驱动内部存储或离线 CPU 排版。

保留 Nsight 的 “Not all NVTX events might have been collected”“No NVTX events collected” 和 CUDA events 可能不完整三条诊断。指定的 864 H2D、144 fill kernels、144 验证 copies 与三个范围均核对完整；不声称所有无关 event/API 类别完整。该 profile 仅验证执行路径，不用于主性能耗时。

[完整 CSV](data/native-codec-raw-direct-20261031.csv) 保存全部 672 条主 trace 原始 μs、同 trace 墙钟分段、计数、容量、receive/codec/live/setup/timed peak 和原始结果哈希。两份独立审计分别重建 whole-session LRU、物理 slots、所有传输计数和逐轮不变量；原始 JSON 保留全部请求明细和对照失败边界。本轮没有 infeasible 主配置。

## 来源

实际候选 HEAD 为 `eaa0006a601ae8f609d01f3bd8ae1105e167ca4a`，下列哈希标识新增候选。RawDirect 随后提升为 `74ea52e`，可作为实现复现起点；K 矩阵只改 JSON 配置。没有修改 codec core、attention 或旧 benchmark 的历史结果。

| 源码/证据 | SHA256 |
|---|---|
| `native_codec_raw_direct.py` | `ccb9f8adc5e2eac816b39e3fa6e4920812006a0c7d2d13f9d7458622e2a5b135` |
| `native_codec_budget_bench.py` | `ace561513f5610b57cd9dcbbca482d70f6023b05ae311b88cbda052c3d659370` |
| `native_codec.cu` | `e024cbfe6b8a9e280c24a153f75b11b76d12df0e4610c25fcd4cd9ea18980175` |
| `输入捕获` | `1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322` |
| `评估池` | `90d340536c304f6a0395b10f5c0abe9f73204ac2095b140b1c725485c175cb51` |
| `码表` | `8cd2b8979b12101119104f1ed92652ce938752e49a4e2d1f5c69de47ff9d0a30` |
| `主基线独立审计` | `053951a69d4e736adc673603879263c447b676bfe9a315c3feeb71a4623d75ea` |
| `粒度独立审计` | `0769be4081acd82101bbdf36134645c898610e2bc915e042e88b49c9136769c1` |
| `完整 CSV` | `2165c6954b7c68ea38f8157fa1ac154ecc40841e96c4a800712e144780977014` |

| 原始结果 | SHA256 |
|---|---|
| `raw-direct-contract-r1.json` | `bb13ae00348be9445331e36c05afcd1320db091067c71fc075b2b68dcd0b6afb` |
| `raw-direct-smoke-r1.json` | `cb4e28b863192db3d1be4ff07c8ab2d7dc073e7fd5e420cd731c6b081531a61a` |
| `raw-direct-packed-forward-r1.json` | `5a33c7fb758b5d3cdddd8f52905e33db9a838d26b0ac5e48f761d936f9452ece` |
| `raw-direct-direct-forward-r1.json` | `eac089a84375fe84f6407115752dddffc8fae26d5b7a8310c0798dd06c7a4c92` |
| `raw-direct-direct-reverse-r1.json` | `0c5cc51a309df2755144ccfbf80a42c5578282f17c2a5d8e6c1d9b00f405abc7` |
| `raw-direct-packed-reverse-r1.json` | `12d284d5d875950b96f9a9b119dd9a4b956d97dc54e8bc6531fc67fc8b35dee4` |
| `raw-direct-granularity-forward-r1.json` | `cea40520e835763901778adeebdce178574df3702ba1457812457917a021307d` |
| `raw-direct-granularity-reverse-r1.json` | `20504d753c405927d2e5927cb75884366da2adf5650493dd805ad9b591e9e159` |

| 独立 profiler 证据 | SHA256 |
|---|---|
| `raw-direct-profile-r1.nsys-rep` | `0a60870efd47afb23b2bc4e06f3f90614208ad28c7c8d547710656503a4e22da` |
| `raw-direct-profile-r1.sqlite` | `96036bd92230e1f61a257fb400db186856816b7897fb49592c0c3f10c1ddb22f` |
| `raw-direct-profile-r1-analysis.json` | `81f8fd822def1a6ddd5b11213c63818caaa189040138f0b23e01176834ce79eb` |
| `raw-direct-profile-r1-ranges.csv` | `7e7fc4f0ee4ba09e23e943b952ac7009183aa93db435e9fa3dcf0c852cc67198` |
