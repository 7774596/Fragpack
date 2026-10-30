# 同进程配对的恢复调度实验（2026-10-31）

在容量、所有分配和地址完全相同的 Runtime 内，Native 从逐 chunk 提交/等待改为
Graph 提交＋每请求等待，K=1 的配对中位数改善 **1.097–1.100×**；K=4 两组负载
为 **1.022–1.027×**。K=1 的主要收益来自减少主机等待；只改变 Graph 提交的收益
较小。K=4 有轮次未改善，因此不将这组小幅差异写成稳定的服务端到端加速。

所有模式都使用已验证的 coalesced CUDA `e024cbfe…`，没有再次修改 codec 算法或
attention。本实验单独比较恢复调用方式，不与[上一轮 CUDA 优化的进程间结果](benchmark-native-codec-coalesced-budget-20261031.md)
拼接计算加速比。测量为真实 pinned H2D、状态 D2H、CPU 策略和 48 层只读 attention
的完整 trace wall；没有新 KV 编码/迁出、完整模型服务、TTFT 或生成吞吐。

## 配对协议

每个 arm 只构造一次包含全部 Graph 和 request 元数据的 Runtime，按同一预算确定
容量后，在四种模式间切换。各模式共用相同 KV/receive/codec/Q/output 地址、live
GPU 分配及容量；这衡量**相同额外分配下的调度差异**，不是各模式的最小部署内存。

| 缩写 | restore 提交 | 检查恢复结果的主机等待 |
|---|---|---|
| EC | eager-capacity | 每 chunk |
| GC | CUDA Graph | 每 chunk |
| ER | eager-capacity | 每个 miss 请求结束时 |
| GR | CUDA Graph | 每个 miss 请求结束时 |

Graph 包含原有 GPU 恢复校验、恢复计算和状态初始化；H2D、host 状态检查和后续
attention 在 restore Graph 外。所有模式的 attention 都已经使用相同的 48 层图。
request 模式保持一个接收区和一个有序 stream，预先填好互不覆盖的 pinned 元数据，
每 chunk 的状态复制到不同主机位置，最后检查全部状态再允许 attention。
它减少 CPU 等待次数，**没有实现传输与解码重叠**。

压缩路径的 EC/ER 与 GC/GR 都传入固定 receive 容量的视图；实际 H2D 只搬运归档
payload。归档内容和 offsets 已先核验且保持不变。该视图的长度是**接收区容量，
不是本次实际传入字节数**，因此这不是面向任意动态或不可信输入的 transferred-length
校验接口，也不能据此移除实际长度检查。Raw 仍使用 `count × raw_block_bytes` 的
精确视图。Graph 还固定了捕获时的地址、shape 和 chunk count，不能直接视为通用
动态恢复 API。

F 进程按 Raw→Native→Gather→Fused 执行，R 反转 arm 顺序和四种模式的初始顺序。
每个 arm、每种模式先完成独立校验、cold 预算探测及 1 条预热 trace，再记录 12 轮。
每轮四种模式各运行一次，循环 Latin 顺序使每种模式在四个位置各出现三次；每条
trace 都从空逻辑 LRU 开始。实际执行器使用 `numactl --cpunodebind=0 --membind=0`；
这是 NUMA 节点绑定，不是 CPU 独占或固定单核，仍可能存在主机/GPU 时序波动。

表中时间是各模式 12 条 trace 的中位数；比值是**先按同轮 EC/GR 相除、再取中位数**，
不是相邻两列中位数之比。F/R 分别保留，不将 12 轮当作独立进程或计算置信区间。
[完整 CSV](data/native-codec-paired-dispatch-20261031.csv) 包含全部 288 个配对轮次、
1,152 条计时 trace 的原值和 1,440 个比值；另外两种条件对照 ER/GR、GC/GR 也保留。

## 全部路径结果

时间单位 ms；每格为 F / R。

| 配置 | 路径 | EC ms（F / R） | GR ms（F / R） | 配对 EC/GR（F / R） |
|---|---|---:|---:|---:|
| C=1024，Q=1，0.5 GB，K=1 | Raw | 59.393 / 60.020 | 55.529 / 56.467 | 1.069× / 1.065× |
| C=1024，Q=1，0.5 GB，K=1 | Native | 54.774 / 53.922 | 50.137 / 49.212 | 1.097× / 1.100× |
| C=1024，Q=1，0.5 GB，K=1 | Gather＋FA3 | 36.630 / 36.369 | 34.978 / 34.686 | 1.047× / 1.045× |
| C=1024，Q=1，0.5 GB，K=1 | Fused | 26.148 / 26.261 | 24.401 / 24.621 | 1.057× / 1.064× |
| C=1024，Q=1，0.5 GB，K=4 | Raw | 57.539 / 56.381 | 56.149 / 56.364 | 1.017× / 1.001× |
| C=1024，Q=1，0.5 GB，K=4 | Native | 47.833 / 45.861 | 47.509 / 44.736 | 1.022× / 1.024× |
| C=1024，Q=1，0.5 GB，K=4 | Gather＋FA3 | 33.851 / 33.364 | 32.876 / 32.881 | 1.013× / 1.017× |
| C=1024，Q=1，0.5 GB，K=4 | Fused | 24.551 / 24.119 | 24.249 / 24.023 | 1.010× / 1.011× |
| 混合 C，Q=128，1 GB，K=4 | Raw | 115.601 / 120.353 | 114.687 / 119.429 | 1.005× / 1.012× |
| 混合 C，Q=128，1 GB，K=4 | Native | 97.071 / 102.347 | 95.096 / 100.141 | 1.023× / 1.027× |
| 混合 C，Q=128，1 GB，K=4 | Gather＋FA3 | 73.239 / 72.220 | 71.725 / 70.879 | 1.007× / 1.010× |
| 混合 C，Q=128，1 GB，K=4 | Fused | 168.163 / 167.254 | 166.151 / 167.404 | 1.008× / 1.002× |

Native 的分项配对如下。P10–P90 仅描述这 12 轮的分布，不是置信区间。

| Native 配置 | 仅 Graph：EC/GC | 仅减少等待：EC/ER | 两者：EC/GR | EC/GR 的 P10–P90（F；R） |
|---|---:|---:|---:|---|
| C=1024，Q=1，0.5 GB，K=1 | 1.034× / 1.032× | 1.081× / 1.089× | 1.097× / 1.100× | 1.078–1.136；1.086–1.141 |
| C=1024，Q=1，0.5 GB，K=4 | 1.010× / 1.010× | 1.015× / 1.020× | 1.022× / 1.024× | 0.987–1.055；1.007–1.041 |
| 混合 C，Q=128，1 GB，K=4 | 1.008× / 1.006× | 1.024× / 1.024× | 1.023× / 1.027× | 0.998–1.048；0.992–1.097 |

K=1 时，Raw 的 EC/GR 也达到 1.065–1.069×，Gather/Fused 约 1.045–1.064×：
减少等待是共同调用链的改善，不能全部归为压缩算法收益。K=4 时多数变化只有几个
百分点，若干单独 Graph 对照低于 1。上述配对不能消除所有波动，也不能与早前独立
进程的绝对耗时直接相减。

## 实际容量、字节和等待

Q=1 两组都是 5 个 1024-token 会话、28 次访问；混合为 `[128,1024,4096]` 重复两组，
6 个会话、32 次访问。序列包含首次访问、立即复用、工作集轮巡和连续重复访问，
均为串行、不可变历史读取；“burst”只表示局部性，没有并发到达或生成增长。
所有 arm 使用相同整会话 LRU 和完整 pinned 主机历史，没有 SSD、网络或主机历史淘汰。

下表数值在两进程及四模式中保持一致；仅 restore 等待从 chunk 数变为 miss 数。
**每次 attention 后的完成等待仍存在**，因此该列不是程序全部同步次数。

| 配置 | 容量 blocks：R / N / G / F | miss：R / N / G / F | restore 等待：Raw/Native；Gather/Fused |
|---|---|---|---|
| C=1024，Q=1，0.5 GB，K=1 | 37 / 37 / 51 / 51 | 16 / 16 / 5 / 5 | 128→16；40→5 |
| C=1024，Q=1，0.5 GB，K=4 | 35 / 35 / 47 / 47 | 16 / 16 / 5 / 5 | 32→16；10→5 |
| 混合 C，Q=128，1 GB，K=4 | 66 / 66 / 88 / 91 | 19 / 19 / 6 / 6 | 67→19；22→6 |

传输字节完全不随四种模式改变，实际计数如下。metadata 包含恢复映射/offsets 及
attention 的 block IDs/长度；request 等待保留全部 status D2H，没有省略检查。

| 配置 | payload H2D B：Raw / Native / Gather / Fused | metadata H2D B：R / N / G / F | status D2H B：R / N / G / F |
|---|---|---|---|
| C=1024，Q=1，0.5 GB，K=1 | 1610612736 / 1135222784 / 354758656 / 354758656 | 1520 / 3568 / 1808 / 1808 | 512 / 512 / 160 / 160 |
| C=1024，Q=1，0.5 GB，K=4 | 1610612736 / 1135222784 / 354758656 / 354758656 | 1520 / 2800 / 1568 / 1568 | 512 / 512 / 160 / 160 |
| 混合 C，Q=128，1 GB，K=4 | 3107979264 / 2190618624 / 727252992 / 727252992 | 2956 / 5468 / 3128 / 3128 | 988 / 988 / 328 / 328 |

这些是共用 superset Runtime 的测量容量。HBM 压缩路径在这里容纳全部历史工作集，
miss 较少；Native/Raw 的容量及命中序列相同。Graph 或延后主机等待本身未增加容量，
也未减少需要恢复的数据。

## 内存和校验边界

预算为 Torch GPU 张量的实际 allocated 峰值，包括构造/捕获、最大的首次恢复/read
以及每种模式的完整 trace；不含 CUDA context/驱动开销，allocator reserved 单列。
这不是物理整进程 HBM 上限。CSV 保留各 arm 的组件、构造峰值及每轮峰值/reserved。
Native 的四模式共用分配如下；两进程相同。

| Native 配置 | 共用 live allocated B | 构造＋首次恢复/read 峰值 B | 接收区 B | request pinned 元数据 B |
|---|---:|---:|---:|---:|
| C=1024，Q=1，0.5 GB，K=1 | 478961664 | 479077888 | 12587008 | 192 |
| C=1024，Q=1，0.5 GB，K=4 | 491566080 | 491682304 | 50348032 | 144 |
| 混合 C，Q=128，1 GB，K=4 | 981516288 | 996326400 | 50348032 | 576 |

全部模式预留 `format-max` 接收区以容纳 raw fallback；不能以本次观测压缩大小
替代容量上界。request 的额外 pinned 空间在 EC/GC 中也保留，捕获的 restore 图
及其状态在 eager 模式也保留。Q=128 时，本回放保留全部 48 层的 Q/output，
各为 50,331,648 B，这是验证/回放 fixture，并非真实服务通常的逐层生命周期。
模型权重不在预算内；本轮也没有测每种模式单独部署时可释放多少空间。

两个进程均 exit 0，24 个 arm/进程组合和 96 条独立功能校验 trace 全部通过，
没有失败或不可行配置。恢复前破坏目标，检查全部 48 层 KV bits；Raw/Native/Gather
的每次 attention 输出与原始来源 FA3 reference 逐位一致，Fused 使用既有
`atol=rtol=0.02`，最大绝对误差 0.03125。各实际 chunk count 的两组不同 payload/
目标映射及未选中块保护通过。request 模式还注入首块错误，验证后续成功不会覆盖
早先错误、错误目标不变、全部状态仍被检查、attention 未越过失败检查。

审计独立重算所有 trace 的 LRU、命中/淘汰、字节、chunk/等待次数与五组配对比值，
核对各轮声明的固定地址/live 分配检查；所有峰值均不超过指定预算，下一 block
容量边界实际探测，释放后 allocated=0。校验、预热、模式切换、审计和文件写入不
进入性能样本；计时内保留 CPU policy/metadata、H2D、restore/status、attention/
完成等待和逐请求结果收集。完整输出校验是在计时前单独执行，计时 trace 内保留
每次恢复状态检查。

## 来源

运行时间：2026-10-31 04:42:28–04:52:21 CST。NVIDIA H800 PCIe，
UUID `fb9880e5-6155-b080-837a-948d5fd08b4c`；Python 3.10.20、PyTorch 2.13.0+cu129、
vLLM 0.30.0+cu129。48 层、4 KV heads、32 query heads、D=128、128-token 页，
BF16 KV，seed=314。Q=1 也是只读请求；Fused 使用 decode reader，其余使用原生 FA3。
输入为既有 CPU BF16 捕获，0:8 校准、8:32 构造会话，存在页内容轮转/重复；
这不构成新的真实长上下文模型捕获或模型精度实验，本输入没有 raw fallback record。

归档标识：`20261031-native-codec-budget-optimization`；执行计划/manifest 标识：
`dispatch-paired-main-r1`。实测快照 Git 基点为 `7b552c56666c087514bd5944fdba176aac839ca6`，
包含后续冻结改动，具体身份以以下文件和汇总 SHA256 为准；两进程的源码、case 和
输入结束时 guard 全部为真。

| 对象 | SHA256 |
|---|---|
| driver：`native_codec_dispatch_ab_bench.py` | `2b7f69744350727d7c87224c06fb79a63973b0a2ed7d6da1742bad3a85be5eb8` |
| 本轮 helper：`native_codec_budget_bench.py` | `39d56438794b8d5e27446014431703b8687c030d9b9832c9c35fef1c02bcc701` |
| `native_codec.cu` | `e024cbfe6b8a9e280c24a153f75b11b76d12df0e4610c25fcd4cd9ea18980175` |
| core＋helper＋driver 文件清单摘要 | `4242f568eafea593e86df106db19b8b4b7376f2cab4febb99167e6f434f7d87d` |
| `dispatch-paired-main.json` | `b79e7278d9c28701eec648081a66746af6a564da911711c43014864d3afba9a7` |
| 输入 capture | `1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322` |
| 码表 | `8cd2b8979b12101119104f1ed92652ce938752e49a4e2d1f5c69de47ff9d0a30` |
| held-out 页池 | `90d340536c304f6a0395b10f5c0abe9f73204ac2095b140b1c725485c175cb51` |
| `dispatch-paired-main-r1-forward.json` | `c0ec18e1141a8af24f1b9edf8b97370d03544b6588bf24d19b5a9b6ab43ff510` |
| `dispatch-paired-main-r1-reverse.json` | `60fe6d31ede2900646c3d7ae31cdfa62e88166eb13c0dc33a6c290281a920bcf` |
