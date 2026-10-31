# 合并写入优化的固定预算回放（2026-10-31）

新的 12 进程对照中，合并写入使 Native 路径的完整 trace 耗时下降 **5.48%–8.15%**，
对应 **1.058–1.089×**。三组负载、两种进程内顺序都观察到改善；每个 arm 的实际
容量、命中、淘汰及传输字节在两份源码间完全相同。Raw、Gather 和 Fused 控制的
基线/候选中位数比为 0.996–1.017×，未出现 Native 的同等增幅。

这是独立的新实验，不与[原容量矩阵](benchmark-native-codec-fixed-budget-20261031.md)
或 [resident GPU 时间](optimization-native-codec-coalesced-20261031.md)混合汇总。
本轮测量真实 pinned H2D、每 chunk 的 status D2H/等待、整会话 LRU 和 48 层只读
attention 的 **host wall time**；不是 TTFT、生成吞吐或完整模型服务。
校准、归档编码、参考输出和恢复校验不计时；D2H 在此仅传回恢复状态，未测新 KV
编码及迁出 payload 的往返过程。

## 协议与全部时间结果

每组配置顺序为 **基线 F → 候选 F → 候选 R → 基线 R**，每项是独立进程。
F 的 arm 顺序为 Raw、Native、Gather、Fused，R 反转该顺序。每个 arm 预热 1 条、
记录 3 条相同 trace；每条从空逻辑 LRU 开始。下面的比值是**同顺序两个进程中位数
之比**，不是 resident 实验中同进程的配对轮次比，也不把三条 trace 当作独立服务
重复。所有轮次原值及 12 个文件的 SHA256 保存在[完整 CSV](data/native-codec-coalesced-budget-20261031.csv)。

时间单位 ms；每格保留 F / R 两个进程，不将顺序波动合成置信区间。

| 配置 | 路径 | 基线 ms（F / R） | 候选 ms（F / R） | 基线/候选（F / R） |
|---|---|---:|---:|---:|
| C=1024，Q=1，0.5 GB | Raw | 51.754 / 51.836 | 51.587 / 51.524 | 1.003× / 1.006× |
| C=1024，Q=1，0.5 GB | Native | 46.443 / 46.139 | 43.478 / 43.574 | 1.068× / 1.059× |
| C=1024，Q=1，0.5 GB | Gather＋FA3 | 33.212 / 32.966 | 33.051 / 32.830 | 1.005× / 1.004× |
| C=1024，Q=1，0.5 GB | Fused | 23.279 / 23.147 | 23.247 / 23.223 | 1.001× / 0.997× |
| C=4096，Q=1，0.5 GB | Raw | 95.564 / 95.386 | 95.169 / 95.194 | 1.004× / 1.002× |
| C=4096，Q=1，0.5 GB | Native | 81.497 / 81.656 | 74.863 / 75.001 | 1.089× / 1.089× |
| C=4096，Q=1，0.5 GB | Gather＋FA3 | 80.446 / 80.499 | 80.211 / 80.103 | 1.003× / 1.005× |
| C=4096，Q=1，0.5 GB | Fused | 73.604 / 73.418 | 73.235 / 73.218 | 1.005× / 1.003× |
| 混合 C，Q=128，1 GB | Raw | 108.541 / 109.289 | 108.986 / 107.504 | 0.996× / 1.017× |
| 混合 C，Q=128，1 GB | Native | 97.653 / 97.806 | 92.301 / 91.754 | 1.058× / 1.066× |
| 混合 C，Q=128，1 GB | Gather＋FA3 | 70.265 / 69.886 | 69.650 / 69.775 | 1.009× / 1.002× |
| 混合 C，Q=128，1 GB | Fused | 166.425 / 166.435 | 166.444 / 166.367 | 1.000× / 1.000× |

Native 的改善幅度明显小于单独恢复 kernel 的近 2×：H2D 字节、读取次数和同步
数量都未减少。C=4096 时，每次 miss 的 policy＋restore 平均 wall 从约 6.92 ms
降至 6.26 ms，读取平均 wall 仍约 0.76–0.77 ms。这些分项用于定位变化，不相加
分项中位数来构造总耗时。

在候选进程内部，与同容量、同命中序列的 Raw 相比，Native trace 耗时分别下降
**15.4%–15.7%、21.2%–21.3%、14.7%–15.3%**。这包含既有压缩减少 H2D 的作用，
不能全部归因于本次合并写入优化。

## 容量与历史复用保持一致

C=1024 为 5 个会话、28 次访问；C=4096 为 2 个会话、16 次访问。混合为
`[128,1024,4096]` 重复两组，共 6 个会话、32 次访问。所有路径使用同一 trace、
整会话 LRU、串行请求和完整的 pinned 主机历史；没有 SSD、网络或主机历史淘汰。
下面每格依次为 **Raw / Native / Gather / Fused**，数值在四个进程中完全相同。

| 配置 | 实际容量 blocks | 每条 trace 的 miss | payload H2D GB |
|---|---|---|---|
| C=1024，Q=1，0.5 GB | 35 / 35 / 47 / 47 | 16 / 16 / 5 / 5 | 1.610613 / 1.135223 / 0.354759 / 0.354759 |
| C=4096，Q=1，0.5 GB | 35 / 35 / 46 / 47 | 10 / 10 / 10 / 10 | 4.026532 / 2.838036 / 2.838036 / 2.838036 |
| 混合 C，Q=128，1 GB | 66 / 66 / 88 / 91 | 19 / 19 / 6 / 6 | 3.107979 / 2.190619 / 0.727253 / 0.727253 |

C=1024 时，HBM 压缩容纳全部工作集，miss 为 5 而非 16，因此 Gather/Fused 仍更快。
C=4096 时四条路径都无法同时容纳两个会话，miss 都为 10；候选 Native 约 75 ms，
此时快于 Gather 的约 80 ms，接近 Fused 的约 73 ms。混合 Q=128 中 Gather 的
容量收益仍明显，而当前 Fused reader 约 166 ms，不能用其较高命中率推断更低耗时。
这些容量差异来自表示和 reader 的原有内存构成，本次 CUDA 改动没有增加驻留容量。

预算为 Torch GPU 张量峰值上限；K=4、`workspace=decode`，接收区按 raw-fallback
格式上界预留。预算包含 KV、codec、receive、metadata、attention scratch、Graph
及回放保留的 48 层 Q/output。混合 Q=128 的 Q/output 各占 50,331,648 B，这是
测试 fixture，不是实际 serving 的常驻分配模型。模型权重和 CUDA context/驱动
分配不在该上限内；CSV 分别记录组件、实测 allocated 峰值和 allocator reserved。

## 校验与来源

12 个进程全部通过，共 **48 个 arm/进程记录、144 条计时 trace**，没有失败或
不可行配置，所有轮次均保留。独立校验在每次恢复前破坏目标内容，逐 chunk 检查
全部 48 层 KV bits，并核验每次 attention 输出。Raw/Native/Gather 输出逐位一致；
Fused 在既有 `atol=rtol=0.02` 下通过，最大绝对误差 0.03125。每个 arm 的下一个
block 容量边界已实际探测，所有计时峰值不超预算，释放后 Torch allocated=0。

审计逐请求重算 LRU、命中/淘汰、payload/metadata H2D、status D2H 和 chunk 数；
各源码/顺序的计数、输入、Q、trace 和归档相同。完成后重新核对两份冻结工作树，
其纳入摘要的文件**唯一差异为 `fragpack/ops/native_codec.cu`**。原生写入优化不
影响另外三个 arm 的算法；其小幅变化作为控制波动保留。

硬件为 NVIDIA H800 PCIe，UUID `fb9880e5-6155-b080-837a-948d5fd08b4c`；
Python 3.10.20、PyTorch 2.13.0+cu129、vLLM 0.30.0+cu129。48 层、4 KV heads、
32 query heads、D=128、128-token 页、BF16 KV、seed=314。输入是历史
Qwen3-30B-A3B/OpenHands CPU 捕获：模型 FP8 权重先转 BF16，KV 为 BF16；页 0:8
校准、8:32 循环构造历史。不是 FP8 KV、新长上下文采样、AgentX 或生成测试。

测量时间为 **2026-10-31 04:24:20–04:34:13（UTC+8）**。执行计划为
`coalesced-fixed-budget-abba-r1.json`；原始文件均以 `coalesced-budget-` 开头，
由 CSV 的 `source_report` 和 SHA256 对应。基线工作树 HEAD 为
`b35354c48e1a6108a3df9a6ca01087f04f374f22`，候选基于
`88bffe2eff204d2f952768a586fc6c1d714440f3` 加冻结 CUDA 补丁，具体身份如下：

| 内容 | SHA256 |
|---|---|
| 基线 CUDA | `27a5f604f81a16d14a9849a3bb24fc61c0506c6355ac22645e79aa3007466130` |
| 候选 CUDA | `e024cbfe6b8a9e280c24a153f75b11b76d12df0e4610c25fcd4cd9ea18980175` |
| 两者共同 budget driver | `5ad6be2f235d803f10e2a9db95466b9b1236bdf0a5821b8cdf7938e606264a8f` |
| 基线 core/benchmark 摘要 | `0d82e9ebb9c0bf95d015d94d4d6661aa0158f7e3ce61333a4a47e508a0217a8f` |
| 候选 core/benchmark 摘要 | `67a8e5ff23d52d978731095f568110fa5a346a205450a1bca0f9cb685933c32e` |
| 输入 | `1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322` |
| 码表 | `8cd2b8979b12101119104f1ed92652ce938752e49a4e2d1f5c69de47ff9d0a30` |
