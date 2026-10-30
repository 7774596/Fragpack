# 原生页解码的合并写入优化（2026-10-31）

同一 H800 PCIe 上，完整校验后的 resident FPKB 恢复取得 **1.707–1.979×** 的配对轮次中位速度比。28 个配置/进程组合均保留，未筛除样本。该结果覆盖恢复操作；**恢复＋48 层 FA3 的补充结果见文末，[固定预算回放](benchmark-native-codec-coalesced-budget-20261031.md)已另页记录**。

## 改动与代价

基线直接把 MMA 布局中的解码值写入原生页。候选先在共享内存中整理 token/dimension 顺序，再协作写入连续的 BF16 维度组；满足基址及 page/token stride 对齐时使用 128-bit 写入，否则保持标量写入。输出仍是调用方注册的原生 KV 页，FPKB 格式、码表、attention、目标页映射及错误处理不变。

每个 CTA 的共享内存从 **41,568 B 增至 49,744 B**，增加 8,176 B。压缩记录完成预读后，原输入区域复用为 32,768 B 的解码 tile；escape 区域独立保留。raw-fallback 记录继续从全局输入读取。首个 emission 前和最终 scatter 前的 CTA 屏障保证覆盖与读取顺序。**没有新增全局 BF16 staging**；原生输出页本来就需要存在。此实验未测其他 GPU 的驻留 CTA 数或占用率变化。

## 对照边界

- 基于提交 `88bffe2eff204d2f952768a586fc6c1d714440f3`，实际差异由下方 CUDA 哈希固定。两份真实 CUDA 与 binding 以不同模块/host 符号加载；共同头文件、binding 和编译选项相同（`-O3 --use_fast_math -lineinfo`）。
- 每个进程内两条路径使用完全相同的 payload、offsets、码表、目标地址、page map、record-offset scratch 和 status。基线负责离线生成归档，准备缓冲区在计时前释放。
- GPU 2：NVIDIA H800 PCIe，UUID `GPU-fb9880e5-6155-b080-837a-948d5fd08b4c`；Python 3.10.20、PyTorch 2.13.0+cu129、CUDA runtime 12.9、NumPy 2.2.6。独占执行流程的启动记录显示 GPU 空闲且无其他进程。
- 输入为 Qwen3-30B-A3B 的历史 CPU BF16 KV 捕获；模型 FP8 权重先反量化，**此处不是 FP8 KV**。48 层、K/V 各 4 heads、head dimension 128，每个逻辑压缩块 128 tokens。
- 原始捕获形状 `[32,48,2,128,4,128]`；位置 0:8 校准，8:32 用于评估。扩展到 N 个物理块时循环复制这 24 个捕获块，不代表新的长上下文模型执行。N=1/4/16/64/256，原生页 P=16/128。
- `fallback25/100` 沿用已冻结的构造方法：指定完整 record 的高字节替换为码表外的有限值，低字节保留。仅测 N=4/256、P=128；实际编码后 fallback 比例分别为 25%/100%。
- 每种输入分别运行 forward/reverse 进程。每个配置在每个进程中有 24 个配对轮次，每轮 ABBA 或 BAAB，反向进程交换初始顺序；每条路径各 48 个样本。两次观察先取均值，再计算该轮基线/候选比值。
- `external=True` CUDA events 放在 CUDA Graph 内部，计时包括 header/map、record 校验和 decode/scatter。暖机、源数据准备、编译、逐位校验、host 等待不计入 GPU 时间；没有 H2D/D2H、attention、服务请求或模型计算。

共保留 5,376 个 GPU 时间样本（28 组 × 两种操作 × 两条路径 × 48），以及 1,344 个操作内配对轮次比值。轮次嵌套于各自进程，表中的 p10–p90 是样本分位区间，不是置信区间。

## 全部 resident restore 结果

时间单位为 μs。F/R 表示 forward/reverse 进程；速度比为每轮两次观察均值之比的中位数，**不是两列时间中位数相除**。

| 输入 | 进程 | N | P | 实际 fallback | 基线中位数 | 候选中位数 | 配对速度比 | 配对 p10–p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| capture-repeat | F | 1 | 16 | 0% | 43.648 | 24.016 | 1.810× | 1.689–1.853 |
| capture-repeat | F | 1 | 128 | 0% | 44.704 | 24.128 | 1.835× | 1.775–1.918 |
| capture-repeat | F | 4 | 16 | 0% | 133.728 | 76.096 | 1.746× | 1.726–1.798 |
| capture-repeat | F | 4 | 128 | 0% | 133.296 | 75.456 | 1.765× | 1.746–1.806 |
| capture-repeat | F | 16 | 16 | 0% | 452.688 | 253.024 | 1.791× | 1.777–1.798 |
| capture-repeat | F | 16 | 128 | 0% | 448.000 | 249.616 | 1.795× | 1.783–1.806 |
| capture-repeat | F | 64 | 16 | 0% | 1721.408 | 961.984 | 1.790× | 1.786–1.794 |
| capture-repeat | F | 64 | 128 | 0% | 1696.208 | 948.288 | 1.788× | 1.786–1.792 |
| capture-repeat | F | 256 | 16 | 0% | 7603.328 | 3845.088 | 1.979× | 1.975–2.021 |
| capture-repeat | F | 256 | 128 | 0% | 7433.888 | 3801.600 | 1.956× | 1.953–2.017 |
| capture-repeat | R | 1 | 16 | 0% | 43.616 | 23.872 | 1.832× | 1.797–1.892 |
| capture-repeat | R | 1 | 128 | 0% | 44.896 | 23.808 | 1.944× | 1.841–1.994 |
| capture-repeat | R | 4 | 16 | 0% | 133.344 | 75.808 | 1.777× | 1.734–1.804 |
| capture-repeat | R | 4 | 128 | 0% | 133.856 | 74.992 | 1.798× | 1.759–1.814 |
| capture-repeat | R | 16 | 16 | 0% | 452.848 | 252.848 | 1.792× | 1.782–1.797 |
| capture-repeat | R | 16 | 128 | 0% | 449.856 | 250.256 | 1.795× | 1.780–1.809 |
| capture-repeat | R | 64 | 16 | 0% | 1722.096 | 961.936 | 1.790× | 1.788–1.792 |
| capture-repeat | R | 64 | 128 | 0% | 1696.640 | 948.112 | 1.789× | 1.786–1.792 |
| capture-repeat | R | 256 | 16 | 0% | 7603.168 | 3844.672 | 1.978× | 1.975–2.020 |
| capture-repeat | R | 256 | 128 | 0% | 7436.560 | 3801.120 | 1.957× | 1.953–2.016 |
| fallback25 | F | 4 | 128 | 25% | 131.456 | 74.784 | 1.758× | 1.727–1.791 |
| fallback25 | F | 256 | 128 | 25% | 7348.400 | 3832.480 | 1.917× | 1.915–1.947 |
| fallback25 | R | 4 | 128 | 25% | 131.792 | 74.448 | 1.783× | 1.726–1.802 |
| fallback25 | R | 256 | 128 | 25% | 7371.680 | 3836.192 | 1.922× | 1.920–1.965 |
| fallback100 | F | 4 | 128 | 100% | 124.816 | 72.768 | 1.707× | 1.684–1.744 |
| fallback100 | F | 256 | 128 | 100% | 7137.184 | 3853.184 | 1.852× | 1.850–1.853 |
| fallback100 | R | 4 | 128 | 100% | 124.896 | 72.256 | 1.723× | 1.691–1.744 |
| fallback100 | R | 256 | 128 | 100% | 7134.720 | 3854.000 | 1.851× | 1.850–1.853 |

在 N=256、P=128 的原始捕获构造下，基线为 7.434/7.437 ms，候选为 3.802/3.801 ms（F/R），配对速度比均约 1.957×。25% 和 100% fallback 的同规模速度比分别约 1.917–1.922×、1.851–1.852×。这些不同输入的绝对时间保持分开，不用于推断压缩率与性能的普遍关系。

## 未改动的 raw scatter 对照

另行执行的 `raw_scatter` 不经过 FPKB 解码，代码未改动；它不是解码器内部的 raw-fallback 分支。其全部 28 组配对中位速度比为 **0.963–1.039×**，N≥4 时为 0.993–1.005×。N=1 存在正反序方向差异，保留这些波动；不能把小幅控制差异解释为 codec 收益。

[完整 CSV](optimization-native-codec-coalesced-20261031.csv) 分别保存 28 组 decode 和 28 组 raw scatter 的两个路径中位数、p10/p90、最小/最大值、配对比值、实际字节及来源哈希。原始 JSON 保留逐次样本。

## 正确性与后续范围

`coalesced-correctness-r1.xml` 记录 **32 passed、0 failed、0 skipped**：包括 P=16/32/64/128、压缩和 raw-fallback、不同 stride、非法 record/offset/映射、跨块 alias 拒绝、workspace/pool 的跨流顺序，以及 P=16/128 的混合 head 对齐和更新输入后的 Graph 重放。

六个 A/B 进程均通过完整校验：每条操作在 eager、初次 Graph 与计时后重新 poison 目标页，再检查全部 BF16 bits、空闲页 canary 和零状态。两条解码路径还检查坏 magic、目标 alias 拒绝及失败目标不写入；N=1/P=128 的单原生页配置没有可构造的重复映射。正反进程的物化输入、映射、record lengths 和归档元数据哈希一致。

这说明合并写入能降低当前原生页恢复的 GPU 时间。完整恢复＋FA3 的独立对照见文末；[固定预算回放](benchmark-native-codec-coalesced-budget-20261031.md)已单独完成。上述测量仍不代表 TTFT、生成吞吐或外部服务表现。此报告也没有共享内存占用率、HBM transaction 或 Nsight 因果归因数据。

## 来源标识

| 内容 | SHA256 |
|---|---|
| 基线 `native_codec.cu` | `27a5f604f81a16d14a9849a3bb24fc61c0506c6355ac22645e79aa3007466130` |
| 候选 `native_codec.cu` | `e024cbfe6b8a9e280c24a153f75b11b76d12df0e4610c25fcd4cd9ea18980175` |
| A/B driver | `4df99512a966de5d1545fed3500d8a1cf43d809286f54c64379230d48f42b0b4` |
| 共同 `fragpack_common.cuh` | `7eabc2316bd672bd1f725304f61549aa481fd88a5015c453ff00c0190cf0219a` |
| 共同 binding | `5b65d4d368d33af1cc14441c0bc7232bfd6851db05688b3176fae414efaec00e` |
| 输入捕获 | `1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322` |
| 校准位置 0:8 | `4745da48165dec5f503274b317cbf794144caa9eb14d540641dd1811f0370e62` |
| 评估位置 8:32 | `90d340536c304f6a0395b10f5c0abe9f73204ac2095b140b1c725485c175cb51` |
| 码表 | `8cd2b8979b12101119104f1ed92652ce938752e49a4e2d1f5c69de47ff9d0a30` |
| GPU 测试 XML | `e3092309acebe0571e32aa933ad699074349c10510881fe3e9b799dac842d08f` |

保留的原始测量文件：

| 文件 | SHA256 |
|---|---|
| `coalesced-ab-capture-repeat-forward.json` | `a8d25a3901ec920bf0136df6c4156d91ab98ba73048fb1b420bacdfb5743dee6` |
| `coalesced-ab-capture-repeat-reverse.json` | `2465abd94abeb9b25a13c3c30a1c3d5e298d424c43a68229cc5c6c6f886af4d8` |
| `coalesced-ab-fallback25-forward.json` | `62a521337654c7972b0c1e2c790fec45b7c00a85e147c78a72063826b9408f98` |
| `coalesced-ab-fallback25-reverse.json` | `fbfc160d8adf4261afa424e864a6bc4c56821361d6a93e26a95316d95dc5cb37` |
| `coalesced-ab-fallback100-forward.json` | `ce577f51d8559832002f6d12c7fb2feb60e45b49caaab86e758fd79bb7f33932` |
| `coalesced-ab-fallback100-reverse.json` | `8bf5f90b93a2b15cb1934fac62bf5f890bf6dfe8e225ae0a23ac1a0a8b1a087d` |

本次报告仅整理上述冻结候选和测量，后续代码或完整路径结果应另行标明版本与计时边界。

## 补充：恢复后直接执行 48 层原生 FA3

新增正反两个完整进程均通过。下面直接测量**完整校验、解码恢复和全部 48 层 stock
FA3** 的同一张 Graph；没有把 restore 与 attention 的组件中位数相加。每个进程
保留相同 24 轮 ABBA/BAAB，每条路径各 48 个时间样本。Q=1/128 使用相同的原生
attention 接口，`fa_version=3`、`num_splits=0`、causal；上下文 C=N×128，实际
block table 是恢复目标 permutation map 的展平视图。两条路径共享 native pages、
map、Q 和 output 地址；没有改动 attention/backend。

128-token 上下文的配对速度比只有 **1.033–1.069×**（约 3%–7%）；扩展到 32K
后，Q=1 为 **1.598–1.618×**，Q=128 为 **1.223–1.230×**。较长 query 的完整
窗口收益更小，不能把近 2× 的 restore 收益当成整个读取过程的加速。

下表为全部 full-read 配置；时间为 **ms**，每格按正向 / 反向分别列出各进程中位数，
速度比仍为进程内配对轮次比值的中位数。两个进程保持分层，不提供置信区间。

| C tokens | 原生页 P | Q | 基线 ms（F / R） | 候选 ms（F / R） | 配对速度比（F / R） |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1 | 0.436 / 0.438 | 0.412 / 0.411 | 1.055× / 1.069× |
| 128 | 128 | 128 | 0.603 / 0.608 | 0.587 / 0.580 | 1.033× / 1.040× |
| 2048 | 128 | 1 | 1.106 / 1.107 | 0.907 / 0.908 | 1.220× / 1.220× |
| 2048 | 128 | 128 | 1.711 / 1.712 | 1.512 / 1.510 | 1.132× / 1.134× |
| 32768 | 128 | 1 | 9.858 / 9.854 | 6.158 / 6.163 | 1.601× / 1.598× |
| 32768 | 128 | 128 | 19.919 / 19.858 | 16.190 / 16.175 | 1.230× / 1.228× |
| 32768 | 16 | 1 | 10.037 / 10.030 | 6.202 / 6.208 | 1.618× / 1.616× |
| 32768 | 16 | 128 | 21.029 / 20.972 | 17.157 / 17.146 | 1.225× / 1.223× |

[独立 read CSV](data/native-codec-coalesced-read-20261031.csv) 保存这两个进程的全部
32 个 operation/配置记录，包括 16 个 full-read 记录，以及同环境的 restore-only
和 raw-scatter 对照。共 3,072 个时间样本、768 个配对轮次；其中 full-read 为
1,536 个样本、384 个配对轮次。原 resident CSV 保持不变，各次运行不混合汇总。

每个操作在 eager、首次 Graph、计时后重新 poison KV；full-read 同时把所有 output
置为 BF16 NaN，再检查全部 48 层原始 KV bits、未使用页 canary、零 status，以及
全部输出的 `int16` 位模式。参考输出由原始 CPU KV bits 直接构造原生页后调用同一
stock FA3 产生，不经过 codec。**两个进程、所有层均逐位一致，输出误差为 0**；
坏 magic 与重复目标映射拒绝检查也通过。正反进程的输入、map、Q、参考输出及
FA3/driver/CUDA 哈希相同，运行结束的 source guard 均通过。

内存统计是共享 A/B 诊断上下文：同时保留 raw-control payload、Q=1 和 Q=128 的
48 层 Q/output，以及两臂各自的 Graph。N=256/P=128 时 Torch allocated 为
8,841,193,472 B，峰值 8,975,928,320 B，reserved 为 9,059,696,640 B；峰值包括
参考构建和验证，排除归档编码准备。每条路径的 Q=1/128 Graph private reserved
分别为 2,097,152 / 16,777,216 B，两臂相同；这部分 reserved 不等同于仍被张量
持有的 allocated。CPU reference 在捕获前已移回主机，临时参考 GPU 页已释放。
这些数值及 CSV 的完整分项不能作为单臂部署显存需求。

测量窗口为 2026-10-31 04:02:43–04:10:46（UTC+8）。硬件、输入和 CUDA 候选与
前文相同，vLLM 为 0.30.0+cu129；新 driver 与 FA3 来源如下。所有时间均为 resident
GPU 恢复＋只读 attention，**不含 H2D、host API/等待、完整模型前向或服务调度**；
32K 是既有 24 个捕获页的合成扩展，Q=1 也不是实际生成过程。本轮没有测 fallback
输入下的 full-read，也不把结果写成 TTFT、吞吐或服务端收益。

| 补充来源 | SHA256 |
|---|---|
| 含 read 的 A/B driver | `28925e8451b67d633c68cf0b2a8741e0204a44da5041ffe05f9dc909fd379716` |
| FA3 `__init__.py` | `68d256feb9db6c1c397dc1458f6b42e2bbdc66429ae14d6a2d41c27ef0dbd878` |
| FA3 `flash_attn_interface.py` | `d6fccbb0fe1931c79013f5942199c5c9cbba0eedd7ac5ff037f55db19c38421a` |
| FA3 `_vllm_fa3_C.abi3.so` | `0d757ff21295b789c39bbac6e342da923f8bffb8ff11c0930747d8a015ca1d3a` |
| `coalesced-read-ab-r1-forward.json` | `a9db2640e4f01aaf2a4fd6e983530009d5995e9680d1dfeabb54e255455c0c04` |
| `coalesced-read-ab-r1-reverse.json` | `eebc666f2eac9ea3c522cce31a9d877d032a712966dd676e921a2a5cf6e4250e` |
