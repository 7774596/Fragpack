# 独立 codec：工作区拆分与有界恢复的显存实测

2026-10-31，固定同一份 32K token KV，**Q=128 时独立压缩恢复路径的 GPU
张量峰值由 13.0059 GB 降到 3.3871 GB（减少 73.96%）**。其中，拆分只读解码
工作区先降到 6.5608 GB，再把恢复批量 K 从 256 缩到 4。最终与同 K 的 raw
路径仅差 **37,888 bytes**。这消除了接口额外持有的大缓冲；活跃 KV 仍是原生
BF16，没有获得 HBM 常驻压缩的容量收益。

下表为 Q=128 的关键对照，单位 GB（10⁹ bytes）。current 为一次恢复并读取后的
Torch allocated；peak 覆盖构造、预热、Graph 捕获和一次实际冷恢复/48 层读取；
reserved 为对应 allocator 保留量。源数据准备所用 GPU 张量已释放。

| 路径 | codec 工作区 | K | current | peak | reserved |
| --- | --- | ---: | ---: | ---: | ---: |
| 独立压缩恢复＋FA3 | all | 256 | 12.991063 | 13.005875 | 13.008634 |
| 独立压缩恢复＋FA3 | decode | 256 | 6.546018 | 6.560830 | 6.564086 |
| 独立压缩恢复＋FA3 | decode | 4 | 3.372266 | 3.387079 | 3.393192 |
| Raw 恢复＋FA3 | raw | 4 | 3.372229 | 3.387041 | 3.391095 |
| HBM 压缩＋gather＋FA3 | — | 4 | 2.573592 | 2.588404 | 2.669674 |
| HBM 压缩＋融合读取 | — | 4 | 2.506483 | 2.506484 | 2.585788 |

`all → decode` 去掉恢复阶段不需要的编码 record scratch、编码输出和编码元数据。
固定 K=256 的这一步保留相同接收区。随后 `K=256 → 4` 将接收区从
3,222,274,048 缩到 50,348,032 bytes，始终按 FPKB **最大 raw-fallback 格式**
预留；没有用本次较小的压缩 payload 代替安全容量。decode/K4 自身工作区只有
21,520 bytes。每次恢复按 K 分块，复用同一份 codec 和接收区。

剩余的大头是不可被这项接口优化消除的 **3,221,225,472-byte 原生 BF16 KV**。
Q=128 的查询和输出各占 50,331,648 bytes，FA3 临时分配也计入峰值。
HBM 两条路径则保留 2,340,421,632-byte 压缩 arena；gather 还需要
67,108,864-byte 单层 BF16 workspace。因此它们在这组输入上仍更省显存。
本次 capture 没有 raw fallback；HBM overflow 按该输入需求配置，不能把这张表
外推为任意 fallback 比例下的固定占用。

完整 **40 个 reader 配置**见
[CSV](data/native-codec-memory-20261031.csv)：Q=1/128、K=1/4/16/256、四条路径，
其中独立压缩路径分别测 all/decode。CSV 保存整数 bytes、构造及冷读取峰值、
reserved、各分配组件、校验结果和来源哈希；不含延迟结论。这里统计的是 Torch
GPU 张量，不包含模型权重、驱动/CUDA context 等非 Torch 分配，也不是整进程
物理显存峰值。pinned 元数据和主机归档在 CSV 中单列。

另行实际分配了一槽公共 `NativeCodecPool(workspace="encode")`：

| K | 独立 encode pool current＝peak（bytes） | GB |
| ---: | ---: | ---: |
| 1 | 25,274,368 | 0.025274 |
| 4 | 100,797,952 | 0.100798 |
| 16 | 402,892,288 | 0.402892 |
| 256 | 6,445,831,680 | 6.445832 |

这说明编码工作区也可随在途批量有界化；本表只测分配，没有执行编码，也不包含
原生 KV。它与 reader **分别构造和释放**，不能将 reader 数字视为读写两侧同时
驻留的总量。reader 使用单流串行复用的一份固定工作区，未经过公共 pool 的租借
接口；codec/pool 另有 **30 项 GPU 测试通过**，其中 3 项覆盖双槽跨流复用。

全部 40 项在记录峰值后另行校验：重置驻留状态，逐 chunk 填入破坏值，恢复后
检查全部 48 层 KV 位模式，再与独立原生 FA3 参考比较输出。Raw/native/gather
输出逐位相同；融合读取最大绝对误差为 Q1 的 0.00390625、Q128 的 0.03125，
均通过 `atol=rtol=0.02`。每项结束后的 Torch allocated 均为 0。一次 cold probe 仅用于
覆盖真实使用时的分配，**不用于推断延迟、吞吐或 chunk 大小的性能优劣**。

配置：H800 PCIe，UUID `fb9880e5-6155-b080-837a-948d5fd08b4c`；Python 3.10.20、
PyTorch 2.13.0+cu129、vLLM 0.30.0+cu129。48 层、4 KV heads、32 query heads、
D=128、128-token 页、256 个驻留 blocks。FA3 使用 v3、causal、`num_splits=0`；
HBM 融合路径在 Q1 使用 decode、Q128 使用 prefill。输入是历史 CPU
Qwen3-30B-A3B/OpenHands BF16 KV capture，FP8 权重先转为 BF16；页 0:8 校准、
8:32 循环扩展至 32K，并非新的长上下文模型运行。没有运行完整模型或服务调度。

原始归档为 `20261031-native-codec-budget-optimization/results/` 下的
`deployment-memory-primary.json`，测试记录为 `workspace-pool-tests-r1.xml`。
测量时间为 2026-10-31 02:54:49–02:56:57（UTC+8）。复现入口是
[`native_codec_memory_bench.py`](../benchmarks/native_codec_memory_bench.py)，使用
`--contexts 32768 --queries 1,128 --chunks 1,4,16,256 --receive-capacity format-max
--measure-encode-pool`，另提供 `--dump`、`--out` 和准确的 `--input-description`。

- Git：`88bffe2eff204d2f952768a586fc6c1d714440f3`。
- 源码集合 SHA256：`9f48cbae4879fa791dc71336bf3a1cf0efc623c202eb1c3e035a443f719af631`。
- 原始 JSON SHA256：`c7ce69259370b81fe4210779679c74ad087f31923da3b01ccd396729de7085b5`。
- 输入 SHA256：`1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322`。
- 码表 SHA256：`8cd2b8979b12101119104f1ed92652ce938752e49a4e2d1f5c69de47ff9d0a30`。
