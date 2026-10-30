# 独立 native-page codec：往返验证与两轮优化

实验于 2026-10-30—31 完成。分支 `feat/native-attention-external-codec`，基础提交
`9edec04`。新增代码尚未提交。**独立 codec 已能逐位往返；本次最终压缩路径比对应
raw 路径减少约 29.5% 的实传字节，完整往返中位数降低 7.0%–19.5%。**
第二轮主要改善小批次固定成本，大批次总耗时没有稳定优于第一轮。

这是 GPU codec 与真实主机传输实验，不是模型端到端、TTFT 或吞吐测试。
未修改 attention、vLLM、LMCache 或既有 HBM store，也未接通原生服务的 offload 调度。

## 实现与边界

[NativeKVPages / NativeCodec](../fragpack/ops/native_codec.py) 和
[独立 CUDA extension](../fragpack/ops/native_codec.cu) 提供：

- 原生 BF16 页 → 编码 record 槽 → 长度/偏移规划 → 紧排 FPKB v1。
- 实际 pinned-host D2H → H2D 到独立接收缓冲 → 校验 → 直接 scatter 回原生页。
- raw 对照采用相同页映射、同样的 gather / D2H / H2D / scatter。

没有完整 BF16 解码中间 workspace；编码临时 record 槽和压缩传输缓冲仍存在。
码表、FPKB 格式、逐位语义和 raw fallback 均保持一致。core 不做主机 readback 或
全设备同步；调用者负责 stream 依赖、生命周期和成功状态检查。
变长传输必须观察实际长度，本实验把该观察、元数据复制与完成等待计入总耗时。
记录格式校验不等于 checksum，结构合法的任意数据位翻转仍需传输层完整性保障。

## 配置与计时

| 项目 | 本次设置 |
| --- | --- |
| GPU | 物理卡 2，H800 PCIe，UUID `GPU-fb9880e5-6155-b080-837a-948d5fd08b4c`；每阶段前检查独占 |
| 环境 | Python 3.10.20、PyTorch 2.13.0+cu129、CUDA 12.9、SM90；CPU/内存绑定 NUMA 0 |
| KV | 48 层、每层 4 个 KV heads、head_dim 128、BF16；每个逻辑 block 含 128 token |
| 实测布局 | 128-token 物理页，源/目标非同一缓冲，目标页映射重排；16/32/64 页与非连续 stride 由功能测试覆盖 |
| 输入 | 历史 CPU Transformers forward 的 4096-token capture；Qwen3-30B-A3B-Instruct-2507-FP8 权重转为 BF16 后捕获，KV 本身为 BF16 |
| 内容来源 | 拼接的 OpenHands 轨迹，不是 AgentX，也不是 vLLM GPU serving capture |
| 拆分 | 页 `0:8` 校准；`8:24` 开发；`24:32` 保留位置集，不参与优化选择 |
| 主计时 | 每 arm 3 次预热、30 次重复，逐轮交替先后顺序；保留集 12 次，反转起始顺序 |

输入为 uint16 位表示 `[32,48,2,128,4,128]`，SHA256 为
`1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322`。
校准/评估分离的是同一条序列中的位置，不能解释为独立文档或跨负载泛化。
所有性能 case 的 raw fallback 为 0；高 fallback 输入只验证正确性，未测性能。

主结果使用 host wall time，包含布局处理、编码/紧排、实际字节传输、长度/status
观察及完成等待；只有必要的完成事件，raw 两个、压缩三个。初始化、分配、JIT、
校准、输入上传与正确性校验在计时外。不包含 RPC、网络、SSD、attention 和模型执行。
每对样本间保存日志，两条路径交替顺序；这不是持续并发吞吐测量。

分阶段 profiling 单独执行，每 arm 3 个样本。CUDA event 区间包含主机 launch 间隙，
不能当作纯 kernel 时间，也不能把各阶段中位数相加冒充完整往返中位数。

## 两轮变化与完整往返

1. **R1：页定位使用 shift/mask。** 16/32/64/128 页大小均为 2 的幂，替换运行时
   除法/取模；raw 和压缩共同使用相同优化。
2. **R2：并行记录偏移规划与 header 长度校验。** 256-thread block scan 按 tile
   处理任意记录数，使用 64 位偏移；保留全 blob 拒写、payload 和映射检查。
   同时补上主机操作阶段检查，禁止使用未初始化的规划数据；该公共 Python 版本
   在三个最终对照中完全相同，唯一源码差异是 `native_codec.cu`。

下表为完整往返中位数，单位 **ms**；下降率比较 R2 同次运行中的 raw 与压缩。

| block 数 | Raw 基线 | 压缩基线 | Raw R1 | 压缩 R1 | Raw R2 | 压缩 R2 | R2 下降 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.693 | 0.704 | 0.661 | 0.668 | 0.688 | 0.640 | 7.00% |
| 4 | 2.277 | 1.986 | 2.208 | 1.904 | 2.261 | 1.886 | 16.58% |
| 16 | 8.453 | 6.906 | 8.220 | 6.690 | 8.399 | 6.761 | 19.51% |

R1 压缩往返较初版降低 5.18% / 4.11% / 3.12%。R2 较 R1 在 1/4 blocks 降低
4.16% / 0.92%，但 **16 blocks 增加 1.05%**。两轮合计相对初版降低
9.12% / 4.99% / 2.11%，不能写成每轮、每个批量都加速。

R2 主计时 p10–p90：raw 为 `0.641–0.727 / 2.203–2.310 / 8.333–8.465 ms`，
压缩为 `0.584–0.696 / 1.833–1.937 / 6.710–6.850 ms`。这些是同进程重复样本的
分位数，不是置信区间。

分阶段中位数，单位 **μs**；每格按“初版 → R1 → R2”排列：

| blocks | 编码 | 偏移规划 | 校验＋解码＋scatter |
| ---: | --- | --- | --- |
| 1 | 83.55 → 78.11 → 78.11 | 36.00 → 35.49 → 8.13 | 111.58 → 98.37 → 61.50 |
| 4 | 211.10 → 200.99 → 201.54 | 37.63 → 37.63 → 8.64 | 180.35 → 179.68 → 142.08 |
| 16 | 712.45 → 679.30 → 679.26 | 40.96 → 40.35 → 10.27 | 500.35 → 490.37 → 456.29 |

16 blocks 的 R2 诊断中，D2H/H2D 为 2.753/2.583 ms，紧排为 0.159 ms；传输仍是
主要耗时。相比 R1，D2H/H2D 分别增加约 112/12 μs，raw 控制的 D2H 也变慢。
这是观察到的传输时段变化，尚不能确定原因；仅三个独立诊断样本不能精确归因
完整往返的 70 μs 回退。

## 保留位置集

下表仍为 ms，所有值均通过真实传输后的逐位检查：

| blocks | 初版 raw / 压缩 | R1 raw / 压缩 | R2 raw / 压缩 | R2 相对 raw 下降 |
| ---: | --- | --- | --- | ---: |
| 1 | 0.694 / 0.727 | 0.665 / 0.684 | 0.710 / 0.669 | 5.77% |
| 4 | 2.254 / 1.982 | 2.198 / 1.887 | 2.248 / 1.863 | 17.12% |
| 8 | 4.323 / 3.659 | 4.179 / 3.559 | 4.288 / 3.548 | 17.27% |

R2 的单块 raw p90 达 1.663 ms，存在明显尾部抖动；不能据此宣布稳定的小批量
收益。初次 R2 开发集 12 轮的 16-block 试验也有抖动：raw/压缩中位数
9.955/7.042 ms，最大值 13.832/12.624 ms。该次结果完整保留，未删除离群样本；
随后三个版本统一执行上面的 30 轮确认协议，没有从多次结果中选最小耗时。

## 实传容量与额外代价

单方向 payload 包含 header、对齐 padding 和所有记录：

| blocks | Raw bytes | 压缩 bytes | 减少 |
| ---: | ---: | ---: | ---: |
| 1 | 12,582,912 | 8,859,648 | 29.59% |
| 4 | 50,331,648 | 35,471,360 | 29.52% |
| 16 | 201,326,592 | 141,905,920 | 29.51% |

16 blocks 时，按功能用途核算的 GPU 工作/传输缓冲为 raw **402.653 MB**、
压缩 **604.293 MB**。压缩侧包括 201.327 MB record scratch、201.392 MB 输出、
201.392 MB 接收区，以及码表/元数据；当前输出和接收区按最坏 raw 长度预留。
pinned 区约为 raw 201.327 MB、压缩 201.392 MB。

这不包含两边相同的原生 source/target、描述符和页映射，也不是两个独立进程的
实测峰值。计时进程中两 arm 共存，Torch allocated 约 1747.775 MB。
**实传字节减少不等于工作区预留减少，也没有扩大活跃 GPU KV 容量。**

下一步优先限制在途批量、复用/缩小 record 和传输缓冲，再考虑 coalesced layout
conversion 与异步重叠。接服务时应比较相同 lazy-offload 策略的 raw/压缩路径，
核算源页 pin、后台 GPU 争用、实际复用率与 TTFT；本轮没有实施这些后续优化。

## 验证、源码与复现

- CPU suite：**149 passed**；目标 GPU suite：**23 passed**，无跳过。
- GPU 覆盖不同页大小、stride/2-byte offset、随机 uint16 位模式及 NaN/Inf 等边界、raw fallback、
  CPU 参考逐 record 字节比对、未选页 canary、重复目标拒写、非法 header/length/payload、
  跨 256-record tile 和操作阶段顺序。
- 三个最终版本的非 CUDA 源码哈希完全相同；码表、输入、实际记录长度及页映射一致。
  CUDA SHA256 前缀：初版 `abcf31a27a2f`，R1 `0a4a345a9075`，R2 `27a5f604f81a`。
- 原始 JSON、逐样本数据、日志、码表与完整源码快照保存于仓库外验证目录
  `20261030-native-codec-roundtrip`。主文件为 `baseline-final.json`、`round1-final.json`、
  `round2-final.json` 与相应 `*-heldout.json`；早期仪表化结果单独保留，未混入主表。
- 未进行原生模型服务、CUDA Graph、TP>1、网络/SSD、并发重叠或质量评估。

确认 GPU 独占并选择正确 NUMA 节点后运行；首次 JIT 编译不计入稳态结果：

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" TORCH_EXTENSIONS_DIR=/path/to/data/jit-cache \
  numactl --cpunodebind=0 --membind=0 \
  python -m benchmarks.native_codec_bench \
  --dump /path/to/captured-kv.npy --out /path/to/data/roundtrip.json \
  --blocks 1,4,16 --calib-pages 8 --eval-start 8 --eval-count 16 \
  --page-tokens 128 --rounds 30 --warmup 3
```

保留集使用 `--eval-start 24 --eval-count 8 --blocks 1,4,8 --rounds 12 --order reverse`。
数据来源应通过 `--input-description` 记录，单独准备输入可用 `--prepare-only`。
