# FragPack 系统设计文档

FragPack 是面向 LLM 推理的无损 BF16 KV 表示与 GPU 编解码模块，保留两条独立路径。
HBM 常驻压缩路径让已满 block 以压缩记录驻留 GPU，通过自有 attention adapter 使用。
原生 attention 路径让 GPU KV 保持 BF16，只在外存边界编解码；现已通过自有
`FragPackNativeConnector` 和 `NativeBlockTransfer` 接通 LMCache 0.5.6 CPU L1。
后者的小模型功能和[有限协议主模型对照](../validation-native-codec-serving-20261101.md)
已完成：原生 FA3、TP=1、eager、CPU L1、离线串行回放。它不增加活跃 GPU KV 容量。
本轮总请求及发布窗口改善不等于 TTFT 加速，W18 外部恢复的 TTFT 反而高于重算。
HTTP、NVMe、TP>1 和 CUDA Graph 尚未验证；HBM 路线的 scheduler connector 仍待实现。

本文档组沿用原 DaseR `docs/design/` 的组织方式：整体架构、组件职责、
完整数据流程和专题设计。记录的是独立 FragPack 仓库的实现与决策，
不把 DaseR 的 server、NVMe backend 或性能结果视为已经迁入的能力。

## 目录

| 文档 | 内容 |
|------|------|
| [整体架构](architecture.md) | 模块依赖、进程与线程拓扑、启动顺序、固定版本和关键设计决策 |
| [组件详解](components.md) | core、vLLM adapter、LMCache client/server 的职责、公共接口和状态归属 |
| [数据流程](flows.md) | 新 KV、前缀匹配、保存、恢复、失败与退出的完整流程 |
| [HBM 常驻压缩 KV](hbm-resident-kv.md) | 记录格式、GPU 生命周期、内存区域、direct 写入、prepare 优化、gather/FA3 与 fused/split-KV attention |
| [LMCache 压缩记录接入](lmcache-integration.md) | 固定 0.5.6 的 CPU L1 插件、前缀键、字节协议、有界传输、所有权与失败处理 |
| [LMCache 后续接入与优化](lmcache-roadmap.md) | 原 HBM 压缩记录接线的规划；变长 L2 恢复、预留代次与回滚边界仍待实现 |
| [原生 attention 与外部 KV 压缩](native-attention-external-codec.md) | native-page codec、已实现的 CPU L1 模型接线、当前验证范围及容量边界 |
| [原生页与 LMCache 的传输](native-lmcache-transfer.md) | 有界后台 NativeBlockTransfer、FPKB 压缩与 FPNR Raw 对照、源读取和发布完成点 |
| [原生 vLLM connector](native-codec-vllm-connector.md) | 公开 scheduler/worker 接口、完整 prompt 保存、异步恢复与原生页所有权 |
| [编码前存在性查询](lmcache-probe.md) | 逐对象 probe、在编码前避免重复工作、非租约和失败边界 |
| [原生 codec 服务验证协议](native-codec-serving-validation.md) | 固定预算的 Raw/压缩对照、组件与模型实验的计时和证据边界 |

---

## 阅读与维护约定

先读整体架构和数据流程，再按问题进入专题。`已实现` 表示存在对应代码，
不自动意味着已经通过模型服务、TP 多卡或性能验证；`待接入`、`待验证`、
`候选方案` 分别记录尚缺的实现、证据和未选择的设计。

接口、记录格式、block 生命周期或所有权改变时，更新本目录对应的设计。
配置命令见 [开发说明](../development.md) 和
[LMCache 接入与验证](../lmcache-plugin-056.md)；历史实验报告保留其原始版本、
配置与测量范围，不随当前设计改写。证据入口见
[验证范围](../validation.md)、[LMCache 固定版本选择](../lmcache-integration-target-20261030.md)
及 [daser-codec 参考实现评审](../lmcache-daser-codec-review-20261030.md)。
