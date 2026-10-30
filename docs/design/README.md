# FragPack 系统设计文档

FragPack 是面向 LLM 推理的无损 BF16 KV 表示、GPU 存储与 attention 模块。
已满 block 以压缩记录常驻 HBM；现有 vLLM adapter 负责模型执行适配，
可选 LMCache 0.5.6 插件负责压缩记录的外部内存缓存存取。

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
| [LMCache 后续接入与优化](lmcache-roadmap.md) | 待完成的 vLLM 调度接线、变长 L2 恢复、预留代次与回滚边界、替代方案和验证顺序 |
| [原生 attention 与外部 KV 压缩](native-attention-external-codec.md) | 独立 native-page codec 往返已验证；原生服务与延迟迁出调度尚待接入 |

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
