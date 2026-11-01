# FragPack 设计文档入口

当前设计已按原 DaseR 的文档结构整理到 [系统设计目录](design/README.md)。
HBM 常驻压缩与原生 attention 的外部 codec 是两条独立路径，分别说明执行边界与证据。

| 文档 | 内容 |
|------|------|
| [整体架构](design/architecture.md) | 进程/线程拓扑、模块边界、固定版本和关键决策 |
| [组件详解](design/components.md) | 组件职责、公共接口与所有权 |
| [数据流程](design/flows.md) | 新 KV、前缀保存/恢复、失败与关闭 |
| [HBM 常驻压缩 KV](design/hbm-resident-kv.md) | 表示与 GPU 生命周期、direct/prepare 优化、gather/FA3 和 fused/split-KV |
| [原生 attention 外部 codec](design/native-attention-external-codec.md) | 标准 BF16 KV pages 与独立 codec；压缩外部对象，辅助工作区另计 |
| [原生 connector](design/native-codec-vllm-connector.md) / [传输](design/native-lmcache-transfer.md) | 已实现的 vLLM 请求生命周期与有界 CPU L1 路径 |
| [LMCache 接入](design/lmcache-integration.md) | 0.5.6 CPU L1 字节服务与 HBM 记录的显式传输 |
| [逐键查询](design/lmcache-probe.md) / [验证设计](design/native-codec-serving-validation.md) | 编码前去重边界与原生路线的实验计时范围 |
| [后续接入与优化](design/lmcache-roadmap.md) | HBM 调度接线、L2 恢复和 reservation 安全边界 |

本入口保留原有链接；设计正文以 `docs/design/` 为准。
配置与检查见 [开发说明](development.md)，实验结果见 [验证范围](validation.md)。
