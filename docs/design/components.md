# 组件详解

本文档按生产代码的实际依赖方向划分组件。表示和 GPU core 不依赖 vLLM、
LMCache 或外部存储策略；两个 integration 分别适配推理引擎和缓存服务，原生
vLLM connector 调用 LMCache integration 的公开传输接口。
整体进程关系见[整体架构](architecture.md)，调用顺序见[数据流程](flows.md)。

## 组件职责

| 组件 | 位置 / 执行上下文 | 职责 |
|------|------|------|
| CPU reference codec | `fragpack/compression/attn_page.py` | 定义记录几何、编码/解码参考实现、FPKB blob 尺寸规则；保留 BF16 原始位模式 |
| `DeviceTables` / table loader | `fragpack/compression/attn_tables.py` | 离线校准、加载码表、构造 GPU 查表张量；不维护请求生命周期 |
| `HbmKvStore` | `fragpack/ops/hbm_kv.py`，模型 worker | 管理记录地址/长度、未完成 block 的 staging、临时 scratch、raw overflow；提供 prepare/write/encode/commit、gather 和 blob 导入导出 |
| `NativeKVPages` / `NativeCodecPool` | `fragpack/ops/native_codec.py`、`native_codec_pool.py`，传输后台线程 | 对标准 BF16 pages 建立视图，以分离的有界 encode/decode 工作区完成外部对象编解码；不替换 attention 或管理引擎 block |
| native store kernels | `fragpack/ops/hbm_kv.cu` | 在 CUDA stream 上执行存储状态变更、编码、读取和搬运；不决定外部缓存接纳或淘汰 |
| decode attention | `fragpack/ops/fused_attn_decode.py` 与同目录 `fused_attn_decode.cu` | 从压缩或 raw 记录恢复计算所需操作数，执行 decode attention 和 partial 合并 |
| prefill attention | `fragpack/ops/fused_attn_prefill.py` 与同目录 `fused_attn_prefill.cu` | 实验性融合 prefill；提供 split 数规划、FP32 partial workspace 规格和 split/combine kernel |
| native extension loader | `fragpack/ops/fragpack_ext.py` 与同目录 `fragpack_ext.cpp` | JIT 编译与 Python/CUDA 接口绑定；缓存目录由部署环境配置 |
| vLLM plugin | `fragpack/integrations/vllm/plugin.py` | 经 `vllm.general_plugins` 注册 `CUSTOM` attention backend，检查固定依赖版本 |
| `OpenBlocks` | `fragpack/integrations/vllm/lifecycle.py`，model-runner 线程 | 镜像 scheduler 的请求 block 列表与已计算长度，识别真正跨步存活的未满块，绑定本轮 request index |
| V2 runner wrapper | `fragpack/integrations/vllm/compat.py` | 包装固定版本的 `execute_model` 和 `gather_batch_req_state`，将 CPU 侧 block 所有权与请求顺序交给 tracker |
| `PrefillPlan` / `plan_prefill_groups` | `fragpack/integrations/vllm/prefill.py` | 按一轮 workspace 内物理 block 的并集分组，生成共享 decoded page 的映射；纯 CPU 规划 |
| `FragPackMetadataBuilder` | `fragpack/integrations/vllm/backend.py` | 绑定引擎分配的 arena，创建 store、码表和各类 workspace，构造本步 decode/prefill 元数据 |
| `FragPackAttentionImpl` | 同上，模型当前 CUDA stream | 每层依次 write、attention、encode；最后一层 commit；默认 prefill 使用 gather + FA3 |
| `FragPackAttentionBackend` | 同上 | 声明 SM90 / BF16 / block 128 / head dimension 128 等约束，调整 KV page spec 和 `LBNHC` 布局 |
| `FragPackCapacity` | `fragpack/integrations/vllm/capacity.py` | 在 store 创建/重绑定时发布主机侧容量快照；读取不触发 CUDA 或 RPC |
| `FragPackNativeConnector` / native state | `fragpack/integrations/vllm/native_connector.py`、`native_connector_state.py`，scheduler / worker | 使用公开 KV connector hooks 协调连续前缀恢复、finish-time 保存、源页延迟释放及空 step 进度；保留标准 attention |
| `CacheIdentity` / `BlockKey` | `fragpack/integrations/lmcache/identity.py` | 生成表示、模型、码表、TP shard 的 namespace 和带 tenant salt 的链式前缀键 |
| FPKB / FPNR / RPC validation | `fragpack/integrations/lmcache/protocol.py` | 按 representation 校验 envelope、几何、表摘要和实际长度；规范化未定义 padding；封装有界 JSON + bytes 消息 |
| `PrefixCacheConnector` | `fragpack/integrations/lmcache/connector.py`，显式调用者 | 将完整 token 前缀、物理 block 列表和 `PrefixMatch` 转成 lookup/store/restore 操作 |
| `BlockTransfer` | `fragpack/integrations/lmcache/transfer.py`，提交线程 + 单个后台线程 | 拥有固定 GPU / pinned-host 槽池、任务队列、CUDA event 和完成 Future；后台执行长度回读、DMA、RPC、导入状态检查 |
| `NativeBlockTransfer` | `fragpack/integrations/lmcache/native_transfer.py`，提交线程 + 单个后台线程 | 保留 native page map，在有界批次内执行预查、编解码、DMA 和 RPC；raw 对照使用独立 FPNR namespace |
| `BlobClient` / `RemoteBlobClient` | `fragpack/integrations/lmcache/client.py` | 定义同步字节存取契约，并通过官方 `server_module_call` 发起有超时的串行 RPC |
| `FragPackServerModule` | `fragpack/integrations/lmcache/server.py`，LMCache server | 由官方 external-module factory 加载，分派 `fragpack.blob.v1` 请求；持有插件状态，不拥有共享 server context |
| `LMCacheL1Store` | `fragpack/integrations/lmcache/storage.py`，LMCache server | 仅用 0.5.6 公共 StorageManager 接口申请实际长度的 CPU uint8 对象、读锁管理、发布确认和错误处理 |
| LMCache StorageManager | 官方 LMCache 0.5.6 | 拥有 CPU L1 分配、驻留、淘汰和预留生命周期；当前 FragPack 插件拒绝配置任何 L2 adapter |

`PrefixCacheConnector` 是 HBM 路线的显式调用接口，**不是 vLLM 的
`KVConnectorBase_V1` 实现**。HBM attention backend 的 runner wrapper 只提供
本地 KV 生命周期和 batch identity，不会自动调用 LMCache lookup 或报告外部
load/save 完成；这条路线的调度器接线属于[后续设计](lmcache-roadmap.md)。独立的
[`FragPackNativeConnector`](native-codec-vllm-connector.md) 已实现公开 connector
接口，使用原生 BF16 pages，不经过该 runner wrapper。

---

## 状态与内存所有权

| 状态 / 内存 | 所有者 | 可以复用或释放的条件 |
|------|------|------|
| 请求的物理 block 分配 | 引擎 / 显式调用者 | 满足模型计算和外部 transfer 的保留条件后，由引擎决定回收；transfer 不替调用者 pin 引擎 block |
| 每层压缩 arena | 引擎提供 tensor，`HbmKvStore` 管理内部记录 | 记录仍被 attention/export 读取时不可改写；每层保存自己的 KV 内容 |
| 原生 BF16 pages | 引擎分配，原生 connector 保留源/目标所有权 | 源页保留到本请求所有批次源读取成功完成；恢复目标须等本请求全部恢复批次成功才可交给 attention；本地 APC 共享页不作为可写恢复目标 |
| 原生 encode/decode 工作区与 page maps | `NativeCodecPool` / `NativeBlockTransfer` | 单个传输后台线程借还；按批次上限复用，不能由 future 取消提前归还 |
| persistent staging | `HbmKvStore` | 对应未满 block 跨步保留所有层数据；在生命周期允许时释放或交接给新 tail |
| scratch 与 attention workspace | store / metadata builder | 前一层或前一轮消费者在有序 stream 上完成后，地址才能复用；不表示不同层 KV 内容相同 |
| raw overflow slots | `HbmKvStore` | 由记录地址表和原子分配状态共同维护；失败导入不得释放仍由旧记录拥有的槽 |
| direct 模式原始 K/V | 模型层调用者 | 完整层 batch 的原始张量必须保持存活且不可变，直到该层所有 gather 和 encode 完成 |
| export 源 block | 引擎 / transfer 调用者 | `StoreTicket.source_ready` 完成后才允许复用；不必继续等待远端发布 |
| GPU / pinned-host transfer 槽 | `BlockTransfer` / `NativeBlockTransfer` | GPU 操作和 CPU 字节消费者全部结束后归池；无法确认 CUDA 完成时隔离并保留 |
| restore 目标 block | 引擎 / transfer 调用者 | 持有独占目标直到已提交写入全部结束；transport ticket 报告连续成功前缀，原生 connector 另按整请求判定恢复成功 |
| RPC request / Future | `RemoteBlobClient` 和调用者持有的 RequestClient | 每个 wrapper 最多一个在途 RPC；不确定传输失败后拒绝新请求，由调用者关闭底层连接 |
| LMCache L1 reservation / read lock | 官方 StorageManager | 插件遵守公共 reserve/finish/read 协议；不能把 Python 对象引用当作超出 TTL 的空间保护 |

这里有三种不同的完成：GPU `commit()` 完成本步 store 状态转换；
`source_ready` 表示 export 或 native encode 不再依赖源 block；外部 `completion`
才报告发布或恢复结果。HBM ticket 使用已记录的 CUDA event，native ticket 使用
后台确认 GPU 完成后成功解析的 Future。原生请求的源页在预查、排队及尚未提交的
后续批次期间也被保留；只有本请求所有批次的 `source_ready` 均成功，worker 才报告
`finished_sending` 允许引擎释放源页。它不必等待外部发布排空。
它们不能相互替代。LMCache 预留过期、缺少 generation token 的限制见
[接入设计](lmcache-integration.md)。

---

## HbmKvStore 的公开接口

| 接口 | 输入与输出 | 调用约束 |
|------|------|------|
| `prepare(slot_mapping, num_tokens, live_blocks)` | GPU int64 token-slot 映射、有效 token 数、CPU int32 未满块列表 | 每步第一层 write 前调用；负责 GC/claim/closing 列表；位于 graph capture 外 |
| `write(layer, key, value, slot_mapping)` | 当前层 BF16 `[N, H, 128]`，允许行/head stride | 当前 stream 入队；direct 命中的完整新块跳过 staging 写入 |
| `encode(layer, max_close, key=..., value=...)` | 层号、闭合 block 数上界；direct 模式携带完整层 batch K/V | 必须在该层 attention 后；保留 direct 输入到编码完成 |
| `commit()` | 本步状态，无外部缓存返回值 | 最后一层 encode 后入队；不是 RPC 发布通知 |
| `gather(layer, blocks, out_k, out_v, key=..., value=...)` | 物理 block 列表与调用者拥有的 BF16 输出 workspace | 同流读取记录；活动 direct 块需要原始层输入；workspace 不可与并发消费者冲突 |
| `layer_tables(layer)` | 返回该层记录地址、长度和查表张量 | 供融合 attention 使用；地址是本进程 GPU 指针，不是可持久化的缓存记录 |
| `export_blocks(blocks, out, stride)` | 闭合 block、对齐 uint8 目标、槽间距；返回 GPU int64 实际长度 | 所有层编码完后提交，源保留至导出完成；长度按 FPKB 对齐规则返回 |
| `import_blocks(src, offsets, blocks)` | 对齐 blob 数据、GPU int64 offsets、不同的 int32 目标 block；返回每块 int32 status | 目标非 open，且不可被 attention 读取；只有 status 为零的整块成功，失败块需重算 |
| `memory_usage()` / `poll()` / `stats()` | 容量分类 / 统计状态 | 容量快照与执行错误分开；阻塞统计不应被加入每层热路径 |

这些接口的 CUDA 异步返回不表示 kernel 已执行完。跨 stream 调用者要建立显式
event 顺序，并保留相关 tensor、记录及其物理 block。具体 kernel 和记录不变量见
[HBM 常驻 KV](hbm-resident-kv.md)。
独立 native codec 的 pages、workspace 与批次契约见
[原生 attention 外部 codec](native-attention-external-codec.md)。

接口实现入口：[HbmKvStore](../../fragpack/ops/hbm_kv.py)、
[vLLM backend](../../fragpack/integrations/vllm/backend.py)、
[V2 兼容适配](../../fragpack/integrations/vllm/compat.py)。

---

## 前缀与外部缓存接口

| 接口 | 可观察行为 |
|------|------|
| `CacheIdentity.block_keys(tokens, cache_salt)` | 每个完整 128-token block 生成一个键，后继键依赖完整前缀；最后不足 128 tokens 不生成外部对象 |
| `PrefixCacheConnector.lookup(tokens, cache_salt=...)` | 返回从首块起连续命中的 `PrefixMatch`；结果不持有 L1 pin，也不保证稍后 restore 仍命中 |
| `store_complete(tokens, block_ids, start_block=0, max_blocks=None)` | 用全前缀 token/block 列表选择有限批次，排除未满尾块；返回 `StoreTicket`，无完整块或未接纳时返回 `None` |
| `restore(match, destination_blocks)` | 提交与 key 一一对应的目标块；返回 `RestoreTicket`，空匹配或未接纳时返回 `None` |
| `BlockTransfer.submit_store/submit_restore` | key 和 block 数必须匹配，设备 ID 必须由调用者保证有效且互不重复；槽不足时整批拒绝，不做部分隐式接纳 |
| `NativeBlockTransfer.submit_store/submit_restore` | key 对应完整 128-token 逻辑对象和 native page map；有界批次满时返回 `None`；源读取完成与发布完成由独立 Future 报告 |
| `StoreTicket.source_ready` | CUDA event；只控制源 block 何时不再被 export 使用 |
| `NativeStoreTicket.source_ready` | Future；只有成功结果证明本批源页读取完成，异常或取消不能作为释放依据；原生请求需汇合所有批次 |
| `StoreTicket.completion` | Future 的有序 bool tuple；第一个未发布块及其后续返回 False；异常可能发生在较早块已发布之后 |
| `RestoreTicket.completion` | transport 批次的连续成功前缀；首次 miss 或 import/decode 失败处及后续为 False；不等同于原生请求恢复成功 |
| `BlobClient.put/get/lookup_prefix` | 确认已驻留或相同已有字节 / 拷贝得到 bytes 或 miss / 连续前缀长度；调用可以阻塞，应放在模型每层执行路径之外 |
| `BlobClient.probe_many` | 逐 key 返回存在性提示，跨缺口继续，不读取 payload、不持有驻留租约；native transfer 用于编码前跳过已存在对象 |
| `BlockTransfer.close(timeout=...)` | 停止接纳并等待已有任务；超时不等于取消，worker 和 buffer 所有权继续保留 |

`PrefixCacheConnector` 不把一个超出槽池的长前缀自动拆成多个排队任务。
调用者按池容量分批、确认先前批次结果后继续；保存后续批次时仍传入完整 token
前缀，不能截去前文重新计算哈希。恢复分批遇到第一个缺口即停止后续复用。
原生 connector 自行维护有界分批进度；其 source Future、延迟释放和取消规则见
[原生页传输](native-lmcache-transfer.md)，逐键预查语义见[查询设计](lmcache-probe.md)。
原生请求只有全部计划恢复对象成功后才报告成功接收并跳过相应 prefill。任一批次
失败时，worker 等待已提交的目标写入结束，再报告 failed receive 交由 vLLM 重算；
不会把部分 ticket 的成功前缀直接当作该请求已经恢复成功。

接口实现入口：[prefix facade](../../fragpack/integrations/lmcache/connector.py)、
[transfer worker](../../fragpack/integrations/lmcache/transfer.py)、
[RPC client](../../fragpack/integrations/lmcache/client.py)、
[L1 公共接口适配](../../fragpack/integrations/lmcache/storage.py)。

---

## 配置边界

下表的 attention 环境变量属于 HBM `CUSTOM` 路线；原生路线通过
`FragPackNativeConnector` 配置，具体字段见[开发说明](../development.md)。

| 配置 | 当前含义 |
|------|------|
| `VLLM_PLUGINS=fragpack` | 选择本库 attention plugin |
| `VLLM_USE_V2_MODEL_RUNNER=1` | 选择固定版本 V2 runner；兼容检查要求 `vllm==0.30.0+cu129` |
| `--attention-backend CUSTOM` / `--block-size 128` | 选择 FragPack backend 与对应 block 几何 |
| `FRAGPACK_TABLES` | 指向校准后的 `.npz` 码表；加载时按 TP rank 选择 KV heads |
| `FRAGPACK_OVERFLOW` | raw overflow **记录数**，不是字节数；默认 8,192 |
| `FRAGPACK_PREFILL=gather` | 默认 gather 到 BF16 workspace，再运行 FA3 |
| `FRAGPACK_PREFILL=fused` | 实验性融合 prefill / split-KV；不分配 BF16 prefill workspace |
| `FRAGPACK_WRITE=staged` | 默认写入路径 |
| `FRAGPACK_WRITE=direct` | 对符合完整连续覆盖条件的新块跳过 staging 搬运；目前只允许与 `gather` 搭配 |
| LMCache `--server-module` | 加载 `fragpack.integrations.lmcache.server` 的 factory；不是 vLLM connector 配置 |
| `BlockTransfer(..., num_slots=2)` | 每个 transfer 实例默认两个全层 blob 槽；GPU 与 pinned host 各按最坏 raw blob 大小预留 |

HBM attention 路径在 worker 初始化时选定。当前要求 BF16 KV、SM90、head dimension
128、GQA group 1/2/4/8，不支持 speculative decoding。TP head 切片的实现检查不等于
多卡服务或外部缓存跨 rank 一致性已验证。配置命令与实测范围分别见
[开发文档](../development.md)、[LMCache 接入运行记录](../lmcache-plugin-056.md)。
