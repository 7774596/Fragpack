# 编码前的外部对象存在性查询

`probe_many(keys)` 为独立 codec 的保存路径提供逐 key 的 CPU L1 存在性提示。
当前 native 传输层在已保留 GPU 源页引用的前提下、提交编码之前，跳过已有外部副本的对象。
查询本身不持有 GPU 页；上层仍须在整个预查和源读取阶段保护这些页。
它不改变 FPKB 表示、缓存淘汰策略或任何 attention 接口。

## 接口与开销

- `BlobClient.probe_many(Sequence[BlockKey]) -> tuple[bool, ...]` 按输入顺序返回
  每个对象的存在状态；遇到缺失仍继续查询，重复 key 保持重复结果位置。
- `RemoteBlobClient` 把长列表切分为每次最多 `MAX_PREFIX_KEYS`（32）个 key 的
  `probe` RPC。请求和响应都没有对象 payload；响应的 `present` 必须是等长的
  JSON bool 列表。空列表不发 RPC，但仍检查客户端没有处于失败状态。
- server 和 `LMCacheL1Store` 的单批上限都是 32。首版在一批 RPC 内逐 key
  使用已验证的 stock 0.5.6 `submit_prefetch_task(skip_l2=True)` 契约，立即释放
  每次成功查询的读引用。它减少 RPC 次数，但尚未把底层 manager 查询合为一个任务。
  查询不读取或复制 FPKB payload；不触发 L2 I/O。
- 操作可能阻塞，应在后台/调度侧异步任务运行，不能放到每层 attention 路径。
  它不持有 GPU 页、不提交 CUDA 工作，也不延长外部对象的驻留时间。

## 正确性边界

结果只反映各次查询时的存在性，不是跨批次一致快照，也不是租约。查询结束后对象
可以被正常淘汰；未来恢复仍必须 `get`、校验对象并等待 GPU 解码成功。存在性本身
不验证任意存储字节的语义正确性，不能用来宣布请求已经恢复。

不同 namespace 在查询之前拒绝。RPC 超时或 manager 查询失败按现有规则停止
不确定通道，异常不转成 `False` 或成功命中。已取得但无法确认释放的读引用由
共享 manager 的关闭负责最终清理，不能按 key 删除数据或假定调用已取消。

恢复继续使用 `lookup_prefix` 的连续前缀语义；`probe_many` 只服务于保存去重，
不能据其离散命中结果跳过中间缺失 token 的计算。
