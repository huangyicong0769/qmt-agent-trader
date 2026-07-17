# Chat Run 生命周期

Chat Agent 的后台 Run 由应用级 `ChatRunManager` 持有，不属于 NiceGUI 页面、浏览器 WebSocket 或 SSE 请求。页面和 SSE 客户端只通过带序号的订阅读取 snapshot、重放事件和实时事件；断线、刷新或标签页关闭只会移除订阅，不会取消 Run。

Run 的状态转换由 manager 集中管理：`PENDING -> RUNNING -> COMPLETED/FAILED`，显式停止走 `RUNNING -> CANCELLING -> CANCELLED`。同步工具调用无法被 Python 安全地强制杀死，因此取消请求在工具执行期间保持 `CANCELLING`，工具返回后由 worker 观察令牌并结束后续模型轮次。

运行派生的聊天消息由 manager 使用 repository 的原子 `update` 写入，并以 `run_id + event_sequence` 去重；metadata 同时保留 `event_type`、工具字段和阶段信息。token delta 只广播和累积草稿，不逐条写入聊天存储。`EventBus` 仅接收 manager 标准化后的一次广播，不负责 Run 状态或 UI 生命周期。

`RunEvent.terminal` 是唯一的终态判断来源：`done`、`cancelled` 和非 fallback 的 `error` 为终态；fallback `error` 只是诊断事件，后续 `final_message` 和 `done` 仍会继续发送。订阅注册、snapshot/replay 捕获和事件 append/broadcast 使用同一原子边界，锁顺序固定为 `manager._lock -> run.event_lock`。

snapshot 会携带 `accumulated_draft_through_sequence`。重连时 manager 以 repository 的持久化 cursor 作为 `after_sequence`，过滤已经被 snapshot 草稿覆盖的 token；早期 token 即使被有限历史裁剪，也不会重复拼接。同步工具返回后先广播并持久化 `tool_done`，再确认取消并停止下一轮模型请求。

页面只为当前 session 建立订阅；successor watcher 不会抢占另一个当前 session。待发送消息按 session 保存，token 不启动 persistence thread，sidebar 只在消息预览或终态改变时刷新，草稿更新以约 50ms 批次节流。

取消令牌是每个 Run 独立的线程安全对象。它会唤醒合作式检查，并在模型 stream 支持 `close()` 时触发关闭；关闭只能缩短模型等待，不能安全强制终止正在运行的同步 Python 工具。持久化失败会请求同一令牌取消，并在 orchestrator worker 完全退出后才释放 session 的运行所有权。

当前 Run 状态和后台 task 只保存在服务进程内存中。支持的是同一服务进程内的断线、刷新和标签页关闭恢复；服务进程重启后的运行恢复不在本次范围内，已持久化的聊天消息仍然保留。
