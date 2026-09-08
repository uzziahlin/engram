# Engram - Claude Code 接入指南

## 1. 构建

```bash
cd /path/to/engram
cargo build --release
```

## 2. 配置 MCP Server

编辑 `~/.claude/settings.json`，添加 engram 作为 MCP server：

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram/target/release/engram",
      "args": [],
      "env": {}
    }
  }
}
```

## 3. 可选：配置文件

默认数据库路径 `~/.engram/memory.db`。可创建 `~/.engram/config.toml` 自定义：

```toml
[storage]
database_path = "~/.engram/memory.db"
# query_log 保留天数，`engram maintain` 会清理超期行（默认 90）
query_log_retention_days = 90

[retrieval]
# 省略 limit 时工具使用的默认值
default_limit = 10
# 意图关键词只调整排序权重，不会缩小检索的记忆类型
intent_routing = true

[context]
memory_budget_percent = 15

[mcp]
# stdio 是唯一传输方式；请求处理线程数（1 = 顺序处理，最安全）
worker_threads = 1
```

## 4. 添加 CLAUDE.md 指令

在项目根目录的 `CLAUDE.md` 中添加以下内容，指导 Claude Code 自动使用记忆工具：

```markdown
## Memory System (engram)

engram 是一个工程记忆系统，通过 MCP 接入。在以下场景中使用记忆工具：

### 自动记忆写入
- 完成调试/修复后：调用 `create_episodic` 记录过程、根因和涉及的文件
- 讨论并确定架构方案后：调用 `create_decision` 记录决策、理由和权衡
- 修复 bug 后：调用 `create_failure` 记录故障、根因、修复方案和预防措施
- 总结工作流程后：调用 `create_procedural` 记录步骤和工具
- 新项目开始时：调用 `ingest_commits` 从 git 历史批量导入记忆

### 自动记忆查询
- 开始调试前：调用 `search_memory` 查找相关历史记忆
- 遇到类似问题时：调用 `recent_failures` 查找历史故障
- 考虑架构变更时：调用 `architectural_decisions` 查找历史决策
- 查看项目时间线：调用 `timeline` 查看近期工程活动

所有工具都需要 `project_id` 参数来隔离不同项目的记忆。
```

## 5. 重启 Claude Code

配置完成后重启 Claude Code，engram MCP server 会自动启动。

## 6. 验证

在 Claude Code 中对话时，你可以测试：

```
请帮我记住：我们选择了 Redis 作为缓存方案，因为需要亚毫秒级延迟。
```

Claude Code 会自动调用 `create_decision` 工具记录这个决策。

```
上次 auth 模块有什么故障记录？
```

Claude Code 会调用 `recent_failures` 工具查找相关故障记忆。

## 7. 自动记忆闭环（推荐）：Claude Code hooks

依赖 agent "自觉" 写记忆不可靠。用 Claude Code 的 hooks 在会话结束时自动蒸馏记忆：

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "~/.engram/bin/engram hook --project my-project"
          }
        ]
      }
    ]
  }
}
```

说明：
- 推荐挂 `SessionEnd`（整个会话结束时触发一次，transcript 完整），stdin 收到含 `transcript_path` 的 JSON。
- `engram hook` 直接读 stdin 的 hook payload，无需 jq/xargs 管道；会从会话 JSONL 提取用户需求、结论和改动文件，蒸馏成一条 episodic 记忆（`--dry-run` 可预览）。
- 挂 `Stop`（每次回复结束触发）也可以：session-import 对同一 session **幂等**——重复导入会原地刷新既有记忆而非堆叠近似重复，记忆随会话增长实时更新。
- 想更省心，可配合 cron 定期跑维护：`engram maintain --apply`（去重 + FTS 修复 + query_log 清理 + 过期报告）。

## 可用工具一览

| 工具 | 类型 | 必填参数 | 说明 |
|------|------|----------|------|
| `search_memory` | 读 | project_id, query | 全文搜索记忆（结果含完整字段，支持 tags/before 过滤） |
| `get_memory` | 读 | project_id, memory_type, id | 按 id 读取一条记忆的完整记录 |
| `related_files` | 读 | project_id, file | 查看文件关联拓扑 |
| `timeline` | 读 | project_id | 项目时间线 |
| `recent_failures` | 读 | project_id | 最近故障记录 |
| `architectural_decisions` | 读 | project_id | 架构决策记录 |
| `create_episodic` | 写 | project_id, session_id, summary, content | 创建任务记忆 |
| `create_decision` | 写 | project_id, title, context, rationale, tradeoffs | 记录架构决策 |
| `create_failure` | 写 | project_id, incident, root_cause, fix, prevention | 记录故障 |
| `create_procedural` | 写 | project_id, workflow_name, steps | 记录工作流 |
| `ingest_commits` | 写 | project_id, repo_path | 从 git 历史自动导入 |
