# Engram MCP 集成测试计划

## 测试环境

- **被测对象**: engram MCP server (release build)
- **二进制路径**: `target/release/engram`
- **传输协议**: JSON-RPC over stdio
- **测试数据库**: 每轮测试前清理 `~/.engram/memory.db`

## 测试策略

按照 MCP 协议层 → 写入工具 → 读取工具 → 端到端链路 → 边界条件的顺序，
逐层验证所有 14 个工具的行为正确性。

---

## 1. 协议层测试

### 1.1 MCP 初始化握手
- **输入**: `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}`
- **预期**: 返回 `result.protocolVersion == "2024-11-05"`, `result.serverInfo.name == "engram"`, `result.capabilities.tools` 存在

### 1.2 tools/list 工具列表
- **输入**: `{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`
- **预期**: 返回 14 个工具，每个工具包含 `name`、`description`、`input_schema`
- **工具名称**: `search_memory`, `related_files`, `timeline`, `recent_failures`, `architectural_decisions`, `create_episodic`, `create_decision`, `create_failure`, `create_procedural`, `ingest_commits`, `delete_episodic`, `delete_decision`, `delete_failure`, `delete_procedural`
- **验证**: 每个工具的 `input_schema.required` 均包含 `project_id`

### 1.3 未知方法错误
- **输入**: `{"jsonrpc":"2.0","id":3,"method":"nonexistent/method","params":{}}`
- **预期**: `error.code == -32601`, `error.message` 包含 "Method not found"

### 1.4 畸形 JSON 解析错误
- **输入**: `{bad json}`
- **预期**: `error.code == -32700`, `error.message` 包含 "Parse error"

### 1.5 缺少必填参数
- **输入**: `tools/call` `search_memory` 只传 `project_id` 不传 `query`
- **预期**: `error.code == -32602`, `error.message` 包含 "missing field `query`"

### 1.6 未知工具
- **输入**: `tools/call` `name: "nonexistent_tool"`
- **预期**: `error.code == -32603`, `error.message` 包含 "Unknown tool"

### 1.7 Notification 静默跳过
- **输入**: `{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}`（无 id 字段）
- **预期**: 无响应输出（stdout 无内容）

---

## 2. 写入工具测试

### 2.1 create_episodic
- **用例 A - 基本创建**:
  - 输入: `project_id="test-project"`, `session_id="sess-001"`, `summary="修复 OAuth bug"`, `content="详细描述"`, `files_touched=["src/auth.rs"]`, `tags=["auth"]`, `importance=0.8`
  - 预期: `status == "created"`, `id` 为 UUID 格式, `created_at` 为 Unix 时间戳
  - 验证: 记录返回的 `id` 用于后续测试

- **用例 B - 默认值**:
  - 输入: 只传必填字段 (`project_id`, `session_id`, `summary`, `content`)
  - 预期: `importance` 默认 0.5, `files_touched` 为空数组, `tags` 为空数组

- **用例 C - 中文内容**:
  - 输入: `summary="修复了用户登录认证失败的问题"`, `tags=["认证","bugfix"]`
  - 预期: 正常创建，不报错

### 2.2 create_decision
- **用例 A - 基本创建**:
  - 输入: 全部必填字段 + `related_files=["src/arch.rs"]`, `tags=["architecture"]`
  - 预期: `status == "created"`

- **用例 B - 最少参数**:
  - 输入: 只传 `project_id`, `title`, `context`, `rationale`, `tradeoffs`
  - 预期: 正常创建

### 2.3 create_failure
- **用例 A - 基本创建**:
  - 输入: 全部必填字段 + `severity=4`, `tags=["database"]`
  - 预期: `status == "created"`, `severity == 4`

- **用例 B - severity 下界校验**:
  - 输入: `severity=0`
  - 预期: `error.code == -32603`, `error.message` 包含 "severity must be between 1 and 5"

- **用例 C - severity 上界校验**:
  - 输入: `severity=6`
  - 预期: `error.code == -32603`

- **用例 D - severity 边界值**:
  - 输入: `severity=1` 和 `severity=5`
  - 预期: 均正常创建

- **用例 E - 默认 severity**:
  - 输入: 不传 severity
  - 预期: 默认值 3

### 2.4 create_procedural
- **用例 A - 基本创建**:
  - 输入: `workflow_name="部署流程"`, `steps=["测试","构建","部署"]`, `related_tools=["docker"]`
  - 预期: `status == "created"`

- **用例 B - 空步骤列表**:
  - 输入: `steps=[]`
  - 预期: 正常创建（schema 不强制 steps 非空）

### 2.5 ingest_commits
- **用例 A - 从真实仓库摄取**:
  - 输入: `repo_path="."`, `count=5`
  - 预期: `ingested >= 1`, `memories` 数组非空，每个元素包含 `id`、`summary`、`files_touched`

- **用例 B - 不存在的路径**:
  - 输入: `repo_path="/nonexistent/path"`
  - 预期: `error` 非空

---

## 3. 读取工具测试

### 前置条件
先执行写入工具创建以下测试数据（项目 ID: `integration-test`）:
1. episodic: summary="修复 OAuth token 刷新循环 bug", files_touched=["src/auth/token.rs","src/cache/mod.rs"], tags=["auth","oauth"]
2. decision: title="使用 SQLite FTS5 替代 Elasticsearch", related_files=["src/storage/repository.rs"], tags=["architecture","search"]
3. failure: incident="FTS5 虚拟表同步失败导致搜索结果不一致", tags=["database","fts5"], severity=4
4. procedural: workflow_name="发布流程", steps=["cargo test","cargo build --release"], related_tools=["cargo"], tags=["deploy"]

### 3.1 search_memory
- **用例 A - 英文关键词搜索**:
  - 输入: `query="OAuth token"`
  - 预期: 返回 episodic 记录, `total >= 1`, `results[0].memory_type == "episodic"`

- **用例 B - 中文关键词搜索**:
  - 输入: `query="刷新循环"`
  - 预期: 返回 episodic 记录

- **用例 C - 中文单字搜索**:
  - 输入: `query="同步"`
  - 预期: 返回 failure 记录（FTS5 CJK 预处理生效）

- **用例 D - 按类型过滤**:
  - 输入: `query="FTS5"`, `memory_type="decision"`
  - 预期: 只返回 decision 类型结果

- **用例 E - 按类型过滤 failure**:
  - 输入: `query="FTS5"`, `memory_type="failure"`
  - 预期: 只返回 failure 类型结果

- **用例 F - 按标签搜索**:
  - 输入: `query="database"`
  - 预期: 返回 failure 记录（tags 已纳入 FTS5 索引）

- **用例 G - 按标签搜索 architecture**:
  - 输入: `query="architecture"`
  - 预期: 返回 decision 记录

- **用例 H - 跨项目隔离**:
  - 输入: `project_id="other-project"`, `query="OAuth"`
  - 预期: `total == 0`

- **用例 I - 无匹配结果**:
  - 输入: `query="zzz_nonexistent_keyword"`
  - 预期: `total == 0`, `results` 为空数组

- **用例 J - limit 限制**:
  - 输入: `limit=1`
  - 预期: `results.length <= 1`

- **用例 K - BM25 分数非零**:
  - 输入: 任意匹配查询
  - 预期: `relevance_score > 0.0`（真实 BM25 分数生效）

### 3.2 related_files
- **用例 A - 存在的文件**:
  - 输入: `file="src/auth/token.rs"`
  - 预期: `entities[0].id` 非空, `entities[0].type == "File"`, `entities[0].relations` 包含 `type: "Touches"` 的关系

- **用例 B - 不存在的文件**:
  - 输入: `file="nonexistent/file.rs"`
  - 预期: `entities[0].id` 为空, `relations` 为空数组

- **用例 C - decision 关联文件**:
  - 输入: `file="src/storage/repository.rs"`
  - 预期: `relations` 包含 `type: "References"` 的关系

### 3.3 timeline
- **用例 A - 当日时间线**:
  - 输入: `days=1`
  - 预期: `events` 数组非空, 包含当日日期和 `episodic_count >= 1`

- **用例 B - 7 天时间线**:
  - 输入: `days=7`（默认值）
  - 预期: 包含当日时间线事件

- **用例 C - 无数据项目**:
  - 输入: `project_id="empty-project"`, `days=7`
  - 预期: `events` 为空数组

### 3.4 recent_failures
- **用例 A - 无过滤条件** (Bug 1 回归):
  - 输入: 只传 `project_id`
  - 预期: 返回 failure 列表, 不崩溃

- **用例 B - 带过滤条件**:
  - 输入: `service="FTS5"`
  - 预期: 返回匹配的 failure 列表

- **用例 C - limit 生效**:
  - 输入: `limit=1`
  - 预期: `failures.length <= 1`

### 3.5 architectural_decisions
- **用例 A - 无过滤条件** (Bug 1 回归):
  - 输入: 只传 `project_id`
  - 预期: 返回 decision 列表, 不崩溃

- **用例 B - 带 topic 过滤**:
  - 输入: `topic="SQLite"`
  - 预期: 返回匹配的 decision

---

## 4. 删除工具测试

### 4.1 delete_episodic
- **用例 A - 正常删除**:
  - 步骤: 1) create_episodic 记录 ID  2) delete_episodic 传入该 ID
  - 预期: `status == "deleted"`
  - 验证: search_memory 搜索该记录的内容，`total == 0`

- **用例 B - 删除不存在的记录**:
  - 输入: `id="nonexistent-uuid"`
  - 预期: `status == "not_found"`

### 4.2 delete_decision
- **用例 A - 正常删除**:
  - 步骤: 1) create_decision  2) delete_decision
  - 预期: `status == "deleted"`

### 4.3 delete_failure
- **用例 A - 正常删除**:
  - 步骤: 1) create_failure  2) delete_failure
  - 预期: `status == "deleted"`

### 4.4 delete_procedural
- **用例 A - 正常删除**:
  - 步骤: 1) create_procedural  2) delete_procedural
  - 预期: `status == "deleted"`

---

## 5. 端到端链路测试

### 5.1 写入→搜索→删除→验证 完整链路
1. `create_episodic` 创建记录，获取 ID
2. `search_memory` 搜索，确认结果包含该记录
3. `delete_episodic` 删除该记录
4. `search_memory` 再次搜索，确认 `total == 0`

### 5.2 写入→related_files 图关系验证
1. `create_episodic` 传入 `files_touched=["src/graph_test.rs"]`
2. `related_files` 查询 `"src/graph_test.rs"`
3. 验证: 实体存在，类型为 File，关系类型为 Touches

### 5.3 写入→tags 搜索验证
1. `create_decision` 传入 `tags=["unique-tag-test-xyz"]`
2. `search_memory` 搜索 `"unique-tag-test-xyz"`
3. 验证: 能通过 tag 找到该 decision

### 5.4 ingest_commits→搜索 验证
1. `ingest_commits` 从 engram 仓库摄取 commit
2. `search_memory` 搜索 commit message 中的关键词（如 "engram"）
3. 验证: 搜索结果包含摄取的记录

### 5.5 中文写入→中文搜索 完整链路
1. `create_failure` 传入 `incident="数据库连接池耗尽导致服务不可用"`, `tags=["数据库","连接池"]`
2. `search_memory` 搜索 `"连接池耗尽"`
3. `search_memory` 搜索 `"数据库"`
4. 验证: 两个查询都能找到该记录

### 5.6 多类型混合搜索
1. 创建包含关键词 "Redis" 的 episodic、decision、failure 各一条
2. `search_memory` 搜索 `"Redis"`（不指定类型）
3. 验证: 结果包含 3 种类型，按 `relevance_score` 降序排列

---

## 6. 边界条件与健壮性测试

### 6.1 空字符串处理
- `search_memory` `query=""` — 预期: 错误或空结果
- `create_episodic` `summary=""` — 预期: 正常创建（schema 不限制非空）

### 6.2 超长字符串
- `create_episodic` `content` 传入 10000 字符的文本
- 预期: 正常创建，不截断

### 6.3 特殊字符
- `search_memory` `query` 传入 `test"quote` 或 `test'quote`
- 预期: 不崩溃（SQL 注入防护）

### 6.4 并发写入同项目
- 快速连续创建 10 条 episodic memory
- 验证: 全部创建成功，`search_memory` 能找到 10 条

### 6.5 数据库迁移兼容性
- 前置: 删除 `~/.engram/memory.db`
- 启动 MCP server（会自动 `initialize_schema` + `migrate_fts5_add_tags`）
- 验证: 正常启动，无报错
- 写入→搜索验证正常

### 6.6 重复启动
- 连续启动 server 两次
- 验证: 第二次启动不报错（幂等初始化）

---

## 7. 性能基准（可选）

### 7.1 写入延迟
- 单条 `create_episodic` 延迟 < 20ms
- 单条 `create_episodic` + 3 个 files_touched 延迟 < 30ms

### 7.2 搜索延迟
- `search_memory` 在 100 条数据中搜索 < 150ms

### 7.3 批量写入
- 连续写入 50 条 episodic memories
- 验证: 全部成功

---

## 测试执行命令模板

```bash
ENGRAM=target/release/engram

# 通用请求模板
echo '{"jsonrpc":"2.0","id":<ID>,"method":"tools/call","params":{"name":"<TOOL>","arguments":{<ARGS>}}}' \
  | RUST_LOG=off $ENGRAM 2>/dev/null

# 清理测试数据
rm -f ~/.engram/memory.db ~/.engram/memory.db-wal ~/.engram/memory.db-shm
```
