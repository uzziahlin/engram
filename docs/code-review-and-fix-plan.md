# Engram 代码库审查报告与优化修复计划

**审查日期：** 2026-06-13
**审查范围：** 28 个 Rust 源文件，5,462 行代码
**发现问题：** 4 个 CRITICAL / 13 个 HIGH / 14 个 MEDIUM / 8 个 LOW

---

## 一、审查报告

### 🔴 CRITICAL（必须修复）

#### C1. Reranker 评分公式中 `graph_weight` 被计算但从未参与评分
**文件：** `src/retrieval/reranker.rs:51-53`
**置信度：** HIGH

Planner 为不同意图精心计算了 `graph_weight`（Debugging=0.8, Incident=0.9 等），但实际评分公式完全忽略了这个值：

```rust
// 当前：graph_weight 从未参与计算
let final_score = base_relevance * 0.5
    + recency_normalized * plan.recency_weight
    + base_relevance * plan.importance_weight;  // ← 缺少 graph_weight
```

**影响：** 系统的核心价值主张——基于意图的权重调整——是失效的。

#### C2. `ingest_commits` 中 `skipped_duplicates` 计算存在 usize 下溢
**文件：** `src/mcp/server.rs:588`
**置信度：** HIGH

`input.count - ingested.len()` 当 `ingested` 数量超过请求的 count 时，`usize` 下溢会导致 debug panic 或 release 模式下的荒谬大值。

#### C3. CLI `ingest` 命令完全缺少去重逻辑
**文件：** `src/cli/commands.rs:287-315`

MCP server 的 `ingest_commits` 有完整的去重流程（`get_ingested_commits` + filter），但 CLI 的 `ingest` 命令直接无条件写入。用户反复执行会产生大量重复记忆，破坏数据完整性。

#### C4. `remove_entity` / `create_entity` / `create_relation` 缺少事务保护
**文件：** `src/storage/repository.rs:746-808`

两条 DELETE 不在事务中——如果删 relations 成功但删 entity 失败，数据库将处于不一致状态。

---

### 🟠 HIGH（应当修复）

#### H1. `failure_memories` 缺少 `project_id` 索引
**文件：** `src/storage/repository.rs:170-178`

其他三个表都有 `(project_id, created_at DESC)` 复合索引，只有 failure 表遗漏，查询会全表扫描。

#### H2. `ensure_linked_entities` 存在 N+1 查询模式
**文件：** `src/storage/repository.rs:313-335`

N 个文件 = 3N+1 条 SQL，大型项目会成为瓶颈。

#### H3. 缺少 `PRAGMA busy_timeout`
**文件：** `src/storage/repository.rs:42-44`

WAL 模式下并发写操作会直接收到 SQLITE_BUSY 错误。

#### H4. `fts_integrity_check` 用 `format!` 拼接 SQL
**文件：** `src/storage/repository.rs:897`

虽然当前表名硬编码，但 `format!` 拼接 SQL 是危险模式。

#### H5. `.lock().unwrap()` 在所有业务方法中
**文件：** `src/mcp/server.rs:251,312,...`

Mutex poisoned 时直接 panic，长期运行的服务会崩溃。

#### H6. `related_files` 同时持有三个 Mutex
**文件：** `src/mcp/server.rs:310-320`

潜在死锁风险，锁获取/释放次数过多。

#### H7. JSON-RPC 解析错误时 `id` 被丢弃
**文件：** `src/mcp/server.rs:827`

违反 JSON-RPC 2.0 规范。

#### H8. `severity`/`importance` 无范围校验
**文件：** `src/mcp/server.rs:122-154`

文档声明 1-5 / 0-1，但实际接受任意值。

#### H9. `importance_weight` 语义错误
**文件：** `src/retrieval/reranker.rs:53`

用 `base_relevance` 而非实际 importance 分数参与计算，本质上只是给 relevance 增加了一个额外系数。

#### H10. `search_all` 与 `search_by_type` 大量重复代码
**文件：** `src/retrieval/bm25.rs:38-157`

~80 行代码约 60 行是复制粘贴。

#### H11. `"deploy"`/`"refactor"` 关键词跨意图组重复
**文件：** `src/retrieval/intent_classifier.rs:44,61`

导致分类二义性。

#### H12. `Config::load()` 静默吞掉解析错误
**文件：** `src/config.rs:169-178`

用户精心配置但拼写错误时，系统静默使用默认值。

#### H13. `estimate_importance` 关键词覆盖问题
**文件：** `src/git_integration/listener.rs:119-139`

`"fix: update docs"` 重要性被 docs 的 0.2 覆盖了 fix 的 0.7。

---

### 🟡 MEDIUM（建议修复）

| # | 模块 | 问题 | 文件 |
|---|------|------|------|
| M1 | Storage | `preprocess_cjk` 对纯 ASCII 文本也遍历所有字符检查 CJK | `repository.rs:190-213` |
| M2 | Storage | Entity/Relation 读取方法大量重复的 row mapping 代码 | `repository.rs:765-887` |
| M3 | Storage | `row_get_json!` 宏在 JSON 反序列化失败时静默返回默认值 | `macros.rs:6-8` |
| M4 | Storage | `EntityType::from_str` 错误转换语义不正确 | `repository.rs:773-774` |
| M5 | Storage | `get_ingested_commits` 应用层展开 JSON，应使用 `json_each` | `repository.rs:666-680` |
| M6 | Retrieval | BM25 归一化使用硬编码 sigmoid 参数 | `bm25.rs:33` |
| M7 | Retrieval | Planner 中 `sources` 去重使用 `format!("{a:?}")` 排序 | `planner.rs:117` |
| M8 | Retrieval | `detect_content_type` 对空字符串返回 `Code` | `composer.rs:90-96` |
| M9 | Retrieval | `search_memory` 中 Intent 分类结果仅用于 reranking，未影响检索策略 | `server.rs:253-258` |
| M10 | MCP | Schema 声明 `default: 5` 但代码 `default_limit()` 返回 10 | `server.rs:699` |
| M11 | MCP | `ingest_commits` 创建的记忆没有建立 graph 实体关系 | `server.rs:555-591` |
| M12 | Config | 缺少字段验证（`memory_budget_percent` 可为 255） | `config.rs:83-86` |
| M13 | CLI | 手写参数解析，`--flag` 值可能被误读为另一个 flag | `commands.rs:27-58` |
| M14 | Consolidation | `DefaultHasher` 不保证跨版本一致性 | `engine.rs:24` |

---

### 🟢 LOW（可选改进）

| # | 模块 | 问题 |
|---|------|------|
| L1 | Storage | FTS tags 列存储 JSON 数组格式，产生噪声 token |
| L2 | Storage | 单一 `Connection` 无法支持并发读（WAL 模式下本可以） |
| L3 | Storage | macro 中 `score_col_idx` 硬编码列索引，维护隐患 |
| L4 | MCP | `JsonRpcError` 缺少可选的 `data` 字段 |
| L5 | MCP | `line.trim().to_string()` 不必要的字符串分配 |
| L6 | Retrieval | `BM25Retriever` 零大小类型设计不一致 |
| L7 | CLI | `timeline` 命令绕过 Repository 层直接写 SQL |
| L8 | Models | 四种 Memory model 字段全 pub，缺少构造函数和验证 |

---

### ✅ 做得好的地方

1. **FTS5 双写策略** — macro 生成的 CRUD 在事务内同步更新主表和 FTS 表，保证数据一致性
2. **Trait 解耦设计** — `MemoryToolProvider` 将业务逻辑与 MCP 传输层完全分离
3. **`dispatch_tool!` 宏** — 统一了参数反序列化和错误处理
4. **CJK 分词处理** — `preprocess_cjk` 是生产环境经常被忽略的重要细节
5. **增量图加载** — `GraphEngine` 的 `loaded_projects` 避免重复加载
6. **日志与 MCP 分离** — tracing 写 stderr 不与 JSON-RPC 的 stdout 冲突
7. **测试覆盖** — 每个模块都有针对核心场景和边界条件的单元测试

---

## 二、修复计划

按 4 批次修复，每批次完成后运行 `cargo build && cargo test && cargo clippy -- -D warnings` 验证。

### 第一批：关键修复（CRITICAL + 紧急 HIGH）

预计工作量：~40 分钟。

#### 1.1 Storage: 添加缺失的 PRAGMA 和索引

**文件：** `src/storage/repository.rs`

**1.1a** 在 `initialize_schema()` 中 `PRAGMA journal_mode = WAL;` 后追加：
```rust
self.conn.execute_batch("PRAGMA busy_timeout = 5000;")?;
self.conn.execute_batch("PRAGMA synchronous = NORMAL;")?;
```

**1.1b** 在索引块末尾添加：
```sql
CREATE INDEX IF NOT EXISTS idx_failure_project_time ON failure_memories(project_id, created_at DESC);
```

#### 1.2 Storage: 为 `remove_entity`/`create_entity`/`create_relation` 添加事务

**文件：** `src/storage/repository.rs`

用 `unchecked_transaction()` 包裹每个方法内的多条 SQL 语句。

#### 1.3 MCP Server: 修复 `skipped_duplicates` 计算错误

**文件：** `src/mcp/server.rs:585-590`

```rust
let total_before_dedup = memories.len();
let skipped = total_before_dedup - new_memories.len();
// 使用 skipped 替代 input.count - ingested.len()
```

#### 1.4 CLI: 为 `ingest` 命令添加去重逻辑

**文件：** `src/cli/commands.rs:287-315`

对齐 MCP server 的去重流程：调用 `repo.get_ingested_commits()` 后 filter。

#### 1.5 MCP Server: 将 `.lock().unwrap()` 替换为安全的错误处理

**文件：** `src/mcp/server.rs`

添加 `lock_repo()` / `lock_graph()` / `lock_loaded()` 辅助方法，全局替换所有 `.lock().unwrap()`。

#### 1.6 Retrieval: 修复 Reranker 评分公式

**文件：** `src/retrieval/reranker.rs:51-53`

采用静态 `type_boost` 方案，为不同 memory_type 提供固定的图权重加成：
```rust
let type_boost: f32 = match result.memory_type.as_str() {
    "failure" => 0.9,
    "decision" => 0.7,
    "episodic" => 0.5,
    "procedural" => 0.3,
    _ => 0.4,
};

let final_score = base_relevance * 0.4
    + recency_normalized * plan.recency_weight
    + base_relevance * plan.importance_weight
    + type_boost * plan.graph_weight;
```

---

### 第二批：核心逻辑修复（剩余 HIGH）

预计工作量：~50 分钟。

#### 2.1 Git: 修复 `estimate_importance` 的关键词覆盖问题

**文件：** `src/git_integration/listener.rs:119-139`

将顺序覆盖改为取最大值策略。

#### 2.2 MCP Server: 修复 JSON-RPC 解析错误时 `id` 丢失

**文件：** `src/mcp/server.rs:816-837`

缓存第一次解析的 `id`，在第二次解析失败时复用。

#### 2.3 MCP Server: 添加 `severity` 和 `importance` 范围校验

**文件：** `src/mcp/server.rs`

在 `create_failure` 和 `create_episodic` 入口添加范围检查。

#### 2.4 Config: 修复 `load()` 静默吞掉解析错误

**文件：** `src/config.rs:163-179`

区分「文件不存在」和「文件存在但解析失败」。

#### 2.5 Config: 添加字段验证

**文件：** `src/config.rs`

新增 `validate()` 方法，校验 `memory_budget_percent`、`default_limit`、`max_nodes` 等字段。

---

### 第三批：代码质量优化（MEDIUM）

预计工作量：~60 分钟。

| # | 修改 | 文件 |
|---|------|------|
| 3.1 | BM25Retriever 消除重复代码，提取辅助方法 | `retrieval/bm25.rs` |
| 3.2 | IntentClassifier 修复关键词冲突（移除 deploy/refactor 重复） | `retrieval/intent_classifier.rs` |
| 3.3 | `row_get_json!` 宏添加失败日志 | `storage/macros.rs` |
| 3.4 | `MemorySource` 派生 `Ord`，简化 Planner 去重 | `retrieval/planner.rs` |
| 3.5 | Reranker `apply_fallback_limits` 简化双重 truncate | `retrieval/reranker.rs` |
| 3.6 | `detect_content_type` 修复空字符串问题 | `context/composer.rs` |
| 3.7 | 统一 Entity/Relation row mapping 代码 | `storage/repository.rs` |
| 3.8 | Schema default 与代码 default 统一 | `mcp/server.rs` |

---

### 第四批：锦上添花（LOW）

预计工作量：~45 分钟。可视时间安排选择性执行。

| # | 修改 | 文件 |
|---|------|------|
| 4.1 | `JsonRpcError` 添加可选 `data` 字段 | `mcp/server.rs` |
| 4.2 | `preprocess_cjk` 优化容量预分配 | `storage/repository.rs` |
| 4.3 | BM25 归一化 sigmoid 参数可配置化 | `retrieval/bm25.rs` |
| 4.4 | CLI `require_str` 添加参数值验证 | `cli/commands.rs` |
| 4.5 | `ensure_linked_entities` 使用 `RETURNING` 优化 N+1 | `storage/repository.rs` |
| 4.6 | Consolidation 使用确定性哈希 | `consolidation/engine.rs` |

---

### 修改文件清单

| 文件 | 批次 | 修改类型 |
|------|------|----------|
| `src/storage/repository.rs` | 1, 3, 4 | PRAGMA/索引/事务、辅助函数、N+1优化 |
| `src/storage/macros.rs` | 3 | 失败日志 |
| `src/mcp/server.rs` | 1, 2, 3 | 计算错误/lock/JSON-RPC/校验/schema |
| `src/cli/commands.rs` | 1, 4 | 去重逻辑、参数验证 |
| `src/retrieval/reranker.rs` | 1, 3 | 评分公式、truncate简化 |
| `src/retrieval/bm25.rs` | 3, 4 | 辅助方法、常量提取 |
| `src/retrieval/planner.rs` | 3 | `MemorySource` Ord |
| `src/retrieval/intent_classifier.rs` | 3 | 关键词冲突 |
| `src/context/composer.rs` | 3 | 空字符串判定 |
| `src/config.rs` | 2 | 错误处理、验证 |
| `src/git_integration/listener.rs` | 2 | importance 取最大值 |
| `src/consolidation/engine.rs` | 4 | 哈希算法 |

### 验证策略

每批次完成后执行：
1. `cargo build` — 确保编译通过
2. `cargo test` — 确保所有现有测试通过
3. `cargo clippy -- -D warnings` — 确保无 lint 警告

最终验收：
- 手动测试 MCP server：`echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | cargo run`
- 手动测试 CLI ingest 去重：连续两次执行 `cargo run -- ingest --project test --repo .`
- 验证 Config 错误报告：创建格式错误的 `config.toml`，确认启动时报错
