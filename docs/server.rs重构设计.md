# server.rs 重构设计（路线图 4.1）

> **关联**：`docs/迭代优化路线图.md` §4.1（P2 · server.rs god object）
> **基线**：commit `b116b67`，`src/mcp/server.rs` = 3468 行（其中测试 ~1365 行，占 39%）
> **状态图例**：每个 Phase = 一个独立 PR，可单独 review / merge / 回滚
> **硬约束**：每个 Phase 完成后，lib 190 测试 + e2e 必须全绿（**零行为变更**）
>
> **✅ 完成（2026-06-29）**：5 个 Phase 落地（0 测试外移 / 1 embedding 独立 / 2 传输层独立 / 5 SQL 回收 / 4 NoopProvider 宏化），Phase 3 评审后放弃（避免过度设计）。server.rs 生产代码 2103→**1711 行**（-392），拆出 `embedding_service.rs`(326) + `transport.rs`(134) + `server_tests.rs`(1367)。全程 lib 187 + e2e 4 全绿、双 feature clippy `-D warnings` 通过。注：`MemoryToolProvider` trait 实为 **22** 方法（非早先所记 23）。分支 `refactor/server-decompose`。

---

## 一、背景与目标

`src/mcp/server.rs` 单文件承载 6 类职责，是典型的 god object。核心痛点不是「行数多」，而是**加一个工具 / 改一处功能的边际成本过高**——这是后续所有产品演进（README roadmap 的 HTTP transport、多 agent 共享）的承重墙。

**目标**：把「加一个工具要同步修改的点」从 6 处降到 2 处以内，并让文件按职责拆分到可独立演化的模块。**不追求行数指标，追求职责单一与改动隔离。**

---

## 二、现状剖析

### 2.1 职责地图（行号区间）

| 行号区间 | 职责 | 行数 |
|---|---|---|
| 1–330 | 23 个 Input struct + JsonRpc 协议 struct + `ToolDefinition` + `ReindexReport` | ~330 |
| 336–366 | `trait MemoryToolProvider`（23 方法声明） | ~30 |
| 399–645 | `impl DefaultMemoryProvider`：构造 + **embedding 子系统**（`init_embedder`/`index_embedding`/`reindex_one`/`reindex_embeddings`） | ~246 |
| 646–1303 | `impl MemoryToolProvider`：23 个工具业务逻辑（含 4 个 `create_*`） | ~657 |
| 1304–1354 | 辅助函数（`merge_patch`/`resolve_kinds`/`render_bootstrap_prompt`） | ~50 |
| 1356–1447 | `McpServer` + `NoopProvider`（又 stub 一遍 23 方法） | ~90 |
| 1468–1780 | `list_tools()`：**312 行手写工具 schema** | ~312 |
| 1780–2100 | **传输层**：`run()`（stdio + worker 线程）+ `handle_request()`（JSON-RPC 路由 + prompts + dispatch） | ~320 |
| 2103–3468 | 测试 | ~1365 |

### 2.2「加一个工具」要同步修改的 6 处

> 路线图原说「4 处」，实测为 **6 处**，这是最痛的点。

| # | 位置 | 改动 |
|---|---|---|
| ① | `struct XxxInput`（27–322） | 新增入参 struct |
| ② | `trait MemoryToolProvider`（336–366） | 新增方法声明 |
| ③ | `impl ... for DefaultMemoryProvider`（646–1303） | 新增业务实现 |
| ④ | `NoopProvider`（1371–1447） | 新增 stub 实现 |
| ⑤ | `list_tools()`（1468–1780） | 新增手写 schema |
| ⑥ | `handle_request` 的 match（1942+） | 新增 dispatch 分支 |

### 2.3 关键依赖事实（影响各 Phase 方案）

| 事实 | 位置 | 影响 |
|---|---|---|
| `JsonRpcRequest/Response/Error` 是**私有** struct | 27–52 | 测试构造它们 → Phase 0 用子模块 `#[path]` 访问，零可见性改动 |
| `dispatch_tool!` 宏**已存在** | 56 | dispatch 已半自动化（每工具一行）→ Phase 4 有基础 |
| `embedder` 有**双使用点** | create_* 的 `index_embedding`（写）+ `search_memory` semantic fusion（`680`，读） | Phase 1 的 `EmbeddingService` 须封装读写两条路径 |
| models **无统一 Memory trait** | `embedding_text()` 是 4 个独立 inherent method（`episodic/decision/failure/procedural.rs`） | Phase 3 泛型化需新建 trait → 易过度设计，**标为可选** |
| `consolidate_memories` 走 `ConsolidationEngine` | 1278 | **不写裸 SQL**，Phase 5 无需动它 |
| mcp 层裸 SQL **仅剩 `timeline`** | 795 | Phase 5 只迁这一处；`related_files` 已在 3.4 收回 |
| `list_tools` schema 与 Input struct 是**双事实源** | 1468 vs 27 | 手写 schema 与 serde 字段可能漂移 → Phase 4 治本点 |
| 依赖无 `schemars`，仅 `serde + serde_json` | `Cargo.toml` | Phase 4 schema 自动生成需引入新依赖（决策点） |

---

## 三、设计原则

1. **零行为变更**：所有重构不得改变任何工具的输入/输出/错误语义。190 lib 测试 + e2e 全绿是每个 PR 的硬性验收门槛。
2. **一个 Phase 一个 PR**：拒绝大爆炸。每个 Phase 独立可 review、可 merge、可回滚。
3. **防过度设计**：Phase 3/4 是路线图点名的高危区。**抽象不顺手就放弃、保留重复**——重复 4 处 `create_*` 的成本 < 错误抽象的成本。
4. **先铺安全网再动业务**：Phase 0（测试外移）必须最先做，它是后续所有改动的回归保障。
5. **渐进可见**：每个 Phase 结束都应让文件更短、职责更清，而非「为未来留接口」。

---

## 四、分 Phase 详细设计

### Phase 0 · 测试外移（P0，最先做）　✅ 已完成（2026-06-29）

> **落地**：server.rs 3468→**2101** 行，测试 ~1365 行移入 `src/mcp/server_tests.rs`（`#[cfg(test)] #[path] mod server_tests`，零可见性改动——`pub` 计数 0）。验收：lib 187 + e2e 4 全绿、clippy `-D warnings` 通过、fmt clean、行数守恒（2101+1367=3468）。

- **目标**：把 2103–3468 的 ~1365 行测试移出 `server.rs`，主文件从 3468 → ~2100 行，聚焦业务。这是后续所有 Phase 的安全网。
- **方案**：用 `#[path]` 子模块，**零可见性改动**。
  ```rust
  // server.rs 末尾追加
  #[cfg(test)]
  #[path = "server_tests.rs"]
  mod server_tests;
  ```
  将 2103–3468 内容原样移入 `src/mcp/server_tests.rs`，文件首行加 `use super::*;`。子模块可访问父模块一切私有项（`JsonRpcRequest`、`dispatch_tool!`、私有字段），无需任何 `pub` 改动。
- **涉及文件**：`src/mcp/server.rs`（删测试块）、新建 `src/mcp/server_tests.rs`
- **风险**：极低。唯一注意点：测试块开头的 `mod tests { ... }` 包裹要拆掉（外层 mod 已是 `server_tests`），保留所有 `#[test]`/`#[cfg(feature="semantic")]`/`#[ignore]` 属性原样。
- **验收**：`cargo test` 全绿（190 测试 + semantic ignored 不变）；`server.rs` 行数降到 ~2100；`cargo clippy -D warnings` 通过。

### Phase 1 · embedding 子系统独立（P0）

- **目标**：抽 `src/mcp/embedding_service.rs`，封装 embedder 生命周期 + 向量索引（写）+ semantic fusion（读）。`DefaultMemoryProvider` 不再直接持有 embedder。
- **设计**：
  ```rust
  // src/mcp/embedding_service.rs
  pub struct EmbeddingService {
      repo: MemoryRepository,            // 共享同一 repo（与 provider 同生命周期）
      embedder: Option<Box<dyn EmbeddingProvider>>,
  }
  impl EmbeddingService {
      pub fn new(repo: MemoryRepository, config: &Config) -> Self { /* init_embedder 逻辑迁入 */ }
      pub fn is_active(&self) -> bool { self.embedder.is_some() }

      /// 写路径：create_* 写完记忆后建索引（原 index_embedding）
      #[cfg(feature = "semantic")]
      pub fn index(&self, memory_type: &str, id: &str, project_id: &str, text: &str) { ... }

      /// 读路径：search_memory 的 semantic fusion（原 server.rs:678-719 的 #[cfg(semantic)] 块）
      #[cfg(feature = "semantic")]
      pub fn fuse(&self, repo: &MemoryRepository, query: &str, project_id: &str,
                  bm25: Vec<SearchResult>, cfg: &SemanticConfig) -> Vec<SearchResult> { ... }

      /// reindex 全量回填（原 reindex_embeddings + reindex_one）
      #[cfg(feature = "semantic")]
      pub fn reindex(&self, project: Option<&str>, force: bool, dry_run: bool) -> Result<ReindexReport> { ... }
  }
  ```
  - `ReindexReport`（372）随之迁入本文件。
  - `DefaultMemoryProvider` 持有 `embedding: EmbeddingService`；`create_*` 改调 `self.embedding.index(...)`；`search_memory` 的 fusion 块改调 `self.embedding.fuse(...)`；CLI 的 reindex 入口改调 `provider.embedding.reindex(...)`（注意可见性）。
- **涉及文件**：新建 `src/mcp/embedding_service.rs`；改 `src/mcp/server.rs`（移除 embedding 方法、改调用点）；`src/mcp/mod.rs` 注册新模块
- **关键边界**：`fuse()` 需要 `repo.load_active_embeddings` + `BM25Retriever::fetch_by_ids`——这些依赖保持不变，只是调用主体从 provider 换成 service。`SearchResult` 类型来自 retrieval，确认可见性。
- **风险**：中低。embedder 此前是 provider 字段，迁移后 provider 不再直接感知 embedder 存在——`#[cfg(feature="semantic")]` 的条件编译边界要逐处核对，避免 feature 关闭时编译失败。
- **验收**：`cargo test` + `cargo test --features semantic`（ignored 的真实模型测试不强制跑）全绿；`cargo build`（默认）与 `cargo build --features semantic` 都通过。

### Phase 2 · 传输层独立（P1）

- **目标**：抽 `src/mcp/transport.rs`，封装 stdio I/O 循环 + JSON-RPC 帧解析 + worker 线程池。为 README roadmap 的 HTTP transport 铺路（换 transport 时只改这一个文件）。
- **设计**：
  - `JsonRpcRequest/Response/Error`（27–52）迁入 `transport.rs`（或独立 `protocol.rs`，二选一，omp 择优）。
  - `run()`（1780–1861）迁入 `transport.rs`，通过 trait 或闭包回调调用业务 handler：
    ```rust
    pub trait RequestHandler: Send + Sync {
        fn handle(&self, req: JsonRpcRequest) -> JsonRpcResponse;
    }
    pub fn run_stdio<H: RequestHandler>(handler: Arc<H>, worker_threads: usize) -> Result<()>
    ```
  - `McpServer` 实现 `RequestHandler`，其 `handle` 委托现有 `handle_request`。
  - **`handle_request`（路由 + prompts + dispatch）留在 `server.rs`**——它依赖 `provider` + `list_tools` + prompts，属协议-业务交界，不宜下沉到纯传输层。
- **涉及文件**：新建 `src/mcp/transport.rs`（含 JsonRpc struct + `run_stdio`）；改 `src/mcp/server.rs`（移除 run/JsonRpc、impl RequestHandler）；`mod.rs` 注册
- **风险**：中。mpsc channel + worker 线程 + poisoned-mutex 恢复（1796/1804 的 `into_inner`）是并发关键路径，迁移时**逐行保留语义**。`concurrent_handle_request_is_safe`（3410）测试是并发正确性的兜底，必须保持绿。
- **验收**：`cargo test` 全绿（含并发测试）；手动跑一次 `engram` MCP 交互确认 stdio 收发正常。

### Phase 5 · SQL 回收（P0，顺手）

- **目标**：消除 mcp 层最后一处裸 SQL（`timeline`），让所有业务 SQL 归 repository。
- **方案**：`repository.rs` 新增方法，`server.rs` 的 `timeline()`（790–815）改调它：
  ```rust
  // src/storage/repository.rs
  pub struct TimelineRow { pub day: String, pub count: i64 }
  pub fn timeline(&self, project_id: &str, days: i64, now: i64) -> Result<Vec<TimelineRow>>
  ```
  现有裸 SQL（`SELECT date(created_at,'unixepoch') ...`，795）原样迁入。`server.rs` 的 `timeline()` 只做 `now` 计算 + 调用 + JSON 组装。
- **涉及文件**：`src/storage/repository.rs`（新增方法 + struct）、`src/mcp/server.rs`（改 timeline 调用）
- **注**：`consolidate_memories`（1278）走 `ConsolidationEngine`，**不碰**。
- **风险**：极低。
- **验收**：`timeline` 工具输出与重构前完全一致（`test_jsonrpc_timeline` 测试兜底）；mcp 层 grep `SELECT/INSERT/UPDATE/DELETE` 无业务 SQL 残留。

### Phase 3 · create_* 去重　⊘ 评审后放弃（2026-06-29，避免过度设计）

> **评审结论**：放弃。create_* 骨架虽重复，但核心是构造特定 Memory struct（字段 inherent 差异）；泛型化需新建 trait `CreateInput` + 4 个 Input impl（`into_memory`/`validate`）+ repo 侧泛型 create。`into_memory` 的字段映射本质与现有构造 Memory 相同，只是搬到 trait impl——**不减少复杂度、只移动它并加间接层**。净省 ~60 行却增一整层抽象与理解成本，属典型过度设计。保留 4 个 `create_*` 原样（共 ~145 行，清晰直白）。符合本设计预案「不顺手则放弃」。

- **目标**：4 个 `create_*`（897–1040，共 ~145 行）骨架高度重复，评估泛型化收益。
- **现状**：骨架一致（`lock_repo` + now + uuid + 构造 Memory + `repo.create_xxx` + `index_embedding` + 返回 json），差异仅在：字段映射、校验（importance 0–1 / severity 1–5）、memory_type 字符串、返回额外字段（failure 塞 severity）。
- **方案（若做）**：新建 trait 收敛差异点：
  ```rust
  trait CreateInput: Sized {
      type Mem;
      fn project_id(&self) -> &str;
      fn validate(&self) -> Result<()>;
      fn into_memory(self, id: String, now: i64) -> Self::Mem;
      fn memory_type() -> &'static str;
      fn extra_response_fields(&self) -> serde_json::Map<String, serde_json::Value>;
  }
  // + helper
  fn create_one<I: CreateInput>(&self, input: I) -> Result<Value>
      where I::Mem: /* 提供 repo.create + embedding_text 的 trait，需新建 */
  ```
- **判断**：models **无统一 Memory trait**，泛型化需新建 trait + 4 个 Memory 的 impl + 4 个 Input 的 impl。**省 ~80 行但新增一整层抽象**。**优先级最低；若 trait 抽象不顺手（字段映射散落、校验逻辑难以统一），立即放弃、保留 4 个函数。** 不要为了去重而强行抽象。
- **涉及文件**：`src/mcp/server.rs`、`src/models/*.rs`（若建 trait）
- **风险**：中（过度设计）。
- **验收**：若实施——4 个 create 工具行为不变（现有 create 测试兜底）；代码净减且更易读。

### Phase 4 · 工具注册单一事实源（选项 c）　✅ 已完成（2026-06-29）

> **落地**：选项 (c)——NoopProvider 的 22 个重复 stub 用 `noop_stubs!` 声明宏生成（`$name:ident,$input:ty,$json:tt` 三元组、`;` 分隔、`$json:tt` 捕获 token tree 透传 `json!()`），加工具时从「手写新 fn」→「列表加一行」，减一处同步点。22 个占位 JSON 逐字保留（零行为变更）。选项 (a) schemars 未做（需引新依赖 + 全量 Input struct 改 + schema 还原度风险，单独立项）。

- **目标**：根治「加工具改 6 处」与「schema/Input 双事实源漂移」。
- **三选项权衡**：

| 选项 | 做法 | 减同步点 | 根治 schema 漂移 | 风险 |
|---|---|---|---|---|
| **(a) schemars derive** | 引 `schemars`，Input 加 `#[derive(JsonSchema)]`，`list_tools` 用 `schema_for!` 生成 | ⑤ 自动 | ✅ | 中（新依赖 + 23 struct 改 + 字段 description 靠 attribute，需校验 enum/default 还原） |
| **(b) 声明宏 `register_tool!`** | 一处声明展开 ⑤⑥④（schema 仍手写 json） | 6→3 | ❌ | 中 |
| **(c) 折中（推荐先做）** | 仅用宏统一 ④ NoopProvider + ⑥ dispatch（已有 `dispatch_tool!` 基础） | 6→4 | ❌ | 低 |

- **推荐路径**：**先做 (c) 的 PoC**——NoopProvider 用宏生成 stub、dispatch 已有宏，立即去掉 2 处同步点、风险最低。**(a) schemars 作为独立可选增强**，单独立项评估（需确认它对 `#[serde(default="fn")]`、enum、嵌套 array schema 的还原度，PoC 1 个工具验证后再铺开）。
- **涉及文件**：`src/mcp/server.rs`（宏定义 + NoopProvider/dispatch 改造）；若 (a) 则改 `Cargo.toml` + 全部 Input struct
- **风险**：(a) 高（依赖 + 全量 struct + schema 还原度）；(c) 低。
- **验收**：加一个 stub 工具验证同步点减少；所有现有工具 schema 与重构前字节级一致（`test_list_tools`/`test_tool_schemas_require_project_id` 兜底）。

---

## 五、推进顺序与里程碑

| 批次 | Phase | 产出 | 风险 |
|---|---|---|---|
| **P0**（先做，立竿见影） | **0 ✅** → **1 ✅** → **5 ✅** | server.rs 生产代码 2103→1711、embedding/传输独立、消除最后裸 SQL | 低 |
| **P1** | **2 ✅** | 传输/业务分离，为 HTTP transport 铺路 | 中 |
| **P2**（易过度设计，克制） | 3 ⊘ 放弃 → **4 ✅**（NoopProvider 宏化） | NoopProvider 减一处同步点 | 中 |

> 每个 Phase 完成后：更新本文件状态、回写路线图 §4.1、commit 一个独立 PR。P2 的 Phase 3/4 在动工前需重新评审，确认抽象收益。

---

## 六、风险登记

| 风险 | 缓解 |
|---|---|
| feature gate（`semantic`）迁移导致默认编译失败 | Phase 1 逐处核对 `#[cfg]`，`cargo build` 与 `--features semantic` 双跑 |
| 并发路径（worker/mpsc/poisoned mutex）回归 | Phase 2 逐行保留，`concurrent_handle_request_is_safe` 测试兜底 |
| 过度设计（Phase 3/4） | 抽象不顺手即放弃；PoC 先行、单工具验证再铺开 |
| schema 漂移（Phase 4a） | schemars 还原度 PoC 验证；schema 字节级比对 |
| 大文件改动引入隐性 bug | 严格「一 Phase 一 PR」+ 每步全量测试 |

---

## 七、omp 委派清单（每 Phase 的任务边界）

> 每个 Phase 委派一次 `omp-coder`，附本设计文档对应小节 + 验收标准。

- **Phase 0**：§4.Phase0。零可见性改动，`#[path]` 子模块外移测试。验收：测试全绿、server.rs ~2100 行。
- **Phase 1**：§4.Phase1。抽 `embedding_service.rs`，封装 index/fuse/reindex 三路径。验收：双 feature 编译 + 测试全绿。
- **Phase 5**：§4.Phase5。timeline 裸 SQL 迁入 repository。验收：timeline 输出一致、mcp 无业务 SQL。
- **Phase 2**：§4.Phase2。抽 `transport.rs`（JsonRpc + run_stdio + RequestHandler trait）。验收：含并发测试全绿 + 手动 stdio 交互。
- **Phase 3**：§4.Phase3。**可选**，先评审 trait 抽象收益再决定。
- **Phase 4**：§4.Phase4。先做选项 (c) PoC；选项 (a) schemars 单独立项。
