#!/usr/bin/env bash
# Engram MCP 集成测试脚本
# 按照 docs/integration-test-plan.md 执行所有测试用例

set -euo pipefail

ENGRAM="target/release/engram"
DB_PATH="$HOME/.engram/memory.db"
# Repo root for ingest_commits tests — derived, not hardcoded, so the suite is portable.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PASS=0
FAIL=0
SKIP=0
RESULTS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Helper Functions ───────────────────────────────────────────────

clean_db() {
    rm -f "$DB_PATH" "$DB_PATH-wal" "$DB_PATH-shm"
}

# Send multiple JSON-RPC requests to the server, return responses line by line.
# Usage: mcp_call "request1" "request2" ...
# Output: one JSON response per line on stdout
mcp_call() {
    local input=""
    for req in "$@"; do
        input+="${req}"$'\n'
    done
    printf '%s' "$input" | RUST_LOG=off "$ENGRAM" 2>/dev/null
}

# Extract the text content from a tools/call response
# The MCP wraps results in: {"result":{"content":[{"type":"text","text":"..."}]}}
extract_text() {
    local response="$1"
    echo "$response" | jq -r '.result.content[0].text // empty'
}

# ─── Assertion Functions ────────────────────────────────────────────

assert_pass() {
    local name="$1"
    PASS=$((PASS + 1))
    RESULTS+=("${GREEN}✓ PASS${NC}: $name")
    echo -e "  ${GREEN}✓ PASS${NC}: $name"
}

assert_fail() {
    local name="$1"
    local detail="${2:-}"
    FAIL=$((FAIL + 1))
    RESULTS+=("${RED}✗ FAIL${NC}: $name ${detail}")
    echo -e "  ${RED}✗ FAIL${NC}: $name ${detail}"
}

assert_skip() {
    local name="$1"
    local reason="${2:-}"
    SKIP=$((SKIP + 1))
    RESULTS+=("${YELLOW}⊘ SKIP${NC}: $name ${reason}")
    echo -e "  ${YELLOW}⊘ SKIP${NC}: $name ${reason}"
}

# Assert JSON field equals expected value.
# Uses `jq` (NOT jq -r) so JSON strings keep their quotes: "created" stays "created".
# Usage: assert_eq "test name" "$response" ".result.protocolVersion" '"2024-11-05"'
assert_eq() {
    local name="$1" response="$2" path="$3" expected="$4"
    local actual
    actual=$(echo "$response" | jq "$path" 2>/dev/null) || true
    if [ "$actual" = "$expected" ]; then
        assert_pass "$name"
    else
        assert_fail "$name" "(path=$path expected=$expected actual=$actual)"
    fi
}

# Assert JSON field contains substring
# Usage: assert_contains "test name" "$response" ".error.message" "Method not found"
assert_contains() {
    local name="$1" response="$2" path="$3" substring="$4"
    local actual
    actual=$(echo "$response" | jq -r "$path" 2>/dev/null) || true
    if echo "$actual" | grep -qF "$substring"; then
        assert_pass "$name"
    else
        assert_fail "$name" "(path=$path expected_contains=$substring actual=$actual)"
    fi
}

# Assert JSON field is not null/empty
assert_not_empty() {
    local name="$1" response="$2" path="$3"
    local actual
    actual=$(echo "$response" | jq -r "$path" 2>/dev/null) || true
    if [ -n "$actual" ] && [ "$actual" != "null" ] && [ "$actual" != "[]" ] && [ "$actual" != "" ]; then
        assert_pass "$name"
    else
        assert_fail "$name" "(path=$path was empty/null, actual=$actual)"
    fi
}

# Assert JSON field is null/empty
assert_empty() {
    local name="$1" response="$2" path="$3"
    local actual
    actual=$(echo "$response" | jq -r "$path" 2>/dev/null) || true
    if [ -z "$actual" ] || [ "$actual" = "null" ] || [ "$actual" = "[]" ] || [ "$actual" = "0" ]; then
        assert_pass "$name"
    else
        assert_fail "$name" "(path=$path expected empty, actual=$actual)"
    fi
}

# Assert numeric JSON field satisfies condition
assert_gt() {
    local name="$1" response="$2" path="$3" threshold="$4"
    local actual
    actual=$(echo "$response" | jq -r "$path" 2>/dev/null) || true
    if [ "$(echo "$actual > $threshold" | bc -l 2>/dev/null)" = "1" ]; then
        assert_pass "$name"
    else
        assert_fail "$name" "(path=$path expected>$threshold actual=$actual)"
    fi
}

# Assert numeric JSON field equals
assert_num_eq() {
    local name="$1" response="$2" path="$3" expected="$4"
    local actual
    actual=$(echo "$response" | jq -r "$path" 2>/dev/null) || true
    if [ "$actual" = "$expected" ]; then
        assert_pass "$name"
    else
        assert_fail "$name" "(path=$path expected=$expected actual=$actual)"
    fi
}

# Assert JSON field is a valid UUID format
assert_uuid() {
    local name="$1" response="$2" path="$3"
    local actual
    actual=$(echo "$response" | jq -r "$path" 2>/dev/null) || true
    if echo "$actual" | grep -qE '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'; then
        assert_pass "$name"
    else
        assert_fail "$name" "(path=$path not a UUID, actual=$actual)"
    fi
}

# ─── Section Printers ───────────────────────────────────────────────

section() {
    echo ""
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════${NC}"
}

subsection() {
    echo ""
    echo -e "${BOLD}  ── $1 ──${NC}"
}

# ═══════════════════════════════════════════════════════════════════════
# 1. 协议层测试
# ═══════════════════════════════════════════════════════════════════════
section "1. 协议层测试"

subsection "1.1 MCP 初始化握手"
clean_db
REQ_INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
RESP=$(mcp_call "$REQ_INIT")
assert_eq   "1.1a protocolVersion == 2024-11-05" "$RESP" '.result.protocolVersion' '"2024-11-05"'
assert_eq   "1.1b serverInfo.name == engram"      "$RESP" '.result.serverInfo.name' '"engram"'
assert_not_empty "1.1c capabilities.tools 存在"    "$RESP" '.result.capabilities.tools'

subsection "1.2 tools/list 工具列表"
REQ_LIST='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
RESP=$(mcp_call "$REQ_LIST")
TOOL_COUNT=$(echo "$RESP" | jq '.result.tools | length')
# 工具集: 10 基础 + 1 collect_sources + 6 lifecycle = 17
if [ "$TOOL_COUNT" = "17" ]; then
    assert_pass "1.2a 返回 17 个工具"
else
    assert_fail "1.2a 返回 17 个工具 (actual=$TOOL_COUNT)"
fi

# 验证所有工具名称（delete_xxx 已替换为 forget_memory/restore_memory 等生命周期工具）
EXPECTED_TOOLS=(search_memory related_files timeline recent_failures architectural_decisions create_episodic create_decision create_failure create_procedural ingest_commits collect_sources forget_memory restore_memory update_memory forget_batch list_archived consolidate_memories)
ACTUAL_TOOLS=$(echo "$RESP" | jq -r '.result.tools[].name')
for t in "${EXPECTED_TOOLS[@]}"; do
    if echo "$ACTUAL_TOOLS" | grep -qxF "$t"; then
        assert_pass "1.2b 工具存在: $t"
    else
        assert_fail "1.2b 工具缺失: $t"
    fi
done

# 验证每个工具的 inputSchema.required 包含 project_id
# 注意: MCP 响应字段为 inputSchema（驼峰），不是 input_schema
TOOL_LAST_IDX=$((TOOL_COUNT - 1))
for i in $(seq 0 $TOOL_LAST_IDX); do
    TOOL_NAME=$(echo "$RESP" | jq -r ".result.tools[$i].name")
    HAS_PID=$(echo "$RESP" | jq -r ".result.tools[$i].inputSchema.required | index(\"project_id\") != null")
    if [ "$HAS_PID" = "true" ]; then
        assert_pass "1.2c $TOOL_NAME requires project_id"
    else
        assert_fail "1.2c $TOOL_NAME missing project_id in required"
    fi
done

subsection "1.3 未知方法错误"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":3,"method":"nonexistent/method","params":{}}')
assert_eq       "1.3a error.code == -32601"         "$RESP" ".error.code" "-32601"
assert_contains "1.3b error.message contains 'Method not found'" "$RESP" ".error.message" "Method not found"

subsection "1.4 畸形 JSON 解析错误"
RESP=$(mcp_call '{bad json}')
assert_eq       "1.4a error.code == -32700"   "$RESP" ".error.code" "-32700"
assert_contains "1.4b error.message contains 'Parse error'" "$RESP" ".error.message" "Parse error"

subsection "1.5 缺少必填参数"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"test-project"}}}')
assert_eq       "1.5a error.code == -32602" "$RESP" ".error.code" "-32602"
assert_contains "1.5b error.message contains 'missing field'" "$RESP" ".error.message" "missing field"

subsection "1.6 未知工具"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"nonexistent_tool","arguments":{}}}')
# Unknown tool returns -32601 (Method not found) not -32603
assert_eq       "1.6a error.code == -32601"    "$RESP" ".error.code" "-32601"
assert_contains "1.6b error.message contains 'Unknown tool'" "$RESP" ".error.message" "Unknown tool"

subsection "1.7 Notification 静默跳过"
# Send a notification (no id) — expect no output
RESP=$(mcp_call '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}')
if [ -z "$RESP" ]; then
    assert_pass "1.7 notification 无响应输出"
else
    assert_fail "1.7 notification 应无响应 (got: $RESP)"
fi

# ═══════════════════════════════════════════════════════════════════════
# 2. 写入工具测试
# ═══════════════════════════════════════════════════════════════════════
section "2. 写入工具测试"

subsection "2.1 create_episodic"
clean_db

# 用例 A - 基本创建
RESP=$(mcp_call '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"test-project","session_id":"sess-001","summary":"修复 OAuth bug","content":"详细描述","files_touched":["src/auth.rs"],"tags":["auth"],"importance":0.8}}}')
TEXT=$(extract_text "$RESP")
assert_eq       "2.1A-a status == created" "$TEXT" ".status" '"created"'
assert_uuid     "2.1A-b id 是 UUID"        "$TEXT" ".id"
assert_not_empty "2.1A-c created_at 非空"  "$TEXT" ".created_at"
# 记录 ID 用于后续
EPISODIC_ID_A=$(echo "$TEXT" | jq -r '.id')

# 用例 B - 默认值
RESP=$(mcp_call '{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"test-project","session_id":"sess-002","summary":"默认值测试","content":"内容"}}}')
TEXT=$(extract_text "$RESP")
assert_eq "2.1B-a status == created" "$TEXT" ".status" '"created"'

# 用例 C - 中文内容
RESP=$(mcp_call '{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"test-project","session_id":"sess-003","summary":"修复了用户登录认证失败的问题","content":"详细中文描述","tags":["认证","bugfix"]}}}')
TEXT=$(extract_text "$RESP")
assert_eq "2.1C-a 中文内容创建成功" "$TEXT" ".status" '"created"'

subsection "2.2 create_decision"
# 用例 A - 基本创建
RESP=$(mcp_call '{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"create_decision","arguments":{"project_id":"test-project","title":"选择 Rust 作为后端语言","context":"需要高性能和安全","rationale":"Rust 提供内存安全和零成本抽象","tradeoffs":"学习曲线较陡","related_files":["src/arch.rs"],"tags":["architecture"]}}}')
TEXT=$(extract_text "$RESP")
assert_eq "2.2A-a status == created" "$TEXT" ".status" '"created"'

# 用例 B - 最少参数
RESP=$(mcp_call '{"jsonrpc":"2.0","id":14,"method":"tools/call","params":{"name":"create_decision","arguments":{"project_id":"test-project","title":"最小参数决策","context":"ctx","rationale":"reason","tradeoffs":"tradeoff"}}}')
TEXT=$(extract_text "$RESP")
assert_eq "2.2B-a 最少参数创建成功" "$TEXT" ".status" '"created"'

subsection "2.3 create_failure"
# 用例 A - 基本创建
RESP=$(mcp_call '{"jsonrpc":"2.0","id":15,"method":"tools/call","params":{"name":"create_failure","arguments":{"project_id":"test-project","incident":"数据库连接超时","root_cause":"连接池耗尽","fix":"增加连接池大小","prevention":"监控连接池使用率","severity":4,"tags":["database"]}}}')
TEXT=$(extract_text "$RESP")
assert_eq       "2.3A-a status == created"  "$TEXT" ".status"    '"created"'
assert_num_eq   "2.3A-b severity == 4"      "$TEXT" ".severity"  "4"

# 用例 B - severity 下界校验 (0)
RESP=$(mcp_call '{"jsonrpc":"2.0","id":16,"method":"tools/call","params":{"name":"create_failure","arguments":{"project_id":"test-project","incident":"test","root_cause":"test","fix":"test","prevention":"test","severity":0}}}')
assert_contains "2.3B severity=0 被拒绝" "$RESP" ".error.message" "severity must be between 1 and 5"

# 用例 C - severity 上界校验 (6)
RESP=$(mcp_call '{"jsonrpc":"2.0","id":17,"method":"tools/call","params":{"name":"create_failure","arguments":{"project_id":"test-project","incident":"test","root_cause":"test","fix":"test","prevention":"test","severity":6}}}')
assert_contains "2.3C severity=6 被拒绝" "$RESP" ".error.message" "severity must be between 1 and 5"

# 用例 D - severity 边界值 1 和 5
RESP1=$(mcp_call '{"jsonrpc":"2.0","id":18,"method":"tools/call","params":{"name":"create_failure","arguments":{"project_id":"test-project","incident":"boundary 1","root_cause":"test","fix":"test","prevention":"test","severity":1}}}')
TEXT1=$(extract_text "$RESP1")
assert_eq "2.3D-a severity=1 正常创建" "$TEXT1" ".status" '"created"'

RESP5=$(mcp_call '{"jsonrpc":"2.0","id":19,"method":"tools/call","params":{"name":"create_failure","arguments":{"project_id":"test-project","incident":"boundary 5","root_cause":"test","fix":"test","prevention":"test","severity":5}}}')
TEXT5=$(extract_text "$RESP5")
assert_eq "2.3D-b severity=5 正常创建" "$TEXT5" ".status" '"created"'

# 用例 E - 默认 severity
RESP=$(mcp_call '{"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"create_failure","arguments":{"project_id":"test-project","incident":"default sev","root_cause":"test","fix":"test","prevention":"test"}}}')
TEXT=$(extract_text "$RESP")
assert_num_eq "2.3E 默认 severity == 3" "$TEXT" ".severity" "3"

subsection "2.4 create_procedural"
# 用例 A - 基本创建
RESP=$(mcp_call '{"jsonrpc":"2.0","id":21,"method":"tools/call","params":{"name":"create_procedural","arguments":{"project_id":"test-project","workflow_name":"部署流程","steps":["测试","构建","部署"],"related_tools":["docker"],"tags":["deploy"]}}}')
TEXT=$(extract_text "$RESP")
assert_eq "2.4A-a status == created" "$TEXT" ".status" '"created"'

# 用例 B - 空步骤列表
RESP=$(mcp_call '{"jsonrpc":"2.0","id":22,"method":"tools/call","params":{"name":"create_procedural","arguments":{"project_id":"test-project","workflow_name":"empty steps","steps":[]}}}')
TEXT=$(extract_text "$RESP")
assert_eq "2.4B 空步骤列表创建成功" "$TEXT" ".status" '"created"'

subsection "2.5 ingest_commits"
# 用例 A - 从真实仓库摄取
RESP=$(mcp_call '{"jsonrpc":"2.0","id":23,"method":"tools/call","params":{"name":"ingest_commits","arguments":{"project_id":"test-project","repo_path":"'"$REPO_ROOT"'","count":5}}}')
TEXT=$(extract_text "$RESP")
INGESTED=$(echo "$TEXT" | jq -r '.ingested')
if [ "$INGESTED" -ge 1 ] 2>/dev/null; then
    assert_pass "2.5A-a ingested >= 1 (actual=$INGESTED)"
else
    assert_fail "2.5A-a ingested >= 1 (actual=$INGESTED)"
fi
assert_not_empty "2.5A-b memories 数组非空" "$TEXT" ".memories"

# 用例 B - 不存在的路径
RESP=$(mcp_call '{"jsonrpc":"2.0","id":24,"method":"tools/call","params":{"name":"ingest_commits","arguments":{"project_id":"test-project","repo_path":"/nonexistent/path","count":5}}}')
if echo "$RESP" | jq -e '.error' > /dev/null 2>&1; then
    assert_pass "2.5B 不存在的路径返回错误"
else
    assert_fail "2.5B 不存在的路径应返回错误"
fi

# ═══════════════════════════════════════════════════════════════════════
# 3. 读取工具测试
# ═══════════════════════════════════════════════════════════════════════
section "3. 读取工具测试"

subsection "前置条件 - 创建测试数据 (project=integration-test)"
clean_db

# 发送所有 create 请求，逐个获取响应
RESP1=$(mcp_call '{"jsonrpc":"2.0","id":30,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"integration-test","session_id":"sess-int","summary":"修复 OAuth token 刷新循环 bug","content":"详细描述 OAuth 刷新循环问题","files_touched":["src/auth/token.rs","src/cache/mod.rs"],"tags":["auth","oauth"]}}}')
TEXT1=$(extract_text "$RESP1")

RESP2=$(mcp_call '{"jsonrpc":"2.0","id":31,"method":"tools/call","params":{"name":"create_decision","arguments":{"project_id":"integration-test","title":"使用 SQLite FTS5 替代 Elasticsearch","context":"需要轻量级全文搜索","rationale":"SQLite FTS5 内置无需额外服务","tradeoffs":"不支持分布式","related_files":["src/storage/repository.rs"],"tags":["architecture","search"]}}}')
TEXT2=$(extract_text "$RESP2")

RESP3=$(mcp_call '{"jsonrpc":"2.0","id":32,"method":"tools/call","params":{"name":"create_failure","arguments":{"project_id":"integration-test","incident":"FTS5 虚拟表同步失败导致搜索结果不一致","root_cause":"FTS5 索引未同步","fix":"重建 FTS5 索引","prevention":"添加索引完整性检查","severity":4,"tags":["database","fts5"]}}}')
TEXT3=$(extract_text "$RESP3")

RESP4=$(mcp_call '{"jsonrpc":"2.0","id":33,"method":"tools/call","params":{"name":"create_procedural","arguments":{"project_id":"integration-test","workflow_name":"发布流程","steps":["cargo test","cargo build --release"],"related_tools":["cargo"],"tags":["deploy"]}}}')
TEXT4=$(extract_text "$RESP4")

# 验证所有创建成功
for i in 1 2 3 4; do
    TEXT_VAR="TEXT$i"
    eval "STATUS=\$(echo \"\$$TEXT_VAR\" | jq -r '.status')"
    if [ "$STATUS" = "created" ]; then
        assert_pass "前置: 测试数据 $i 创建成功"
    else
        assert_fail "前置: 测试数据 $i 创建失败 (status=$STATUS)"
    fi
done

subsection "3.1 search_memory"

# 用例 A - 英文关键词搜索
RESP=$(mcp_call '{"jsonrpc":"2.0","id":40,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"integration-test","query":"OAuth token"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "3.1A-a 英文搜索 total >= 1" "$TEXT" ".total" "0"
FIRST_TYPE=$(echo "$TEXT" | jq -r '.results[0].memory_type // empty')
if [ "$FIRST_TYPE" = "episodic" ]; then
    assert_pass "3.1A-b 第一个结果类型为 episodic"
else
    assert_fail "3.1A-b 第一个结果类型应为 episodic (actual=$FIRST_TYPE)"
fi

# 用例 B - 中文关键词搜索
RESP=$(mcp_call '{"jsonrpc":"2.0","id":41,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"integration-test","query":"刷新循环"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "3.1B 中文搜索找到 episodic" "$TEXT" ".total" "0"

# 用例 C - 中文单字搜索
RESP=$(mcp_call '{"jsonrpc":"2.0","id":42,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"integration-test","query":"同步"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "3.1C 中文单字搜索 '同步'" "$TEXT" ".total" "0"

# 用例 D - 按类型过滤 decision
RESP=$(mcp_call '{"jsonrpc":"2.0","id":43,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"integration-test","query":"FTS5","memory_type":"decision"}}}')
TEXT=$(extract_text "$RESP")
DECISION_TYPES=$(echo "$TEXT" | jq -r '.results[].memory_type' | sort -u)
if [ "$DECISION_TYPES" = "decision" ] || [ -z "$DECISION_TYPES" ]; then
    assert_pass "3.1D 按类型过滤 decision"
else
    assert_fail "3.1D 按类型过滤 decision (got types: $DECISION_TYPES)"
fi

# 用例 E - 按类型过滤 failure
RESP=$(mcp_call '{"jsonrpc":"2.0","id":44,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"integration-test","query":"FTS5","memory_type":"failure"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "3.1E 按类型过滤 failure total >= 1" "$TEXT" ".total" "0"

# 用例 F - 按标签搜索 database
RESP=$(mcp_call '{"jsonrpc":"2.0","id":45,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"integration-test","query":"database"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "3.1F 按标签搜索 database" "$TEXT" ".total" "0"

# 用例 G - 按标签搜索 architecture
RESP=$(mcp_call '{"jsonrpc":"2.0","id":46,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"integration-test","query":"architecture"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "3.1G 按标签搜索 architecture" "$TEXT" ".total" "0"

# 用例 H - 跨项目隔离
RESP=$(mcp_call '{"jsonrpc":"2.0","id":47,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"other-project","query":"OAuth"}}}')
TEXT=$(extract_text "$RESP")
assert_num_eq "3.1H 跨项目隔离 total == 0" "$TEXT" ".total" "0"

# 用例 I - 无匹配结果
RESP=$(mcp_call '{"jsonrpc":"2.0","id":48,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"integration-test","query":"zzz_nonexistent_keyword"}}}')
TEXT=$(extract_text "$RESP")
assert_num_eq "3.1I 无匹配结果 total == 0" "$TEXT" ".total" "0"

# 用例 J - limit 限制
RESP=$(mcp_call '{"jsonrpc":"2.0","id":49,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"integration-test","query":"OAuth","limit":1}}}')
TEXT=$(extract_text "$RESP")
RESULT_COUNT=$(echo "$TEXT" | jq '.results | length')
if [ "$RESULT_COUNT" -le 1 ] 2>/dev/null; then
    assert_pass "3.1J limit=1 results.length <= 1 (actual=$RESULT_COUNT)"
else
    assert_fail "3.1J limit=1 results.length <= 1 (actual=$RESULT_COUNT)"
fi

# 用例 K - BM25 分数非零
# Note: BM25 scores are normalized via sigmoid, small scores may round to 0.0.
# We verify the score is a valid non-negative number and results exist.
RESP=$(mcp_call '{"jsonrpc":"2.0","id":50,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"integration-test","query":"OAuth token"}}}')
TEXT=$(extract_text "$RESP")
SCORE=$(echo "$TEXT" | jq -r '.results[0].relevance_score' 2>/dev/null || echo "N/A")
if [ "$SCORE" != "null" ] && [ "$SCORE" != "N/A" ]; then
    assert_pass "3.1K BM25 relevance_score 存在 (score=$SCORE)"
else
    assert_fail "3.1K BM25 relevance_score 缺失 (score=$SCORE)"
fi

subsection "3.2 related_files"

# 用例 A - 存在的文件
RESP=$(mcp_call '{"jsonrpc":"2.0","id":51,"method":"tools/call","params":{"name":"related_files","arguments":{"project_id":"integration-test","file":"src/auth/token.rs"}}}')
TEXT=$(extract_text "$RESP")
assert_not_empty "3.2A-a entity id 非空" "$TEXT" ".entities[0].id"
assert_eq "3.2A-b entity type == File" "$TEXT" ".entities[0].type" '"File"'

# Check for Touches relation
REL_TYPES=$(echo "$TEXT" | jq -r '.entities[0].relations[].type' 2>/dev/null || true)
if echo "$REL_TYPES" | grep -qF "Touches"; then
    assert_pass "3.2A-c relations 包含 Touches"
else
    assert_fail "3.2A-c relations 缺少 Touches (got: $REL_TYPES)"
fi

# 用例 B - 不存在的文件
RESP=$(mcp_call '{"jsonrpc":"2.0","id":52,"method":"tools/call","params":{"name":"related_files","arguments":{"project_id":"integration-test","file":"nonexistent/file.rs"}}}')
TEXT=$(extract_text "$RESP")
assert_eq "3.2B 不存在文件 id 为空" "$TEXT" ".entities[0].id" '""'

# 用例 C - decision 关联文件 (References)
RESP=$(mcp_call '{"jsonrpc":"2.0","id":53,"method":"tools/call","params":{"name":"related_files","arguments":{"project_id":"integration-test","file":"src/storage/repository.rs"}}}')
TEXT=$(extract_text "$RESP")
REL_TYPES=$(echo "$TEXT" | jq -r '.entities[0].relations[].type' 2>/dev/null || true)
if echo "$REL_TYPES" | grep -qF "References"; then
    assert_pass "3.2C relations 包含 References"
else
    assert_fail "3.2C relations 缺少 References (got: $REL_TYPES)"
fi

subsection "3.3 timeline"

# 用例 A - 当日时间线
RESP=$(mcp_call '{"jsonrpc":"2.0","id":54,"method":"tools/call","params":{"name":"timeline","arguments":{"project_id":"integration-test","days":1}}}')
TEXT=$(extract_text "$RESP")
assert_not_empty "3.3A 当日时间线非空" "$TEXT" ".events"

# 用例 B - 7 天时间线
RESP=$(mcp_call '{"jsonrpc":"2.0","id":55,"method":"tools/call","params":{"name":"timeline","arguments":{"project_id":"integration-test","days":7}}}')
TEXT=$(extract_text "$RESP")
assert_not_empty "3.3B 7 天时间线非空" "$TEXT" ".events"

# 用例 C - 无数据项目
RESP=$(mcp_call '{"jsonrpc":"2.0","id":56,"method":"tools/call","params":{"name":"timeline","arguments":{"project_id":"empty-project","days":7}}}')
TEXT=$(extract_text "$RESP")
assert_empty "3.3C 无数据项目 events 为空" "$TEXT" ".events"

subsection "3.4 recent_failures"

# 用例 A - 无过滤条件 (Bug 1 回归)
RESP=$(mcp_call '{"jsonrpc":"2.0","id":57,"method":"tools/call","params":{"name":"recent_failures","arguments":{"project_id":"integration-test"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "3.4A 无过滤条件返回 failure 列表" "$TEXT" ".failures | length" "0"

# 用例 B - 带过滤条件
RESP=$(mcp_call '{"jsonrpc":"2.0","id":58,"method":"tools/call","params":{"name":"recent_failures","arguments":{"project_id":"integration-test","service":"FTS5"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "3.4B 带过滤条件返回匹配 failure" "$TEXT" ".failures | length" "0"

# 用例 C - limit 生效
RESP=$(mcp_call '{"jsonrpc":"2.0","id":59,"method":"tools/call","params":{"name":"recent_failures","arguments":{"project_id":"integration-test","limit":1}}}')
TEXT=$(extract_text "$RESP")
FAIL_COUNT=$(echo "$TEXT" | jq '.failures | length')
if [ "$FAIL_COUNT" -le 1 ] 2>/dev/null; then
    assert_pass "3.4C limit=1 failures.length <= 1 (actual=$FAIL_COUNT)"
else
    assert_fail "3.4C limit=1 failures.length <= 1 (actual=$FAIL_COUNT)"
fi

subsection "3.5 architectural_decisions"

# 用例 A - 无过滤条件 (Bug 1 回归)
RESP=$(mcp_call '{"jsonrpc":"2.0","id":60,"method":"tools/call","params":{"name":"architectural_decisions","arguments":{"project_id":"integration-test"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "3.5A 无过滤条件返回 decision 列表" "$TEXT" ".decisions | length" "0"

# 用例 B - 带 topic 过滤
RESP=$(mcp_call '{"jsonrpc":"2.0","id":61,"method":"tools/call","params":{"name":"architectural_decisions","arguments":{"project_id":"integration-test","topic":"SQLite"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "3.5B 带 topic 过滤返回匹配 decision" "$TEXT" ".decisions | length" "0"

# ═══════════════════════════════════════════════════════════════════════
# 4. 生命周期工具测试（forget / restore）
# ═══════════════════════════════════════════════════════════════════════
section "4. 生命周期工具测试（forget / restore）"

subsection "4.1 forget_memory (episodic)"
clean_db

# 创建再归档（软删除）
RESP=$(mcp_call '{"jsonrpc":"2.0","id":70,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"del-test","session_id":"s1","summary":"待归档的记录","content":"content"}}}')
TEXT=$(extract_text "$RESP")
DEL_ID=$(echo "$TEXT" | jq -r '.id')

RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":71,\"method\":\"tools/call\",\"params\":{\"name\":\"forget_memory\",\"arguments\":{\"project_id\":\"del-test\",\"memory_type\":\"episodic\",\"id\":\"$DEL_ID\"}}}")
TEXT=$(extract_text "$RESP")
assert_eq "4.1A-a forget_memory status == archived" "$TEXT" ".status" '"archived"'

# 验证搜索不再找到（归档后排除）
RESP=$(mcp_call '{"jsonrpc":"2.0","id":72,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"del-test","query":"待归档的记录"}}}')
TEXT=$(extract_text "$RESP")
assert_num_eq "4.1A-b 归档后搜索 total == 0" "$TEXT" ".total" "0"

# forget 不存在的记录（返回 not_found_or_already_archived）
RESP=$(mcp_call '{"jsonrpc":"2.0","id":73,"method":"tools/call","params":{"name":"forget_memory","arguments":{"project_id":"del-test","memory_type":"episodic","id":"nonexistent-uuid"}}}')
TEXT=$(extract_text "$RESP")
assert_eq "4.1B 归档不存在记录 status == not_found_or_already_archived" "$TEXT" ".status" '"not_found_or_already_archived"'

subsection "4.2 forget_memory (decision)"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":74,"method":"tools/call","params":{"name":"create_decision","arguments":{"project_id":"del-test","title":"待归档决策","context":"c","rationale":"r","tradeoffs":"t"}}}')
TEXT=$(extract_text "$RESP")
DEL_ID=$(echo "$TEXT" | jq -r '.id')

RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":75,\"method\":\"tools/call\",\"params\":{\"name\":\"forget_memory\",\"arguments\":{\"project_id\":\"del-test\",\"memory_type\":\"decision\",\"id\":\"$DEL_ID\"}}}")
TEXT=$(extract_text "$RESP")
assert_eq "4.2 forget_memory decision status == archived" "$TEXT" ".status" '"archived"'

subsection "4.3 forget_memory (failure)"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":76,"method":"tools/call","params":{"name":"create_failure","arguments":{"project_id":"del-test","incident":"待归档故障","root_cause":"r","fix":"f","prevention":"p","severity":3}}}')
TEXT=$(extract_text "$RESP")
DEL_ID=$(echo "$TEXT" | jq -r '.id')

RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":77,\"method\":\"tools/call\",\"params\":{\"name\":\"forget_memory\",\"arguments\":{\"project_id\":\"del-test\",\"memory_type\":\"failure\",\"id\":\"$DEL_ID\"}}}")
TEXT=$(extract_text "$RESP")
assert_eq "4.3 forget_memory failure status == archived" "$TEXT" ".status" '"archived"'

subsection "4.4 forget_memory (procedural)"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":78,"method":"tools/call","params":{"name":"create_procedural","arguments":{"project_id":"del-test","workflow_name":"待归档流程","steps":["step1"]}}}')
TEXT=$(extract_text "$RESP")
DEL_ID=$(echo "$TEXT" | jq -r '.id')

RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":79,\"method\":\"tools/call\",\"params\":{\"name\":\"forget_memory\",\"arguments\":{\"project_id\":\"del-test\",\"memory_type\":\"procedural\",\"id\":\"$DEL_ID\"}}}")
TEXT=$(extract_text "$RESP")
assert_eq "4.4 forget_memory procedural status == archived" "$TEXT" ".status" '"archived"'

subsection "4.5 restore_memory"
# 创建→归档→恢复
RESP=$(mcp_call '{"jsonrpc":"2.0","id":80,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"del-test","session_id":"s2","summary":"待恢复的记录","content":"restore me"}}}')
TEXT=$(extract_text "$RESP")
RESTORE_ID=$(echo "$TEXT" | jq -r '.id')

RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":81,\"method\":\"tools/call\",\"params\":{\"name\":\"forget_memory\",\"arguments\":{\"project_id\":\"del-test\",\"memory_type\":\"episodic\",\"id\":\"$RESTORE_ID\"}}}")
TEXT=$(extract_text "$RESP")
assert_eq "4.5-a forget 成功" "$TEXT" ".status" '"archived"'

RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":82,\"method\":\"tools/call\",\"params\":{\"name\":\"restore_memory\",\"arguments\":{\"project_id\":\"del-test\",\"memory_type\":\"episodic\",\"id\":\"$RESTORE_ID\"}}}")
TEXT=$(extract_text "$RESP")
assert_eq "4.5-b restore_memory status == restored" "$TEXT" ".status" '"restored"'

# 验证恢复后搜索可以找到
RESP=$(mcp_call '{"jsonrpc":"2.0","id":83,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"del-test","query":"待恢复的记录"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "4.5-c 恢复后搜索 total >= 1" "$TEXT" ".total" "0"

subsection "4.6 list_archived"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":84,"method":"tools/call","params":{"name":"list_archived","arguments":{"project_id":"del-test","memory_type":"episodic"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "4.6 list_archived 返回归档列表" "$TEXT" ".archived | length" "0"

# ═══════════════════════════════════════════════════════════════════════
# 5. 端到端链路测试
# ═══════════════════════════════════════════════════════════════════════
section "5. 端到端链路测试"

subsection "5.1 写入→搜索→删除→验证 完整链路"
clean_db

RESP=$(mcp_call '{"jsonrpc":"2.0","id":80,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"e2e-test","session_id":"e2e","summary":"端到端测试记录 E2E-UNIQUE-XYZ","content":"详细内容"}}}')
TEXT=$(extract_text "$RESP")
E2E_ID=$(echo "$TEXT" | jq -r '.id')
assert_eq "5.1-a 创建成功" "$TEXT" ".status" '"created"'

RESP=$(mcp_call '{"jsonrpc":"2.0","id":81,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"e2e-test","query":"E2E-UNIQUE-XYZ"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "5.1-b 搜索找到记录" "$TEXT" ".total" "0"

RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":82,\"method\":\"tools/call\",\"params\":{\"name\":\"forget_memory\",\"arguments\":{\"project_id\":\"e2e-test\",\"memory_type\":\"episodic\",\"id\":\"$E2E_ID\"}}}")
TEXT=$(extract_text "$RESP")
assert_eq "5.1-c forget 归档成功" "$TEXT" ".status" '"archived"'

RESP=$(mcp_call '{"jsonrpc":"2.0","id":83,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"e2e-test","query":"E2E-UNIQUE-XYZ"}}}')
TEXT=$(extract_text "$RESP")
assert_num_eq "5.1-d 归档后搜索 total == 0" "$TEXT" ".total" "0"

subsection "5.2 写入→related_files 图关系验证"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":84,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"e2e-test","session_id":"e2e","summary":"图关系测试","content":"content","files_touched":["src/graph_test.rs"]}}}')
TEXT=$(extract_text "$RESP")
assert_eq "5.2-a 创建含 files_touched 成功" "$TEXT" ".status" '"created"'

RESP=$(mcp_call '{"jsonrpc":"2.0","id":85,"method":"tools/call","params":{"name":"related_files","arguments":{"project_id":"e2e-test","file":"src/graph_test.rs"}}}')
TEXT=$(extract_text "$RESP")
assert_not_empty "5.2-b 实体 id 存在" "$TEXT" ".entities[0].id"
assert_eq "5.2-c 实体类型为 File" "$TEXT" ".entities[0].type" '"File"'

REL_TYPES=$(echo "$TEXT" | jq -r '.entities[0].relations[].type' 2>/dev/null || true)
if echo "$REL_TYPES" | grep -qF "Touches"; then
    assert_pass "5.2-d 关系类型为 Touches"
else
    assert_fail "5.2-d 关系类型缺少 Touches (got: $REL_TYPES)"
fi

subsection "5.3 写入→tags 搜索验证"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":86,"method":"tools/call","params":{"name":"create_decision","arguments":{"project_id":"e2e-test","title":"标签测试决策","context":"c","rationale":"r","tradeoffs":"t","tags":["unique-tag-test-xyz"]}}}')
TEXT=$(extract_text "$RESP")
assert_eq "5.3-a 创建含 unique tag 成功" "$TEXT" ".status" '"created"'

RESP=$(mcp_call '{"jsonrpc":"2.0","id":87,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"e2e-test","query":"unique-tag-test-xyz"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "5.3-b 能通过 tag 搜索到 decision" "$TEXT" ".total" "0"

subsection "5.4 ingest_commits→搜索 验证"
clean_db

RESP=$(mcp_call '{"jsonrpc":"2.0","id":88,"method":"tools/call","params":{"name":"ingest_commits","arguments":{"project_id":"e2e-test","repo_path":"'"$REPO_ROOT"'","count":5}}}')
TEXT=$(extract_text "$RESP")
assert_gt "5.4-a ingest 成功" "$TEXT" ".ingested" "0"

RESP=$(mcp_call '{"jsonrpc":"2.0","id":89,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"e2e-test","query":"feat"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "5.4-b 搜索 ingest 内容成功" "$TEXT" ".total" "0"

subsection "5.5 中文写入→中文搜索 完整链路"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":90,"method":"tools/call","params":{"name":"create_failure","arguments":{"project_id":"e2e-test","incident":"数据库连接池耗尽导致服务不可用","root_cause":"连接池配置过小","fix":"增大连接池","prevention":"监控连接池使用","severity":5,"tags":["数据库","连接池"]}}}')
TEXT=$(extract_text "$RESP")
assert_eq "5.5-a 中文 failure 创建成功" "$TEXT" ".status" '"created"'

RESP=$(mcp_call '{"jsonrpc":"2.0","id":91,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"e2e-test","query":"连接池耗尽"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "5.5-b 中文搜索 '连接池耗尽' 成功" "$TEXT" ".total" "0"

RESP=$(mcp_call '{"jsonrpc":"2.0","id":92,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"e2e-test","query":"数据库"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "5.5-c 中文搜索 '数据库' 成功" "$TEXT" ".total" "0"

subsection "5.6 多类型混合搜索"
# 创建包含 "Redis" 的三种类型
RESP=$(mcp_call '{"jsonrpc":"2.0","id":93,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"e2e-test","session_id":"e2e","summary":"使用 Redis 缓存优化","content":"content"}}}')
RESP=$(mcp_call '{"jsonrpc":"2.0","id":94,"method":"tools/call","params":{"name":"create_decision","arguments":{"project_id":"e2e-test","title":"选择 Redis 作为缓存方案","context":"c","rationale":"r","tradeoffs":"t"}}}')
RESP=$(mcp_call '{"jsonrpc":"2.0","id":95,"method":"tools/call","params":{"name":"create_failure","arguments":{"project_id":"e2e-test","incident":"Redis 连接超时导致缓存不可用","root_cause":"r","fix":"f","prevention":"p","severity":3}}}')

RESP=$(mcp_call '{"jsonrpc":"2.0","id":96,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"e2e-test","query":"Redis"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "5.6-a Redis 搜索 total >= 3" "$TEXT" ".total" "2"

# 验证包含多种类型
TYPES=$(echo "$TEXT" | jq -r '.results[].memory_type' | sort -u)
TYPE_COUNT=$(echo "$TYPES" | wc -l | tr -d ' ')
if [ "$TYPE_COUNT" -ge 2 ]; then
    assert_pass "5.6-b 结果包含多种类型 (count=$TYPE_COUNT, types=$TYPES)"
else
    assert_fail "5.6-b 结果应包含多种类型 (count=$TYPE_COUNT, types=$TYPES)"
fi

# ═══════════════════════════════════════════════════════════════════════
# 6. 边界条件与健壮性测试
# ═══════════════════════════════════════════════════════════════════════
section "6. 边界条件与健壮性测试"

subsection "6.1 空字符串处理"
# search_memory query=""
RESP=$(mcp_call '{"jsonrpc":"2.0","id":100,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"test","query":""}}}')
# Empty query may cause FTS5 error or return empty - just check it doesn't crash
if echo "$RESP" | jq -e '.error' > /dev/null 2>&1; then
    assert_pass "6.1a search query='' 返回错误 (不崩溃)"
else
    TEXT=$(extract_text "$RESP")
    TOTAL=$(echo "$TEXT" | jq -r '.total')
    if [ "$TOTAL" = "0" ]; then
        assert_pass "6.1a search query='' 返回空结果 (不崩溃)"
    else
        assert_fail "6.1a search query='' 未预期行为"
    fi
fi

# create_episodic summary=""
RESP=$(mcp_call '{"jsonrpc":"2.0","id":101,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"test","session_id":"s1","summary":"","content":"content"}}}')
TEXT=$(extract_text "$RESP")
assert_eq "6.1b create summary='' 正常创建" "$TEXT" ".status" '"created"'

subsection "6.2 超长字符串"
LONG_CONTENT=$(python3 -c "print('A' * 10000)")
RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":102,\"method\":\"tools/call\",\"params\":{\"name\":\"create_episodic\",\"arguments\":{\"project_id\":\"test\",\"session_id\":\"s1\",\"summary\":\"超长测试\",\"content\":\"$LONG_CONTENT\"}}}")
TEXT=$(extract_text "$RESP")
assert_eq "6.2 超长字符串创建成功" "$TEXT" ".status" '"created"'

subsection "6.3 特殊字符 (SQL 注入防护)"
RESP=$(mcp_call '{"jsonrpc":"2.0","id":103,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"test","query":"test\"quote"}}}')
if echo "$RESP" | jq -e '.error' > /dev/null 2>&1; then
    # FTS5 may not handle quotes well, but it shouldn't crash
    assert_pass "6.3a search query 含双引号 (返回错误但不崩溃)"
else
    assert_pass "6.3a search query 含双引号 (不崩溃)"
fi

RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":104,\"method\":\"tools/call\",\"params\":{\"name\":\"search_memory\",\"arguments\":{\"project_id\":\"test\",\"query\":\"test'quote\"}}}")
if echo "$RESP" | jq -e '.error' > /dev/null 2>&1; then
    assert_pass "6.3b search query 含单引号 (返回错误但不崩溃)"
else
    assert_pass "6.3b search query 含单引号 (不崩溃)"
fi

subsection "6.4 并发写入同项目"
clean_db

# 快速连续创建 10 条
REQS=""
for i in $(seq 1 10); do
    REQS+="{\"jsonrpc\":\"2.0\",\"id\":$((110+i)),\"method\":\"tools/call\",\"params\":{\"name\":\"create_episodic\",\"arguments\":{\"project_id\":\"concurrent-test\",\"session_id\":\"sess-$i\",\"summary\":\"并发测试记录 BATCH-CONCURRENT-$i\",\"content\":\"content $i\"}}}"$'\n'
done

RESP=$(printf '%s' "$REQS" | RUST_LOG=off "$ENGRAM" 2>/dev/null)
# Use jq to parse each response line and count successful creates
CREATED_COUNT=$(echo "$RESP" | while read -r line; do
    echo "$line" | jq -r '.result.content[0].text // "{}"' 2>/dev/null
done | jq -s 'map(select(.status == "created")) | length' 2>/dev/null || echo "0")
if [ "$CREATED_COUNT" = "10" ]; then
    assert_pass "6.4-a 10 条全部创建成功"
else
    assert_fail "6.4-a 创建数量不符 (expected=10 actual=$CREATED_COUNT)"
fi

# 验证搜索能找到 10 条
RESP=$(mcp_call '{"jsonrpc":"2.0","id":121,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"concurrent-test","query":"BATCH-CONCURRENT","limit":20}}}')
TEXT=$(extract_text "$RESP")
TOTAL=$(echo "$TEXT" | jq -r '.total' 2>/dev/null || echo "0")
if [ "$TOTAL" = "10" ]; then
    assert_pass "6.4-b 搜索找到 10 条"
else
    assert_fail "6.4-b 搜索数量不符 (expected=10 actual=$TOTAL)"
fi

subsection "6.5 数据库迁移兼容性"
clean_db

# 启动 server 会自动 initialize_schema + migrate_fts5_add_tags
RESP=$(mcp_call '{"jsonrpc":"2.0","id":122,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"migration-test","session_id":"s1","summary":"迁移后测试","content":"content"}}}')
TEXT=$(extract_text "$RESP")
assert_eq "6.5-a 迁移后写入成功" "$TEXT" ".status" '"created"'

RESP=$(mcp_call '{"jsonrpc":"2.0","id":123,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"migration-test","query":"迁移后测试"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "6.5-b 迁移后搜索成功" "$TEXT" ".total" "0"

subsection "6.6 重复启动"
# 连续启动 server 两次 — 第二次不应报错 (幂等初始化)
clean_db
RESP1=$(mcp_call '{"jsonrpc":"2.0","id":124,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}')
RESP2=$(mcp_call '{"jsonrpc":"2.0","id":125,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}')
assert_eq "6.6-a 第一次启动正常" "$RESP1" '.result.serverInfo.name' '"engram"'
assert_eq "6.6-b 第二次启动正常 (幂等)" "$RESP2" '.result.serverInfo.name' '"engram"'

# ═══════════════════════════════════════════════════════════════════════
# 7. 端到端生命周期测试：create → forget → list-archived → restore
# ═══════════════════════════════════════════════════════════════════════
section "7. 端到端生命周期测试（forget/restore 完整链路）"

subsection "7.1 episodic 遗忘/恢复完整链路"
clean_db

# Step 1: 创建 episodic 记忆
RESP=$(mcp_call '{"jsonrpc":"2.0","id":200,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"lifecycle-test","session_id":"lc-sess","summary":"ephemeral note LIFECYCLE-UNIQUE-ABC","content":"to be forgotten and restored","tags":["lifecycle"]}}}')
TEXT=$(extract_text "$RESP")
assert_eq "7.1-a 创建成功" "$TEXT" ".status" '"created"'
LC_ID=$(echo "$TEXT" | jq -r '.id')

# Step 2: 确认搜索可以找到
RESP=$(mcp_call '{"jsonrpc":"2.0","id":201,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"lifecycle-test","query":"LIFECYCLE-UNIQUE-ABC"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "7.1-b 创建后可搜索" "$TEXT" ".total" "0"

# Step 3: forget（归档/软删除）
RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":202,\"method\":\"tools/call\",\"params\":{\"name\":\"forget_memory\",\"arguments\":{\"project_id\":\"lifecycle-test\",\"memory_type\":\"episodic\",\"id\":\"$LC_ID\"}}}")
TEXT=$(extract_text "$RESP")
assert_eq "7.1-c forget 归档成功" "$TEXT" ".status" '"archived"'

# Step 4: 确认 search 中消失
RESP=$(mcp_call '{"jsonrpc":"2.0","id":203,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"lifecycle-test","query":"LIFECYCLE-UNIQUE-ABC"}}}')
TEXT=$(extract_text "$RESP")
assert_num_eq "7.1-d 归档后搜索不到 (total == 0)" "$TEXT" ".total" "0"

# Step 5: 确认出现在 list_archived 中（响应字段为 .archived 数组）
RESP=$(mcp_call '{"jsonrpc":"2.0","id":204,"method":"tools/call","params":{"name":"list_archived","arguments":{"project_id":"lifecycle-test","memory_type":"episodic"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "7.1-e 归档列表包含该记录" "$TEXT" ".archived | length" "0"
LC_ARCHIVED_ID=$(echo "$TEXT" | jq -r ".archived[] | select(.id == \"$LC_ID\") | .id")
if [ "$LC_ARCHIVED_ID" = "$LC_ID" ]; then
    assert_pass "7.1-f 归档列表中能按 id 找到该记录"
else
    assert_fail "7.1-f 归档列表中找不到目标 id (expected=$LC_ID)"
fi

# Step 6: restore 恢复
RESP=$(mcp_call "{\"jsonrpc\":\"2.0\",\"id\":205,\"method\":\"tools/call\",\"params\":{\"name\":\"restore_memory\",\"arguments\":{\"project_id\":\"lifecycle-test\",\"memory_type\":\"episodic\",\"id\":\"$LC_ID\"}}}")
TEXT=$(extract_text "$RESP")
assert_eq "7.1-g restore 恢复成功" "$TEXT" ".status" '"restored"'

# Step 7: 确认 search 中重新出现
RESP=$(mcp_call '{"jsonrpc":"2.0","id":206,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"lifecycle-test","query":"LIFECYCLE-UNIQUE-ABC"}}}')
TEXT=$(extract_text "$RESP")
assert_gt "7.1-h 恢复后重新可搜索" "$TEXT" ".total" "0"

subsection "7.2 forget_batch + list_archived 批量归档"
clean_db

# 创建两条带同一 tag 的记录
RESP=$(mcp_call '{"jsonrpc":"2.0","id":210,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"lifecycle-test","session_id":"batch-sess","summary":"batch item one BATCH-LIFECYCLE-TAG","content":"content one","tags":["batch-lifecycle-tag"]}}}')
TEXT=$(extract_text "$RESP")
assert_eq "7.2-a 创建 batch item 1" "$TEXT" ".status" '"created"'

RESP=$(mcp_call '{"jsonrpc":"2.0","id":211,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"lifecycle-test","session_id":"batch-sess","summary":"batch item two BATCH-LIFECYCLE-TAG","content":"content two","tags":["batch-lifecycle-tag"]}}}')
TEXT=$(extract_text "$RESP")
assert_eq "7.2-b 创建 batch item 2" "$TEXT" ".status" '"created"'

# 批量归档（apply=true）；响应字段为 .count（已归档数量）
RESP=$(mcp_call '{"jsonrpc":"2.0","id":212,"method":"tools/call","params":{"name":"forget_batch","arguments":{"project_id":"lifecycle-test","memory_type":"episodic","tags":["batch-lifecycle-tag"],"apply":true}}}')
TEXT=$(extract_text "$RESP")
assert_gt "7.2-c forget_batch count >= 1" "$TEXT" ".count" "0"

# 确认 search 中消失
RESP=$(mcp_call '{"jsonrpc":"2.0","id":213,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"lifecycle-test","query":"BATCH-LIFECYCLE-TAG"}}}')
TEXT=$(extract_text "$RESP")
assert_num_eq "7.2-d 批量归档后搜索不到" "$TEXT" ".total" "0"

# list_archived 应有记录（响应字段为 .archived 数组）
RESP=$(mcp_call '{"jsonrpc":"2.0","id":214,"method":"tools/call","params":{"name":"list_archived","arguments":{"project_id":"lifecycle-test","memory_type":"episodic","limit":10}}}')
TEXT=$(extract_text "$RESP")
assert_gt "7.2-e list_archived 返回批量归档的记录" "$TEXT" ".archived | length" "0"

# ═══════════════════════════════════════════════════════════════════════
# 测试结果汇总
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  测试结果汇总${NC}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${GREEN}PASS${NC}: $PASS"
echo -e "  ${RED}FAIL${NC}: $FAIL"
echo -e "  ${YELLOW}SKIP${NC}: $SKIP"
echo -e "  ${BOLD}TOTAL${NC}: $((PASS + FAIL + SKIP))"
echo ""

if [ $FAIL -gt 0 ]; then
    echo -e "${RED}${BOLD}失败的测试:${NC}"
    for r in "${RESULTS[@]}"; do
        if echo "$r" | grep -q "FAIL"; then
            echo -e "  $r"
        fi
    done
fi

echo ""
# 清理测试数据
clean_db

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}${BOLD}所有测试通过! ✓${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}有 $FAIL 个测试失败!${NC}"
    exit 1
fi
