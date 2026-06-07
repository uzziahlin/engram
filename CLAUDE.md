# Engram 记忆系统使用规范

## 记忆工具使用策略

本项目配置了 engram MCP server（project_id = "engram"），你应当在以下场景主动调用记忆工具：

### 何时写入记忆

1. **完成重要功能开发后** → `create_episodic`
   - 记录：做了什么、修改了哪些文件、关键决策
   - importance 根据影响范围设定（小修复 0.3，核心功能 0.7+）

2. **做出架构/技术选型决策时** → `create_decision`
   - 记录：决策标题、背景上下文、选择理由、权衡取舍
   - 必须填写 related_files 和 tags

3. **遇到 bug/故障并解决后** → `create_failure`
   - 记录：问题现象、根因分析、修复方案、预防措施
   - severity: 1=微小, 3=中等, 5=严重

4. **建立或发现工作流程/约定时** → `create_procedural`
   - 记录：流程名称、步骤、使用的工具

### 何时读取记忆

- **开始新任务前** → `search_memory` 查找相关的历史记忆
- **修改文件前** → `related_files` 查看该文件的历史关联
- **需要了解项目背景时** → `architectural_decisions`

### 示例

**场景：修复了一个 FTS5 搜索崩溃的 bug**

首先记录经验教训（长期价值更高）：
```
create_failure(project_id="engram", incident="搜索含 UNIQUE 关键词时崩溃",
  root_cause="FTS5 MATCH 将 UNIQUE 解析为列过滤器",
  fix="新增 sanitize_fts_query 将查询包裹为短语查询",
  prevention="所有 FTS5 查询必须经过 sanitize",
  severity=4, tags=["fts5", "security"])
```

然后记录工作过程：
```
create_episodic(project_id="engram", session_id="<session>",
  summary="修复 FTS5 查询注入漏洞",
  content="用户查询中的 SQL 保留词被 FTS5 误解...（详细描述）",
  files_touched=["src/storage/repository.rs"], tags=["fts5", "bugfix"],
  importance=0.7)
```
