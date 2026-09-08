# engram.distill — 精馏 session-import 记忆

> 本 prompt 由 `engram hook`/`engram session-import` 落库的原始会话记忆触发。
> 目标：把"会话证据"提炼成"会话结论"——干净、可检索、有类型的记忆。

你在为项目 {{PROJECT_ID}} 精馏记忆库。步骤：

1. **取材**：`search_memory` 查询 tag `session-import`（可加 `before` 限定时间窗），
   拿到 hook 自动落库的原始会话记忆。它们的特征是：content 含
   "User prompts / Assistant conclusions / Files touched / Errors encountered" 段落。

2. **逐条判断**，按下面的质量门槛提炼成恰当时型的记忆（宁缺毋滥）：
   - 出现了"选型/取舍/为什么这么做" → `create_decision`（title + rationale + tradeoffs）
   - 有明确根因的错误（"Errors encountered" 段落最有价值）→ `create_failure`
     （incident + root_cause + fix + prevention + severity）
   - 可复用的操作流程（多步、有顺序）→ `create_procedural`（steps）
   - 其余只是"做过什么"的流水 → `create_episodic`，或直接不保留
   - 多条原始记忆讲同一件事 → 合并成一条，files 取并集

3. **溯源**：提炼出的记忆 tags 里保留来源标签（如 `session-import`）+ 提炼标记
   `distilled`，便于再次精馏时跳过。

4. **清理**：对已提炼的原始记忆调用 `forget_memory`（软删除，可恢复）。
   注意：仅在对应新记忆创建成功后再删除原始记忆。

质量门槛（不达标就不写）：
- failure 必须有真正的 root_cause（不是"出错了"的复述）
- decision 必须说清为什么（不是"我们用了 X"）
- 一条记忆一件事；细节留在 content，标题给检索用
