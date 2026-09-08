# 深度审查修复与迭代计划（2026-09）

> **基线**：2026-09-08 全链路深度审查（v0.3.0，~15,600 行 Rust）
> **用途**：跟踪本轮审查发现的全部问题修复 + 推荐方向落地。与 `迭代优化路线图.md`（2026-06 轮）互补。
> **状态图例**：`[ ]` 待办 · `[~]` 进行中 · `[x]` 完成

> **状态（2026-09-08）**：全部批次与迭代方向已落地。232+ 单测 + 4 e2e 全绿；
> clippy `-D warnings` 双 feature 通过；schema 迁移 v2（FTS rowid 对齐 + 新分词重建）
> 与 v3（reflection re-arm）已注册。详细变更见 CHANGELOG [Unreleased]。

## 批次 1 · 检索质量修底（P1，所有用户每次搜索受益）

- [x] 1.1 CJK↔非CJK 边界插空格（`用Rust写` 等混排文本当前不可检索）
- [x] 1.2 tags/files_touched 列补 CJK 预处理（索引/查询不对称 → 中文 tag 检索不到）
- [x] 1.3 FTS 重建迁移（新分词对旧索引生效，含 backfill 流式化）
- [x] 1.4 rerank 权重归一化（和=1.0）+ BM25 归一化量纲校准（当前 relevance 信号被 type_prior 淹没）
- [x] 1.5 intent 抬权改抬 relevance（当前只抬静态先验，方向反了）
- [x] 1.6 多 token 查询 AND 优先、空结果回退 OR
- [x] 1.7 空查询/非法 memory_type 显式报错（当前静默空结果）
- [x] 1.8 intent 中文关键词补齐（部署/重构/失败/报错/事故/宕机）+ 词边界含数字
- [x] 1.9 config 权重校验（NaN/负数/归一化）

## 批次 2 · 安全与资源（P1）

- [x] 2.1 repo_path 校验：canonicalize + `[security] allowed_roots` 白名单 + home 目录防护
- [x] 2.2 walk 硬上限：文件数/总字节/深度/时间预算
- [x] 2.3 有界读取：`File::take` 流式截断（当前整读大文件后截断，可 OOM）
- [x] 2.4 embed token 截断（512，防 OOM/卡死）

## 批次 3 · 一致性（P1/P2）

- [x] 3.1 limit 钳制 1..=1000（MCP+CLI）、days 钳制非负有界、时间戳饱和运算
- [x] 3.2 MCP 坏配置 fail-fast（与 CLI 一致，杜绝默认库脑裂）
- [x] 3.3 工具业务错误 → result+isError；参数错误 → -32602；未知工具 → -32602
- [x] 3.4 get_*/update SQL 加 project 谓词；update 禁止改写 project_id
- [x] 3.5 ping/tools_list/prompts_list 在读线程内联处理（防单 worker 饥饿）
- [x] 3.6 JSON-RPC 校验（jsonrpc 字段、Invalid Request 与 Parse Error 区分）

## 批次 4 · 性能（P1/P2）

- [x] 4.1 git ingest 懒取（Sorting::ByCommitTime + take(N)，当前全量加载整史）
- [x] 4.2 FTS rowid 对齐删除（当前 DELETE WHERE memory_id 全表扫描，实测确认）
- [x] 4.3 启动期全表去重写入移入一次性迁移（当前每条 CLI 命令都跑）

## 批次 5 · CLI/MCP 检索统一（P1）

- [x] 5.1 CLI search 复用完整管线（intent/plan/rerank/semantic/config limit；当前是裸 BM25 分叉实现）

## 批次 6 · 启发式修复（P2）

- [x] 6.1 注释关键词去掉 TODO/NOTE（预算被淹没）
- [x] 6.2 CHANGELOG Fixed 小节精确匹配（`contains("fix")` 误收 "Prefix conventions"）
- [x] 6.3 根级 .circleci/.buildkite 匹配修复
- [x] 6.4 py/sh/sql/lua/rb `#`/`--` 注释支持
- [x] 6.5 looks_like_migration 收紧（"rewrite" 子串误报 BREAKING）
- [x] 6.6 parse_conventional type 白名单 + 未闭合括号
- [x] 6.7 is_fix_message 只扫 subject 首行
- [x] 6.8 collectors 共享一次 fs walk（当前三遍）

## 批次 7 · 功能修复（P1/P2）

- [x] 7.1 milestone 聚类加时间窗口（当前跨数年同类 commit 合并成一条）
- [x] 7.2 实体 GC（File/Tool 孤儿无限累积，污染 related_files）
- [x] 7.3 query_log 自动保留清理（接入 maintain）
- [x] 7.4 near-dup 长度预筛 + consolidate 文档修正（无时间窗）

## 批次 8 · 语义检索修复（P1，semantic feature）

- [x] 8.1 cosine/RRF 分数注入 relevance（当前算了就扔，语义只贡献候选并集）
- [x] 8.2 语义路径不再穿透 tags/before/memory_type 显式过滤
- [x] 8.3 update_memory 文本变更后重嵌
- [x] 8.4 blob 维度校验告警 + fuse 失败日志 + 换模型后 0 向量提示 reindex

## 迭代方向 2 · 写入管道自动化

- [x] I2.1 `engram hook` 子命令（Claude Code Stop/SessionEnd hook → 自动 session-import）
- [x] I2.3 session-import 按 session 幂等（upsert：重复触发原地刷新，不堆叠重复记忆）
- [x] I2.2 README/文档 hooks 集成指南

## 迭代方向 3 · 反馈闭环

- [x] I3.1 maintain 输出"知识缺口"（高频零命中查询）
- [x] I3.2 reflection rejected tag re-arm（新证据累积后允许再提案）

## 迭代方向 5 · 图参与检索

- [x] I5.1 搜索命中的记忆经关系图扩展二级候选（小幅 relevance 加成）

## 收尾

- [x] 全量测试 + clippy -D warnings + fmt（双 feature）
- [x] README/CHANGELOG 更新（含 reflection 措辞修正）
- [x] 迁移注册表：v2 FTS 重建（rowid + 新分词）、v3 reflection re-arm
