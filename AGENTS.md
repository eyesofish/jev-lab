# AGENTS.md — 给编码 Agent 的说明

这个仓库是一个**本地**的"决策模型"玩具：用开源复刻 `openjev`（Qwen3.5 训练的 NLI cross-encoder）
模拟 TypeSafe Jev 的接口形状。人类教程见 [`docs/tutorial.md`](docs/tutorial.md)。

## 怎么驱动

- **所有判断都通过 `jev.py` 的子命令**：`claim` / `choice` / `score` / `predict`。
  不要自己写模型加载代码，也不要用 `transformers` 直接调——`jev.py` 已经处理了模板、池化和概率。
- 一律使用 `.venv/bin/python jev.py ...`（系统 `python3` 里没有依赖）。工作目录是仓库根目录。
- **批量请一次传多条**：`--claim` 可重复，或 `--claims <文件>`（一行一条）。
  单次进程启动约 2 秒，一条判断约 0.7–0.9 秒——**不要写 for 循环反复启动进程**。
- 需要解析结果时加 `--json`，输出是 `{mode, ms, results[]}`，每条含 `verdict` 与三个概率。
- 上下文可以是文件、`-`（stdin）或内联字符串。上下文里**不要**写"请你判断……"之类的指令，
  判断写在 question / claim 参数里。

## 输出怎么读

- 三个概率：`contradiction`（材料反驳该陈述）/ `entailment`（材料支持）/ `neutral`（材料没提）。
- `P(entailment)` 是主信号；`[高]/[中]/[低]` 只是抄 TypeSafe 文档的用法示范，**不是校准阈值**。
- "没提"（neutral 高）和"被反驳"（contradiction 高）是两件不同的事，报告时要分开，不要合并成"低分"。
- `score` 模式是**用 NLI 硬拼的**，实测区分度很差（0.545 vs 0.451）。用它时必须在结论里注明它的不可靠，
  不要把它的排序当成评分。

## 不许乱说的结论（重要）

- 这个模型**不是** TypeSafe 的 Jev，也没有 RLCD 校准。它的概率就是三分类 softmax 输出。
  任何"它校准过""它不会幻觉"的说法都是错的。
- 不要声称它做过检索、生成或嵌入；它是 per-pair 的 NLI 判定器。
- 厂商（TypeSafe）声称的性能数字、以及官方 Playground 的可用性，都不是本仓库能验证的东西；
  引用时标明来源与不确定度。
- 概率**不可跨样本当阈值用**。只有"同一批候选里排序"是站得住的用法。

## 改代码时

- 主要文件：`jev.py`（CLI）、`setup.sh`（拉权重 + 建环境）、`docs/tutorial.md`（人类教程）、
  `docs/jev-notes.md`（研究笔记）。
- **不要提交** `model/` 与 `.venv/`（已在 `.gitignore` 里，权重共约 1.6 GB）。
- 改完 `jev.py` 后，跑一遍 `docs/tutorial.md` §1、§3.2、§3.3 里的命令作为回归，
  确认判决与文档里记录的一致（例如英文 demo 第 1 条 ≈ 0.978、第 5 条 neutral ≈ 0.994）。
- 本仓库所有内容都必须是公开可发表的：不要写进任何私人材料、路径或求职上下文。
