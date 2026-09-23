# Jev / System One Model — 研究笔记

> 写于 2026-09-22。凡是厂商自称、第三方实测、以及我自己的推断，都在文中标出来。
>
> **证据分级**：**【厂商】** TypeSafe 官方自称，未经独立验证 · **【独立】** 第三方实测 ·
> **【代码】** 从开源复刻的源码里读到的事实 · **【推断】** 推理，不是事实 · **【未公开】** 厂商明确没有披露。

---

## 0. 这份笔记要回答什么

1. Jev 到底是个什么东西？
2. 它"判断概率"这件事，到底是怎么做成的？

**先纠正一个直觉**：Jev 不是 RAG 的同类替代品，也不是检索器或 embedding 模型。它是**决策层**
——把"待判断的对象"变成"带概率的类型化答案"。它和 RAG 互补，落在 RAG 里"重排、路由、判定"那一段（§1.6）。

---

## 1. Jev 是什么

### 1.1 一句话

Jev 是 TypeSafe AI 的第一个 **System One Model**：调用方给一个 **state**（非结构化上下文）和一组
**typed questions**，模型**并行**地逐题返回**类型化取值 + 概率分布**，全程不生成自由文本。【厂商】

### 1.2 出身与命名

- **发布**：2026-09-15。【厂商】
- **公司**：TypeSafe AI，创始人 **Diogo Almeida**，自述在 OpenAI 参与了让语言模型"会遵循指令"的
  方法研究，那批工作最终成为 ChatGPT 背后的研究；此后两年 stealth 做出这个模型。【厂商】
- **名字**：取自 **William Stanley Jevons**（杰文斯），典故是效率提升不减少需求、反而放大需求。
  所以 `Jev` **不是缩写**，是模型名；缩写的是它背后的类别名 System One Model。【厂商】

### 1.3 它定义的类别：System One Model

借卡尼曼 System 1 / System 2 的说法命名：只做"快、直觉式的判定"，不做慢推理。核心主张是一条交易——

> **放弃字符串生成，换取一个固定的输出空间。**

后果：没有"格式解析失败"这类错误，因为输出空间在调用前就被 schema 定死了；代价是模型只能回答
你声明过的那类问题。【厂商】

这是**厂商造词**，不是学界既定类别；已有评论者指出它命名的是"速度"而不是"类别"。【独立】

### 1.4 三种原语

官方暴露三个 AI primitives，一次调用里可以混用，每题针对同一个 state **并行且相互隔离**地求值；
官方称加题几乎不增加响应时间，也不产生 context rot。【厂商】

| 原语 | 问题形式 | 返回 |
| --- | --- | --- |
| **Choice** | 从列表里选一个 | `choice` + `probabilities` + `confidence` |
| **Score** | 按 rubric 打分 | `score` + `probabilities` + `confidence` |
| **Noul** | 这句话是不是真的？ | `noul`（0–1 的单一值） |

使用规矩：**每题只问一件几秒钟能判断完的小事**，多维度拆成多题，然后在你自己的代码里加权组合。
官方理由是：这样每个判断更可靠，权重变了改代码里的系数，而不是重写 prompt。【厂商】

### 1.5 和 chat 形 LLM 的差别

| 维度 | Chat 形 LLM | System One Model |
| --- | --- | --- |
| 输出 | 字符串，下游还要 parse + validate | 类型化取值，schema 调用前固定 |
| 失败模式 | 幻觉、类型错误、拒答 | 被约束在声明的输出空间里 |
| 采样 | 自回归、顺序逐 token | 并行，一次调用回答所有问题 |
| 置信度 | 通常没有；提示了也不可靠 | 概率分布，可卡阈值 |

### 1.6 它不是 RAG 的替代品

RAG 是"检索 + 生成文本"，Jev 是"直接判定"。Jev 在 RAG 管线里的位置，是那几个**原本让 LLM 当裁判的环节**：

- **重排**——官方 rerank cookbook 的做法是每对 query-候选问一个 Noul，然后按概率排序；
  官方原话是这里**用概率本身，而不是用阈值**。
- **路由 / 分类**——把用户意图压成 Choice。
- **证据与幻觉判定**、相关性评估、结构恢复、逐行检索等。

### 1.7 接入与开源复刻

- **闭源**：typesafe.ai waitlist；2026-09-16 起也可从 Vercel AI Gateway 以 `typesafe-ai/jev` 调用
  （转述自第三方博客，本人未验证其可用性与计费）。官方文档同时提供 `console.typesafe.ai/playground`
  这个 Playground 入口——**实测在 2026-09-22 显示已满，用不上**。
- **开源复刻（与 TypeSafe 无关）**：
  - [`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev)（MIT）：Qwen3.5 训练成 NLI
    cross-encoder，多个 checkpoint，含 demo Space。
  - [`ZefanCai/Open-Jev-2B`](https://huggingface.co/ZefanCai/Open-Jev-2B)：LoRA adapter + decision head，
    需要配钉住的 Qwen 基座权重和对应代码。

---

## 2. 它"判断概率"是怎么做成的

每一层能拿到的证据强度完全不同，必须分开说。

### 2.1 把"判断"压成有限选项 —— 有据

模型不是先自由理解再"报个概率"，而是**问题本身就规定了答案空间**：Choice 给候选列表，Score 给
rubric 档位，Noul 给 true / false 两档外加判定标准。于是"判断"退化成**在固定候选集合上的分布问题**
——这是 softmax 之类归一化能给出概率的前提。【推断，接口设计的直接推论】

### 2.2 非自回归 + 并行 —— 部分有据

官方明确说每题并行、相互隔离、加题几乎不增加响应时间。第三方把它描述为 **non-autoregressive**：
输出决策，而不是逐 token 预测。【厂商 + 独立】

推论：既然不逐 token 生成，概率就不来自"生成文本的 token 概率"，而来自对**声明好的选项集合**的打分。

**这里是最大的未知**：TypeSafe 没有公开架构，也没有公开训练算法。【未公开】

### 2.3 `confidence` 是分布的统计量 —— 有据（官方原文）

- 每个 Choice / Score 答案都带 `probabilities`，即跨选项（或跨档位）的概率分布。
- 分布的**形状**就是确定性：集中 = 自信，摊平 = 不确定。
- `confidence` 是把分布形状压成一个 0–1 的数，方便直接卡阈值；官方明说这只是"方便的默认值"，
  你完全可以用完整 `probabilities` 自己算别的度量。
- `Noul` 答案**不带** confidence（它本身就是 0–1 的值）。
- 官方推荐三段式用法：高置信 → 自动执行；中置信 → 先确认或补信息；低置信 → 不动，转人工。
  阈值随风险分级——同一个系统里，"查余额"和"批准转账"该用不同的门槛。

### 2.4 训练目标 RLCD —— 目标有据，算法未公开

TypeSafe 说 Jev 用自研训练法 **RLCD（Reinforcement Learning for Calibrated Decisions）**，
优化目标是"**认知上诚实的概率**"：说 0.8 的事情，大约 80% 要真的成立。【厂商】

| 方法 | 优化什么 | 产出什么 |
| --- | --- | --- |
| **RLHF** | 人类偏好（评分者喜欢） | 流畅讨喜的文本；置信度听起来对而已 |
| **RLVR** | 可验证奖励（有 checker 的正确答案） | 在有 checker 的地方正确 |
| **RLCD** | 校准的决策 | 概率的数值就是它的字面意思 |

官方 primer 给的论证：偏好优化容易像 GAN 一样 mode collapse——生成器学会反复产出"能骗过判别器"
的那类输出；RLHF 仍然适合对话模型，只是"生产环境的自动化需要另一个训练目标：受约束的决策 +
校准的不确定性"。【厂商】

**校准本身**是通用机器学习概念，不是 TypeSafe 发明的：常见度量是可靠性图（预测 0.8 的那批样本
实际正确率是多少）、ECE、Brier score；常见后处理修正是 temperature scaling、Platt / isotonic。
RLHF 之后模型置信度普遍不可信，正是这一领域存在的理由。【通用知识，非本文来源】

**算法细节依然未公开。**【未公开】

### 2.5 独立推断：它可能是什么

Sebastian Raschka 的中间立场值得完整保留【推断】：

- "很容易把 Jev 贬成'不就是个分类器'"，他认为真相在两端之间。
- 猜测：一个小的 **encoder 类模型**（如 ModernBERT）+ 类似校准奖励 RL 的训练
  （他引的是 *Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty*，arXiv 2507.16806）。
- 关键判断：**"独门配方更可能在数据里，而不在算法里"**，再加一个设计得不错的 API。

### 2.6 开源复刻里的机制 —— 代码级，不是猜的

`openjev` 的 `modeling_openjev.py` 把这层写得很清楚：【代码】

- 基座是 Qwen3.5 解码器（0.8B：24 层、hidden 1024、每 4 层插入一个 full attention，其余是
  linear attention），上面接一个**线性 `score` 头，取最后一个非 pad token 的隐藏状态**。
- 输入是把两句话套进模板 `"Premise: {premise}\nHypothesis: {hypothesis}"`，右填充。
- **损失就是三分类交叉熵**；`predict()` 就是 `softmax(logits)`——**没有别的魔法**。
- 它还实现了 `predict_hypotheses(premise, [h1, h2, ...])`：一次 prefill 共享前缀，再把 KV cache
  分支到每个假设上。这是官方"多问题并行、加题几乎不增加耗时"在复刻里的对应物。
- 另一条独立路线 `LatentMLPHead`：冻结 cross-encoder，只用一个 MLP（d→512→1，GELU，dropout 0.1）
  在 latent 上做逐选项打分，损失是 soft BCE。这跟"分类头 + softmax"是两种不同的取数方式。

这就把 §2.1–§2.3 的推断坐实了一半：**概率确实来自固定输出空间上的 softmax**，难的不是概率，
是训练目标。

**最重要的结论**：这些复刻**都没有复现 RLCD**——它们是"把基座模型改造成决策接口"，而不是
"复现那套校准训练"。所以**不要把开源复刻的能力当作 Jev 的能力**。【推断】

### 2.7 实测证据与争议

**独立实测（搜索场景）**：Hev 于 2026-09-16 发布，在三个 BEIR 子集上把 Jev 当 reranker——
零重排训练、零语料调优，每题对每个候选问一个 Noul——mean nDCG@10 距最好的专用 hosted reranker
只差 **0.003**，并且额外给出**每篇文档的校准概率**（托管 reranker 不给这个）。作者随后补测了
Mixedbread 的 listwise 模型 `mxbai-rerank-v3.1-listwise`，它在 NFCorpus 和 FiQA 上**明显强于 Jev**
（置信区间不含 0）。也就是说：Jev 的故事不是"最强"，而是"**不训练就接近专业模型**"。【独立】

**厂商自家 cookbook（CLERC 法律检索，3,565 段落 / 40 query）**：有两种读法，都要留着——

- 读法 A（宣传口吻）：短名单后重排把 top-1 从约 5% 提到 **18%**。（这个 5% 来自二手博客转述，
  **没有核到 notebook 原始数字**。）
- 读法 B（知识库笔记的读法）：更值得注意的是**上限**——短名单里**100% 含有**正确段落，
  重排却只有 **18%** 把它排到第一。

**批评面**：

- **"不会幻觉"是个更窄的声明**：Hacker News 最高赞评论和 The Register（2026-09-16）都指出同一点
  ——它发不出**非法类型**，但照样能发出**错误却合法**的值。
- **类别是造词**：有人认为 System One Model 命名的是速度，不是类别。
- **"just a classifier"**：见 §2.5。
- **社区质疑**：r/LocalLLaMA 上有 "I really don't understand Jev hype" 这类帖子。正面引用也有——
  Daniel Tunkelang 用一天后说"前沿模型级质量、成本不到 1%"，但同一句话里他念出了限制：
  "**只能做分类和回归**"。

---

## 3. 还没搞清的问题

- 概率到底是"分类头 softmax"，还是"某几个 token 的 logprob 归一化"？厂商未公开，只能从开源复刻反推。
- **校准是对哪个分布校准？**同一模型在不同任务上的校准度是否一致？"calibrated to what?" 这个追问
  目前无人回答。
- 概率能跨 query 比较吗？重排只需**同一 query 内**可排序，而"卡阈值"要求**跨样本可比**——这两件事
  要求不一样，很容易被混为一谈。
- 那 18% 的原始 notebook 数字值得自己核一遍。

---

## 4. 来源

**一手（厂商）**

- 发布公告《Introducing System One Models & Jev》，2026-09-15：https://typesafe.ai/blog/introducing-system-one-models-and-jev
- 文档·介绍与三原语：https://docs.typesafe.ai/introduction
- 文档·置信度：https://docs.typesafe.ai/confidence
- 文档·Noul：https://docs.typesafe.ai/primitives/noul
- 文档·state 怎么写：https://docs.typesafe.ai/concepts/state
- 文档·重排 cookbook：https://docs.typesafe.ai/cookbooks/rerank_typesafe
- 文档·AI primer：https://docs.typesafe.ai/introduction/machine-learning-primer
- 文档·API reference：https://docs.typesafe.ai/api

**独立测量与评论**

- Hev《Jev as a reranker》，2026-09-16：https://hevmind.com/writing/jev-as-a-reranker/
- Sebastian Raschka《It's Easy to Dismiss Jev as Just a Classifier》：https://sebastianraschka.com/blog/2026/jev-classification-generalization.html
- The Register，2026-09-16：https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711
- awesome-search 知识库（Jev 词条、System One Model、RLCD、Reception of Jev、Hev meets Jev 等）：
  https://frutik.github.io/awesome-search/Tools/Jev

**开源复刻**

- openjev（HuggingFace，NLI cross-encoder，MIT）：https://huggingface.co/AlexWortega/openjev
- openjev demo Space：https://huggingface.co/spaces/AlexWortega/openjev
- Open-Jev 项目页与中文构建说明：https://zefan-cai.github.io/open-jev/ · https://zefan-cai.github.io/open-jev/story/
- Open-Jev 权重：https://huggingface.co/ZefanCai/Open-Jev-2B

---

## 5. 在本地亲手验证

上面 §2.6 的机制结论、以及 §2.7 里"概率不可当阈值"的感受，都可以在本仓库直接复现：
见 [`../README.md`](../README.md)。特别建议试这一条——把一段上下文和几条陈述喂进去，
看那些"看似合理但上下文没提"的说法会被判成什么。
