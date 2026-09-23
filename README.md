# jev-lab

> **English** — A small local playground for *decision models* in the shape of TypeSafe's **Jev**.
> It runs the open replication [`openjev`](https://huggingface.co/AlexWortega/openjev) (Qwen3.5
> fine-tuned as an NLI cross-encoder) entirely on your own machine — CPU is enough — and exposes
> one verb per TypeSafe primitive: `claim` ↔ Noul, `choice` ↔ Choice, `score` ↔ Score.
> No API key, no waitlist, no cloud. Background and evidence: [`docs/jev-notes.md`](docs/jev-notes.md).

---

## 这是什么

2026-09-15，TypeSafe 发布 **Jev** 和它背后的类别名 **System One Model**：不生成文本，只接收一个
state 和一组 typed questions，并行地返回**类型化取值 + 概率**。

Jev 的官方 Playground 要排队，所以我用开源复刻 `openjev` 在本地复现了这个**形状**，用来亲手感受
"给一段上下文 → 让它做判断 → 拿到概率"这件事，以及观察概率本身长什么样。

**这不是 Jev。** `openjev` 是一个 NLI 三分类 cross-encoder
（contradiction / entailment / neutral），用普通交叉熵训练，**没有** Jev 声称的 RLCD 校准。
它给的是形状和手感，不是那个模型的实力。判断依据写在 [`docs/jev-notes.md`](docs/jev-notes.md) §2。

## 快速开始

```bash
git clone https://github.com/eyesofish/jev-lab.git
cd jev-lab
bash setup.sh    # 拉建模代码 + 0.8B 权重（约 1.6 GB）+ 建 .venv

.venv/bin/python jev.py claim --context samples/demo-context.txt --claims samples/demo-claims.txt
```

需要 `uv`（推荐）或 `python3`；不需要 GPU。

## 三个动词 ↔ 三种原语

| 动词 | 对应官方原语 | 做什么 | 输出 |
| --- | --- | --- | --- |
| `claim` | **Noul** | 一段上下文 + 若干条陈述，逐条判定 | 每条的三分类概率 + P(entailment) |
| `choice` | **Choice** | 从固定选项里选一个 | 每项 P(entailment)，取最大 |
| `score` | **Score** | 在有序档位上打分 | ★ 每档当一条陈述，再对 P(entailment) 归一化 |

★ `score` 是**用 NLI 硬拼的**，不是模型的原生能力——而且它**明显不好用**（实测 0.545 vs 0.451，
几乎分不开）。这个失败本身就是结论：官方把 Score 做成原生原语是有理由的。
三种原语，在只有一条 NLI 能力的模型上，靠的是"问题怎么写 + 代码怎么组合"
——官方文档管这叫 *atomic questions composed in code*。

```bash
# Noul 形状
.venv/bin/python jev.py claim --context samples/demo-context.txt --claims samples/demo-claims.txt

# Choice 形状
.venv/bin/python jev.py choice --context samples/demo-context.txt \
  --question "What did the recruiter agree to?" \
  --options "Move the interview to Wednesday|Cancel the interview|Keep the interview on Monday"

# Score 形状（拼的）
.venv/bin/python jev.py score --context samples/demo-context.txt \
  --question "How urgent is this email thread?" --levels "not urgent at all|mildly urgent|very urgent"

# 裸接口
.venv/bin/python jev.py predict --premise "A man is playing a guitar." --hypothesis "Someone is making music."
```

`--context` 可以是文件路径、`-`（读 stdin）或内联文本；`--claims` 是"一行一条陈述"的文件。
其他开关：`--device cpu|mps`、`--dtype bf16|fp32`、`--subfolder`（换 checkpoint）、`--json`。

## 真实输出（Apple M4 / 16 GB，2026-09-22）

```
[1] 支持 (entailment)    P(ent)=0.978    The interview was moved to Wednesday.
[2] 支持 (entailment)    P(ent)=0.518    The recruiter agreed to the new date.
[3] 反驳 (contradiction) P(ent)=0.051    The candidate cancelled the interview.
[4] 反驳 (contradiction) P(ent)=0.101    The interview will take place on a weekend.
[5] 没提 / 无关 (neutral) P(ent)=0.000    The candidate is the strongest applicant for the role.
```

第 5 条最值得注意：一个**看似合理、但上下文完全没提**的说法被判成 neutral 0.994。
`choice` 那次，正确答案拿到 P(ent)=0.995，两个干扰项 0.029 / 0.011。

- 模型加载 2.1 s；单次判断约 **0.7–0.9 s**（CPU 即可）
- cpu / mps、bf16 / fp32 四种组合，判决**完全一致**
- 磁盘占用：权重 1.6 GB + 环境 0.7 GB

想更强：`bash setup.sh` 前设 `JEV_SUBFOLDER=qwen3.5-4b-nli-v2`（9 GB），
或用 `--subfolder qwen3.5-4b-nli-v2`。

## 局限（别自我欺骗）

- 这是 **NLI 形状**（两句对 → 3 类），不是 `state + 多个 typed questions` 那套接口；
  多问题是 `jev.py` 里循环拼出来的。
- **没有校准**：概率就是 softmax，`jev.py` 里"高/中/低"三档是抄官方文档的**用法示范**，
  不是可信阈值。
- 训练主力语言是英文：中文材料能传，但别指望准。
- 0.8B 很小，"方向对但只有 0.518"这种模糊判断是常态。

## 结构

```
jev-lab/
├── jev.py                 # claim / choice / score / predict
├── setup.sh               # 拉建模代码 + 权重 + 建 venv
├── requirements.txt
├── samples/               # 一段上下文 + 五条陈述（含一条"看似合理但无依据"的陷阱）
└── docs/jev-notes.md      # 研究笔记：Jev 是什么、概率怎么做出来的、争议与来源
```

## 致谢与许可

本仓库**不分发**模型权重，也不内联上游建模代码：`setup.sh` 在运行时从
[`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev) 拉取，
该模型仓库以 MIT 许可发布，模型、训练数据与 `modeling_openjev.py` 的功劳全归其作者。
本仓库自身代码以 MIT 许可发布，见 [`LICENSE`](LICENSE)。

TypeSafe、Jev、System One Model 是 TypeSafe AI 的商标与命名，本仓库与其无隶属关系。
