# 傻瓜教程：怎么玩 jev-lab

这份教程假设你**什么都不记得**，每一步都可以直接复制粘贴。
所有命令都在 Apple M4 / 16 GB 上实测过，下面贴的输出是真实输出，不是示意。

---

## 0. 先确认它还在（10 秒自检）

```bash
cd ~/Desktop/work/jev-lab
.venv/bin/python jev.py claim --context samples/demo-context.txt --claim "The interview was moved to Wednesday."
```

看到 `P(ent)=0.9xx` 这类数字就是正常的。

**如果报错**：

- 提示 `can't open file` / `No such file` → 你不在 `jev-lab` 目录里，回去 `cd ~/Desktop/work/jev-lab`
- 提示 `ModuleNotFoundError` → 你用了系统的 `python3`。**一定要用 `.venv/bin/python`**
- 提示模型文件找不到 → 跑 `bash setup.sh` 重新拉权重（约 1.6 GB）

> 关键：本仓库所有的命令都用 `.venv/bin/python`，不用 `python3`。

---

## 1. 第一跑：看它到底在干什么

```bash
.venv/bin/python jev.py claim --context samples/demo-context.txt --claims samples/demo-claims.txt
```

真实输出：

```
上下文 221 字符 · 5 条陈述 · 4785 ms

[1] 支持 (entailment)        P(ent)=0.978 [高]   con 0.002 | ent 0.978 | neu 0.019
    The interview was moved to Wednesday.
[2] 支持 (entailment)        P(ent)=0.518 [中]   con 0.006 | ent 0.518 | neu 0.475
    The recruiter agreed to the new date.
[3] 反驳 (contradiction)     P(ent)=0.051 [低]   con 0.915 | ent 0.051 | neu 0.034
    The candidate cancelled the interview.
[4] 反驳 (contradiction)     P(ent)=0.101 [低]   con 0.876 | ent 0.101 | neu 0.022
    The interview will take place on a weekend.
[5] 没提 / 无关 (neutral)      P(ent)=0.000 [低]   con 0.006 | ent 0.000 | neu 0.994
    The candidate is the strongest applicant for the role.
```

**看什么**：

- 每一行是**一次判断**：`上下文 + 这条陈述` 合起来喂给模型，输出三个概率（contradiction / entailment / neutral）。
- `P(ent)` 高 = 上下文**支持**这条陈述；`con` 高 = 上下文**反驳**它；`neu` 高 = 上下文**根本没提**。
- **第 5 条是这套东西的精华**：一句话听着很合理（"候选人是最强申请人"），但上下文完全没提 → `neutral 0.994`。
  "没提"和"被反驳"是两件事，很多系统会把它们混成一个低分。
- 第 2 条提醒你别把概率当真理：方向对了，但只有 0.518。

---

## 2. 换成你自己的材料

**规则只有一条：上下文是材料，陈述是一句句"能被判真假"的话。**

不要把要求写在上下文里（比如"请你判断……"）——那没用。判断本身就是命令行的参数。

### 方式一：写两个文件（推荐，适合反复改）

```bash
cd ~/Desktop/work/jev-lab
cat > my-context.txt <<'EOF'
把你要审的材料整段粘在这里。
可以是一封邮件、一段对话、一段你写的讲稿。
EOF

cat > my-claims.txt <<'EOF'
这里一行写一条陈述。
# 井号开头的行会被忽略，可以拿来做注释。
每一条都应该是"可以被这段话支持或反驳"的断言。
EOF

.venv/bin/python jev.py claim --context my-context.txt --claims my-claims.txt
```

### 方式二：不建文件，直接问一条

```bash
.venv/bin/python jev.py claim --context "My card was charged twice." --claim "The customer was charged twice."
```

### 方式三：从别处管道进来

```bash
cat my-context.txt | .venv/bin/python jev.py claim --context - --claim "这句话被支持吗"
```

---

## 3. 三个动词各跑一次

这三个动词对应 TypeSafe 官方 Jev 的三种原语（Noul / Choice / Score）。
**这一节是理解"决策模型"最快的方式**：同一段材料，只换问题的形状，答案的形状就变了。

### 3.1 `claim` ↔ Noul：这句话成立吗

```bash
.venv/bin/python jev.py claim --context samples/demo-context.txt --claims samples/demo-claims.txt
```
→ 每条给一个 0–1 的 P(entailment)。这是最接近"是/否"的形状。

### 3.2 `choice` ↔ Choice：从固定选项里选一个

```bash
.venv/bin/python jev.py choice --context samples/demo-context.txt \
  --question "What did the recruiter agree to?" \
  --options "Move the interview to Wednesday|Cancel the interview|Keep the interview on Monday"
```

真实输出：

```
1. P(ent)=0.995 [高]  Move the interview to Wednesday  <- 选它
2. P(ent)=0.029 [低]  Cancel the interview
3. P(ent)=0.011 [低]  Keep the interview on Monday

判决: Move the interview to Wednesday
```

**看什么**：选项是**你给死的**。模型不会凭空造出第四个答案——这正是"固定输出空间"的意思。

### 3.3 `score` ↔ Score：在有序档位上打分

```bash
.venv/bin/python jev.py score --context samples/demo-context.txt \
  --question "How urgent is this email thread?" \
  --levels "not urgent at all|mildly urgent|very urgent"
```

真实输出：

```
  not urgent at all        P(ent)=0.004  归一化后 0.004
  mildly urgent            P(ent)=0.525  归一化后 0.451
  very urgent              P(ent)=0.635  归一化后 0.545

判决: very urgent
```

**看什么**：**它几乎没分开**（0.451 vs 0.545）。这不是 bug——`score` 是我用 NLI 硬拼的，不是模型的原生能力。
这个失败本身就是结论：TypeSafe 为什么要把 Score 做成**原生原语**，而不是让用户拿 NLI 自己拼。
**所以别把这个 score 的结果当真。**

---

## 4. 四个实验（每个都有明确目的）

### 实验 1：措辞敏感度——概率不是"相似度"

同一个前提，换五种说法：

```bash
.venv/bin/python jev.py claim --context "The interview was moved to Wednesday." \
  --claim "The interview is on Wednesday." \
  --claim "The interview is not on Wednesday." \
  --claim "The interview was rescheduled." \
  --claim "The interview time changed." \
  --claim "The interview was moved to Thursday."
```

真实输出：

```
[1] 支持 (entailment)    P(ent)=0.924    The interview is on Wednesday.
[2] 反驳 (contradiction) P(ent)=0.006    The interview is not on Wednesday.
[3] 支持 (entailment)    P(ent)=0.799    The interview was rescheduled.
[4] 支持 (entailment)    P(ent)=0.766    The interview time changed.
[5] 反驳 (contradiction) P(ent)=0.003    The interview was moved to Thursday.
```

**收获**：[1] 和 [3] 说的是同一件事，分数差了 0.12。所以这个数字**不是"语义相似度"**，
它是"这条声明是否能从材料里推出来"的估计——换个说法就换个数。任何拿它当阈值用的人都会踩空。

### 实验 2：中文到底行不行（自己测，别听说）

```bash
.venv/bin/python jev.py claim --context samples/demo-context.zh.txt --claims samples/demo-claims.zh.txt
```

真实输出：

```
[1] 支持 (entailment)    P(ent)=0.975    面试改到了周三。
[2] 支持 (entailment)    P(ent)=0.922    招聘官同意了这个新时间。
[3] 反驳 (contradiction) P(ent)=0.016    候选人取消了面试。
[4] 反驳 (contradiction) P(ent)=0.007    面试会在周末进行。
[5] 没提 / 无关 (neutral) P(ent)=0.000    候选人是这个岗位最强的申请人。
```

**收获**：中文这组**比英文那组还干脆**（第 2 条 0.922 vs 英文的 0.518）。
TypeSafe 官方文档说 Jev 以英文为主、CJK 精度更低——**那是关于 Jev 的说法，我们验证不了**；
在本地这个开源复刻上，玩具例子没体现出"中文更差"。
结论：**别把"中文更差"当已知事实，用你自己的材料各测一遍。**

### 实验 3：找"没提"和"被反驳"的差别

在上面任何一次 `claim` 里，故意加一条**材料里根本没提、但听起来合理**的说法
（比如"候选人是这个岗位最强的申请人"）。看它落在 `neutral` 还是 `contradiction`。
前者是"材料没说"，后者是"材料说了相反的"。这个区分是做证据审查时最有用的一格。

### 实验 4：拿它审你自己的中文/英文材料

把你写过的一段文字当上下文，把你对它做过的**主张**一条条写成陈述：

```bash
.venv/bin/python jev.py claim --context 你的材料.txt --claims 你的主张.txt
```

被判成 `neutral` 的那些 = **你讲了，但材料里没有支撑**的地方。
这是这套工具对写作者最实际的一个用法。

---

## 5. 排错与注意事项

- **一条判断约 0.7–0.9 秒**（CPU 即可，不需要 GPU）。要跑很多条时**一次传多条**，不要写 `for` 循环反复启动进程——每次启动都要重新加载模型（约 2 秒）。
- 加载时打印的两条 warning（`causal_conv1d` / `flash-linear-attention` 回退到 PyTorch 参考实现）是**正常的**，只是慢一点。
- 想看机器可读结果，加 `--json`。
- 换更强的 checkpoint：先 `JEV_SUBFOLDER=qwen3.5-4b-nli-v2 bash setup.sh`（9 GB），再用 `--subfolder qwen3.5-4b-nli-v2`。
- 换设备：`--device mps`（Mac 上稍快；实测判决与 `cpu` 完全一致）。

### 千万别做这几件事

- **别把它当 Jev。** 它是 NLI 三分类 cross-encoder，用普通交叉熵训练，**没有** Jev 声称的 RLCD 校准。
  概率就是 softmax 输出。
- **别拿概率当阈值。** 上面 `[高]/[中]/[低]` 三档是抄官方文档的**用法示范**，不是校准过的门槛。
- **别看 `score` 的结论**（见 §3.3）。
- 本工具完全本地运行，不上传任何东西——但反过来也意味着它的判断**不代表 Jev 的判断**。

---

## 6. 参数速查

| 参数 | 作用 |
| --- | --- |
| `--context` | 文件路径 / `-`（读 stdin）/ 内联文本 |
| `--claims` | 一行一条陈述的文件（`#` 开头忽略） |
| `--claim` | 内联一条陈述，可重复 |
| `--question` `--options` | `choice` 用，选项用 `|` 分隔 |
| `--levels` | `score` 用，从低到高用 `|` 分隔 |
| `--json` | 机器可读输出 |
| `--device cpu\|mps` | 默认 cpu |
| `--dtype bf16\|fp32` | 默认 bf16 |
| `--subfolder` | 换 checkpoint |

---

## 7. 如果你用 Coding Agent 来玩

把这个仓库交给 Codex 之类的编码 Agent 时，先让它读 [`../AGENTS.md`](../AGENTS.md)——
那里写了怎么调用、输出怎么读、以及哪些结论不许乱说。
然后可以直接下这样的指令：

> 用 `jev.py` 的 `claim` 模式，把 `<某文件>` 当作上下文，把我列出的这些主张逐条判定，
> 把 `neutral` 和 `contradiction` 的分开列出，并说明哪一条最值得我回去补证据。

背景与证据（包括厂商声称、第三方实测、以及"哪些是未公开的"）见 [`jev-notes.md`](jev-notes.md)。
