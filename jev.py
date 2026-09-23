#!/usr/bin/env python3
"""jev.py —— 在本地玩 openjev（Jev 的开源复刻形态）。

三种"判断"动词，一一对应 TypeSafe 文档里的三种原语：

    claim   ↔ Noul    （这句话成立吗？→ P(entailment)，一个 0–1 的数）
    choice  ↔ Choice  （从固定选项里选一个 → 每项 P(entailment)，取最大）
    score   ↔ Score   （在有序档位上打分 → 每档当一条陈述，再对 P(entailment) 归一化）★

★ score 这个动词是**我自己用 NLI 拼出来的**，不是模型的原生能力。
  它的存在本身说明一件事：TypeSafe 的三种原语，在一个只有一条 NLI 能力的
  模型上，是靠"问题怎么写 + 代码怎么组合"造出来的——这正是官方文档说的
  "atomic questions composed in code"。

关于概率，必须记住（详见 docs 里的研究笔记）：
  * 这个模型的概率就是 **三分类 softmax 的输出**，用普通交叉熵训练的，
    没有 TypeSafe 声称的 RLCD 校准。
  * 所以下面打的"高/中/低"只是复刻官方文档里的一种**用法示范**，
    不是"校准过的可信阈值"。别拿它当生产门槛。

用法示例：

    # Noul 形状：一段上下文 + 若干条陈述
    python jev.py claim --context samples/demo-context.txt --claims samples/demo-claims.txt

    # Choice 形状：从固定选项里选一个
    python jev.py choice --context samples/demo-context.txt \
        --question "What did the recruiter agree to?" --options "Move to Wednesday|Cancel the interview|Keep Monday"

    # Score 形状（我拼的）
    python jev.py score --question "How urgent is this message?" --context "..." --levels "not at all|mildly|very"

    # 裸接口
    python jev.py predict --premise "A man is playing a guitar." --hypothesis "Someone is making music."
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT / "model"
SUBFOLDER = "qwen3.5-0.8b-nli-v2s-long"

LABELS = ["contradiction", "entailment", "neutral"]
CON, ENT, NEU = 0, 1, 2

_model = None


def load(device: str, dtype: str, subfolder: str = SUBFOLDER):
    """加载一次，复用。device: cpu / mps；dtype: bf16 / fp32；subfolder 选 checkpoint。"""
    global _model
    if _model is None:
        import torch
        from modeling_openjev import OpenJevCrossEncoder
        dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        t0 = time.time()
        _model = OpenJevCrossEncoder(str(REPO), subfolder=subfolder, device=device, dtype=dt)
        print(f"[load] {subfolder} on {device}/{dtype} — {time.time() - t0:.1f}s", file=sys.stderr)
    return _model


def read_text(value: str | None) -> str:
    """--context 可以是内联文本、'-'（stdin）、或一个存在的文件路径。"""
    if value is None:
        return ""
    if value == "-":
        return sys.stdin.read()
    p = pathlib.Path(value)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return value


def read_lines(value: str | None, inline: list[str]) -> list[str]:
    """--claims 可以是一个文件（一行一条，# 开头忽略），也可以多次 --claim 内联。"""
    lines: list[str] = []
    if value:
        p = pathlib.Path(value)
        raw = p.read_text(encoding="utf-8") if p.exists() and p.is_file() else value
        lines += [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    lines += [x.strip() for x in inline if x.strip()]
    return lines


def verdict(row) -> str:
    """把三分类概率读成人话：支持 / 反驳 / 没提。"""
    con, ent, neu = float(row[CON]), float(row[ENT]), float(row[NEU])
    if ent >= max(con, neu):
        return "支持 (entailment)"
    if con >= neu:
        return "反驳 (contradiction)"
    return "没提 / 无关 (neutral)"


def band(ent: float) -> str:
    """复刻官方文档的三段式用法。注意：这里不是校准过的阈值。"""
    return "高" if ent >= 0.8 else ("中" if ent >= 0.5 else "低")


def emit(payload: dict, as_json: bool, human: str):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(human)


def cmd_predict(a, model):
    t0 = time.time()
    rows = model.predict([(a.premise, a.hypothesis)])
    row = rows[0]
    dt = (time.time() - t0) * 1000
    human = (
        f"前提: {a.premise}\n假设: {a.hypothesis}\n"
        f"  contradiction {row[CON]:.3f} | entailment {row[ENT]:.3f} | neutral {row[NEU]:.3f}\n"
        f"  判决: {verdict(row)}   P(entailment)={row[ENT]:.3f} [{band(float(row[ENT]))}]\n"
        f"  ({dt:.0f} ms)"
    )
    emit({"probabilities": {l: float(v) for l, v in zip(LABELS, row)}, "ms": round(dt)},
         a.json, human)


def cmd_claim(a, model):
    context = read_text(a.context)
    claims = read_lines(a.claims, a.claim)
    if not claims:
        sys.exit("claim: 需要 --claims <文件> 或至少一条 --claim")

    t0 = time.time()
    rows = model.predict([(context, c) for c in claims])
    dt = (time.time() - t0) * 1000

    out = []
    lines = [f"上下文 {len(context)} 字符 · {len(claims)} 条陈述 · {dt:.0f} ms\n"]
    for i, (c, row) in enumerate(zip(claims, rows), 1):
        lines.append(
            f"[{i}] {verdict(row):22s} P(ent)={row[ENT]:.3f} [{band(float(row[ENT]))}]"
            f"   con {row[CON]:.3f} | ent {row[ENT]:.3f} | neu {row[NEU]:.3f}\n    {c}"
        )
        out.append({"claim": c, "verdict": verdict(row),
                    "probabilities": {l: float(v) for l, v in zip(LABELS, row)}})
    emit({"mode": "claim", "ms": round(dt), "results": out}, a.json, "\n".join(lines))


def cmd_choice(a, model):
    context = read_text(a.context)
    options = [o.strip() for o in a.options.split("|") if o.strip()]
    if len(options) < 2:
        sys.exit("choice: --options 至少两项，用 | 分隔")
    premise = (context + "\n\n" + a.question).strip() if context else a.question
    hyps = [a.hyp_format.format(o) for o in options]

    t0 = time.time()
    rows = model.predict([(premise, h) for h in hyps])
    dt = (time.time() - t0) * 1000

    order = sorted(range(len(options)), key=lambda i: -float(rows[i][ENT]))
    lines = [f"{len(options)} 个选项 · {dt:.0f} ms · 按 P(entailment) 排序\n"]
    out = []
    for rank, i in enumerate(order, 1):
        mark = "  <- 选它" if rank == 1 else ""
        lines.append(
            f"{rank}. P(ent)={rows[i][ENT]:.3f} [{band(float(rows[i][ENT]))}]  {options[i]}{mark}\n"
            f"     con {rows[i][CON]:.3f} | ent {rows[i][ENT]:.3f} | neu {rows[i][NEU]:.3f}"
        )
        out.append({"option": options[i], "probabilities": {l: float(v) for l, v in zip(LABELS, rows[i])}})
    lines.append(f"\n判决: {options[order[0]]}")
    emit({"mode": "choice", "ms": round(dt), "picked": options[order[0]], "results": out},
         a.json, "\n".join(lines))


def cmd_score(a, model):
    context = read_text(a.context)
    levels = [l.strip() for l in a.levels.split("|") if l.strip()]
    if len(levels) < 2:
        sys.exit("score: --levels 至少两档，用 | 分隔")
    premise = (context + "\n\n" + a.question).strip()
    hyps = [a.statement.format(level=l) for l in levels]

    t0 = time.time()
    rows = model.predict([(premise, h) for h in hyps])
    dt = (time.time() - t0) * 1000

    ents = [float(r[ENT]) for r in rows]
    total = sum(ents) or 1.0
    norm = [e / total for e in ents]
    best = max(range(len(levels)), key=lambda i: norm[i])

    lines = ["每档当作一条陈述，先取 P(entailment)，再对档位归一化（★ 我拼的，非原生）：\n"]
    out = []
    for i, lv in enumerate(levels):
        lines.append(f"  {lv:24s} P(ent)={ents[i]:.3f}  归一化后 {norm[i]:.3f}")
        out.append({"level": lv, "p_entailment": round(ents[i], 4), "normalized": round(norm[i], 4)})
    lines.append(f"\n判决: {levels[best]}   （{dt:.0f} ms）")
    emit({"mode": "score", "ms": round(dt), "picked": levels[best], "results": out},
         a.json, "\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Local openjev playground (Jev-style decisions, open replication).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--subfolder", default=SUBFOLDER,
                    help="checkpoint 目录名，例如 qwen3.5-0.8b-nli-v2s-long / qwen3.5-4b-nli-v2")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="裸 NLI：一条前提 + 一条假设")
    p.add_argument("--premise", required=True)
    p.add_argument("--hypothesis", required=True)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("claim", help="↔ Noul：一段上下文 + 若干条陈述")
    p.add_argument("--context", default=None, help="文件路径 / '-' 读 stdin / 内联文本")
    p.add_argument("--claims", default=None, help="一行一条陈述的文件")
    p.add_argument("--claim", action="append", default=[], help="内联一条陈述，可重复")
    p.set_defaults(fn=cmd_claim)

    p = sub.add_parser("choice", help="↔ Choice：从固定选项里选一个")
    p.add_argument("--context", default=None)
    p.add_argument("--question", required=True)
    p.add_argument("--options", required=True, help="用 | 分隔")
    p.add_argument("--hyp-format", default="The correct answer is: {}")
    p.set_defaults(fn=cmd_choice)

    p = sub.add_parser("score", help="↔ Score：在有序档位上打分（我拼的）")
    p.add_argument("--context", default=None)
    p.add_argument("--question", required=True)
    p.add_argument("--levels", required=True, help="从低到高，用 | 分隔")
    p.add_argument("--statement", default="The correct rating is: {level}")
    p.set_defaults(fn=cmd_score)

    a = ap.parse_args()
    sys.path.insert(0, str(REPO))
    a.fn(a, load(a.device, a.dtype, a.subfolder))


if __name__ == "__main__":
    main()
