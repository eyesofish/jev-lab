#!/usr/bin/env bash
# 拉取 openjev 的建模代码与权重，并建好 Python 环境。
# 可重复执行：已存在的文件会跳过。
set -euo pipefail
cd "$(dirname "$0")"

REPO="AlexWortega/openjev"
SUBDIR="${JEV_SUBFOLDER:-qwen3.5-0.8b-nli-v2s-long}"
BASE="https://huggingface.co/$REPO/resolve/main"
DEST="model"

mkdir -p "$DEST/$SUBDIR"

echo "[1/3] 下载建模代码 -> $DEST/"
for f in modeling_openjev.py modeling_qwen35_moe_seqcls.py; do
  if [ -f "$DEST/$f" ]; then echo "  跳过 $f（已存在）"; else
    curl -fSL --retry 3 -o "$DEST/$f" "$BASE/$f"; echo "  $f"
  fi
done

echo "[2/3] 下载权重 -> $DEST/$SUBDIR/（0.8B 约 1.6 GB，4B 约 9 GB）"
for f in config.json tokenizer.json tokenizer_config.json preprocessor_config.json \
         video_preprocessor_config.json model.safetensors; do
  if [ -f "$DEST/$SUBDIR/$f" ]; then echo "  跳过 $f（已存在）"; else
    curl -fSL --retry 3 -o "$DEST/$SUBDIR/$f" "$BASE/$SUBDIR/$f"; echo "  $f"
  fi
done

echo "[3/3] 建立 Python 环境 (.venv)"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
else
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

cat <<'EOF'

完成。跑第一条命令：

  .venv/bin/python jev.py claim --context samples/demo-context.txt --claims samples/demo-claims.txt

想看研究笔记：docs/jev-notes.md
EOF
