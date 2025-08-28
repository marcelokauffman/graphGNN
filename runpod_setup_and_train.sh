#!/usr/bin/env bash
set -euo pipefail

# Simple one-shot setup + training script for a fresh GPU VM (e.g., Runpod)
# - Creates a Python venv
# - Installs CUDA-enabled PyTorch + PyG wheels
# - Optionally downloads dataset from a public Google Drive folder (gdown)
# - Runs the training CLI
#
# Usage:
#   chmod +x scripts/runpod_setup_and_train.sh
#   ./scripts/runpod_setup_and_train.sh \
#       --gdrive-folder-url "https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing" \
#       --data-dir ./data/graphNN \
#       --output-dir ./outputs \
#       --epochs 100

# Defaults
GDRIVE_FOLDER_URL=""
DATA_DIR="./data/graphNN"
OUTPUT_DIR="./outputs"
EPOCHS=100
MARGIN=1.0
LR=1e-3
WEIGHT_DECAY=1e-5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gdrive-folder-url)
      GDRIVE_FOLDER_URL="$2"; shift 2 ;;
    --data-dir)
      DATA_DIR="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --margin)
      MARGIN="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    --weight-decay)
      WEIGHT_DECAY="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo "==> Data dir:      ${DATA_DIR}"
echo "==> Output dir:    ${OUTPUT_DIR}"
echo "==> GDrive URL:    ${GDRIVE_FOLDER_URL:-<none>}"
echo "==> Epochs:        ${EPOCHS}"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

if [[ ! -d .venv ]]; then
  echo "==> Creating Python venv (.venv)"
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Detect GPU availability to choose a torch build
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU detected; installing CUDA build of PyTorch (cu121)"
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
  python -m pip install --extra-index-url "$TORCH_INDEX_URL" \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121
  # PyG wheels matching torch 2.4.1 + cu121
  PYG_INDEX="https://data.pyg.org/whl/torch-2.4.1+cu121.html"
  python -m pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib \
    -f "$PYG_INDEX"
else
  echo "==> No GPU detected; installing CPU builds (training will be slow)"
  python -m pip install torch torchvision torchaudio
  # CPU wheels
  PYG_INDEX="https://data.pyg.org/whl/torch-2.4.1+cpu.html"
  python -m pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib \
    -f "$PYG_INDEX"
fi

python -m pip install torch-geometric
python -m pip install pandas numpy tqdm matplotlib seaborn plotly scikit-learn umap-learn tensorboard wandb gdown

# Optionally fetch dataset from a public Google Drive folder
if [[ -n "$GDRIVE_FOLDER_URL" ]]; then
  echo "==> Downloading dataset from public Google Drive folder"
  mkdir -p "$DATA_DIR"
gdown --fuzzy --no-cookies --folder "$GDRIVE_FOLDER_URL" -O "$DATA_DIR"
fi

echo "==> Verifying required files in $DATA_DIR"
req=(train_pyg_data.pt val_pyg_data.pt test_pyg_data.pt mappings.pkl node_map.csv conversion_summary.json)
missing=()
for f in "${req[@]}"; do
  [[ -f "$DATA_DIR/$f" ]] || missing+=("$f")
done
if (( ${#missing[@]} > 0 )); then
  echo "Missing files in $DATA_DIR:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  echo "Ensure your public Drive folder contains these files." >&2
  exit 2
fi

echo "==> Launching training"
python train_gnn.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --margin "$MARGIN" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY"

echo "==> Done"
