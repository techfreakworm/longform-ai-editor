#!/usr/bin/env bash
# Download Hugging Face model weights for long-form-editor.
#
# Usage:
#   ./scripts/download_models.sh                # core only (default stack, ~41 GB)
#   ./scripts/download_models.sh --with-qwen3-small   # + Qwen3-30B-A3B MoE fast fallback (~17 GB)
#   ./scripts/download_models.sh --with-parakeet      # + Parakeet alt ASR (~2.5 GB)
#   ./scripts/download_models.sh --all                # everything
#
# Verified sizes (via HfApi.repo_info, 2026-04-17):
#   mlx-community/whisper-large-v3-turbo                1.6 GB
#   mlx-community/Llama-3.3-70B-Instruct-4bit          39.7 GB
#   mlx-community/Qwen3-30B-A3B-4bit                   17.2 GB  (opt)
#   mlx-community/parakeet-tdt-0.6b-v3                  2.5 GB  (opt)
#
# Note: the earlier plan called for Qwen3-235B-A22B-MLX-4bit (~125 GB full
# repo size). That's too tight on 128 GB unified memory — model + KV cache
# + OS would exceed physical RAM. Llama 3.3 70B 4-bit is the right primary
# for this host class.
set -euo pipefail

CORE=(
  "mlx-community/whisper-large-v3-turbo:Whisper large-v3 turbo (ASR):~1.6 GB"
  "mlx-community/Llama-3.3-70B-Instruct-4bit:Llama 3.3 70B MLX 4-bit (primary LLM):~39.7 GB"
)

OPTIONAL_QWEN3_SMALL=(
  "mlx-community/Qwen3-30B-A3B-4bit:Qwen3-30B-A3B MLX 4-bit (fast MoE fallback, 3B active):~17.2 GB"
)

OPTIONAL_PARAKEET=(
  "mlx-community/parakeet-tdt-0.6b-v3:Parakeet TDT 0.6B v3 (alternative ASR):~2.5 GB"
)

# --- Parse flags --------------------------------------------------------
WANT_QWEN_SMALL=0
WANT_PARAKEET=0
for arg in "$@"; do
  case "$arg" in
    --all)                WANT_QWEN_SMALL=1; WANT_PARAKEET=1 ;;
    --with-qwen3-small)   WANT_QWEN_SMALL=1 ;;
    --with-parakeet)      WANT_PARAKEET=1 ;;
    -h|--help)
      grep -E "^# " "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "error: unknown flag '$arg' (try --help)" >&2
      exit 2
      ;;
  esac
done

# --- Ensure hf CLI is available -----------------------------------------
if ! command -v hf >/dev/null 2>&1; then
  echo "error: 'hf' CLI not found on PATH" >&2
  echo "  install with: pip install 'huggingface_hub[cli]'" >&2
  echo "  (or run scripts/install.sh which handles this)" >&2
  exit 1
fi

# --- Auth check ---------------------------------------------------------
if hf whoami >/dev/null 2>&1; then
  user="$(hf whoami 2>/dev/null | head -1)"
  echo "✓ authenticated as $user"
else
  echo "▶ Hugging Face login (paste your token from https://huggingface.co/settings/tokens)"
  hf auth login
fi

# --- Build the list -----------------------------------------------------
PLAN=("${CORE[@]}")
[[ $WANT_QWEN_SMALL -eq 1 ]] && PLAN+=("${OPTIONAL_QWEN3_SMALL[@]}")
[[ $WANT_PARAKEET   -eq 1 ]] && PLAN+=("${OPTIONAL_PARAKEET[@]}")

echo
echo "▶ planned downloads:"
for entry in "${PLAN[@]}"; do
  IFS=":" read -r repo label size <<< "$entry"
  echo "   · $label ($size)"
  echo "       $repo"
done
echo

# --- Download ------------------------------------------------------------
for entry in "${PLAN[@]}"; do
  IFS=":" read -r repo label size <<< "$entry"
  echo "▶ $label"
  hf download "$repo"
  echo
done

# --- Report -------------------------------------------------------------
CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
echo "✓ all downloads complete"
echo "  cache: $CACHE"
if command -v du >/dev/null 2>&1 && [[ -d "$CACHE" ]]; then
  total="$(du -sh "$CACHE" 2>/dev/null | awk '{print $1}')"
  echo "  total cache size: $total"
fi

echo
echo "next steps:"
echo "  1. start the LLM server in a terminal:"
echo
echo "       mlx_lm.server --model mlx-community/Llama-3.3-70B-Instruct-4bit --port 8080"
echo
echo "  2. verify everything:"
echo
echo "       python scripts/verify_env.py"
echo
