#!/usr/bin/env bash
# Download Hugging Face model weights for long-form-editor.
#
# Usage:
#   ./scripts/download_models.sh                # core only (Whisper + Qwen3)
#   ./scripts/download_models.sh --all          # core + fallback + parakeet
#   ./scripts/download_models.sh --with-llama   # core + Llama-3.3-70B fallback
#   ./scripts/download_models.sh --with-parakeet
#
# First run does `hf auth login` interactively if not already authed.
# Subsequent runs are idempotent (hf download is a cache-aware fetcher).
set -euo pipefail

CORE=(
  "mlx-community/whisper-large-v3-turbo:Whisper large-v3 turbo (MLX):~3 GB"
  "Qwen/Qwen3-235B-A22B-MLX-4bit:Qwen3-235B-A22B MLX 4-bit (primary LLM):~33 GB"
)

OPTIONAL_LLAMA=(
  "mlx-community/Llama-3.3-70B-Instruct-4bit:Llama 3.3 70B MLX 4-bit (LLM fallback):~40 GB"
)

OPTIONAL_PARAKEET=(
  "mlx-community/parakeet-tdt-0.6b-v3:Parakeet TDT 0.6B v3 (alternative ASR):~2.5 GB"
)

# --- Parse flags --------------------------------------------------------
WANT_LLAMA=0
WANT_PARAKEET=0
for arg in "$@"; do
  case "$arg" in
    --all)           WANT_LLAMA=1; WANT_PARAKEET=1 ;;
    --with-llama)    WANT_LLAMA=1 ;;
    --with-parakeet) WANT_PARAKEET=1 ;;
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
[[ $WANT_LLAMA    -eq 1 ]] && PLAN+=("${OPTIONAL_LLAMA[@]}")
[[ $WANT_PARAKEET -eq 1 ]] && PLAN+=("${OPTIONAL_PARAKEET[@]}")

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
echo "       mlx_lm.server --model Qwen/Qwen3-235B-A22B-MLX-4bit --port 8080"
echo
echo "  2. verify everything:"
echo
echo "       python scripts/verify_env.py"
echo
