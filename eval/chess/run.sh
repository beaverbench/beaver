#!/bin/bash
set -e

# ============================================================================
# CHESS (IR -> SS -> CG) Evaluation Pipeline for Beaver
# Usage: ./run.sh --model gpt-5-mini --dataset dw --setting 0
# Resume an interrupted run: add --timestamp <YYYYMMDD-HHMMSS>
# ============================================================================

# --- Argument Parsing ---
MODEL=""
DATASET="dw"
SETTING=0
TIMESTAMP=""
NUM_WORKERS=8
EMBEDDING_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)           MODEL="$2";           shift 2 ;;
        --dataset)         DATASET="$2";         shift 2 ;;
        --setting)         SETTING="$2";         shift 2 ;;
        --workers)         NUM_WORKERS="$2";     shift 2 ;;
        --timestamp)       TIMESTAMP="$2";       shift 2 ;;
        --embedding_model) EMBEDDING_MODEL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./run.sh --model <model> --dataset <dataset> --setting <0|1|2> [--workers <n>] [--timestamp <ts>] [--embedding_model <model>]"
            echo ""
            echo "Arguments:"
            echo "  --model            LLM for every CHESS step, e.g. gpt-5-mini (required)"
            echo "  --dataset          Dataset split name, e.g. dw, dw_real, nova, neutron (default: dw)"
            echo "  --setting          Hint setting: 0=ETE, 1=schema hints, 2=all hints (default: 0)"
            echo "  --workers          Questions processed in parallel (default: 8); each runs up to 16 LLM calls at once"
            echo "  --timestamp        Resume an existing run folder output/...-log-<timestamp>"
            echo "  --embedding_model  Embedding model for value retrieval (default: EMBED_MODEL in .env, else qwen/qwen3-embedding-8b on OpenRouter)"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${MODEL}" ]; then
    echo "--model is required"; exit 1
fi
if [[ ! "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--workers must be a positive integer"; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="$(cd "${SCRIPT_DIR}/../../data" && pwd)"

# --- Load .env ---
if [ -f "${EVAL_DIR}/../.env" ]; then
    set -a
    source "${EVAL_DIR}/../.env"
    set +a
fi

# --embedding_model, else EMBED_MODEL from .env, else Qwen3-Embedding-8B on OpenRouter
EMBEDDING_MODEL="${EMBEDDING_MODEL:-${EMBED_MODEL:-qwen/qwen3-embedding-8b}}"

# Setup hint arguments for execute.py
HINTS=""
if [ "$SETTING" -eq 1 ]; then
    HINTS="--gold_tables --mapping --join_keys"
elif [ "$SETTING" -eq 2 ]; then
    HINTS="--gold_tables --mapping --join_keys --knowledge --decomp"
fi

TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d-%H%M%S")}"
MODEL_TAG="${MODEL//\//_}"
OUTPUT_DIR="${SCRIPT_DIR}/output/${MODEL_TAG}-beaver-${DATASET}-setting${SETTING}-log-${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"
# Save everything this script prints to the run folder (appended when a run is resumed); Python output
# unbuffered so it lands in order
export PYTHONUNBUFFERED=1
exec > >(tee -a "${OUTPUT_DIR}/run.log") 2>&1

echo "================================"
echo "CHESS on Beaver ${DATASET} Dataset"
echo "Setting ${SETTING}"
echo "================================"
echo "Model: $MODEL"
echo "Embedding model: $EMBEDDING_MODEL"
echo "Output Path: $OUTPUT_DIR"
echo "================================"

cd "$SCRIPT_DIR"

echo ""
echo "Step 1: Building Value Index (skipped if it exists)"
echo "=============================="
python preprocess.py --dataset "$DATASET" --data_dir "$DATA_DIR"

echo ""
echo "Step 2: Generating Predictions"
echo "=============================="
python execute.py \
    --model "$MODEL" \
    --embedding_model "$EMBEDDING_MODEL" \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --q_fn "dev_sampled" \
    --num_workers "$NUM_WORKERS" \
    $HINTS

echo ""
echo ""
echo "Step 3: Unify Predictions"
echo "=============================="
GOLD_FILE="${DATA_DIR}/${DATASET}/dev_sampled.json"

python ../utils/unify.py \
    --baseline chess \
    --input_dir "$OUTPUT_DIR" \
    --gold_file "$GOLD_FILE" \
    --dataset "$DATASET"

echo "========================================"
echo "CHESS Generation Complete!"
echo "Results saved to: $OUTPUT_DIR"
