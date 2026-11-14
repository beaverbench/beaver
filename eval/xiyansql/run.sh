#!/bin/bash
set -e

# ============================================================================
# XiYan-SQL Evaluation Pipeline for Beaver
# Usage: ./run.sh --model xiyansql-32b-2504 --dataset dw --setting 0
# Requires a vLLM server for XGenerationLab/XiYanSQL-QwenCoder-32B-2504 at LLM_BASE_URL
# Resume an interrupted run: add --timestamp <YYYYMMDD-HHMMSS>
# ============================================================================

# --- Argument Parsing ---
MODEL="xiyansql-32b-2504"
DATASET="dw"
SETTING=0
TIMESTAMP=""
NUM_WORKERS=16

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2";     shift 2 ;;
        --dataset)   DATASET="$2";   shift 2 ;;
        --setting)   SETTING="$2";   shift 2 ;;
        --workers)   NUM_WORKERS="$2"; shift 2 ;;
        --timestamp) TIMESTAMP="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./run.sh --model <model> --dataset <dataset> --setting <0|1|2> [--workers <n>] [--timestamp <ts>]"
            echo ""
            echo "Arguments:"
            echo "  --model      Model name served by vLLM, i.e. its --served-model-name (default: xiyansql-32b-2504)"
            echo "  --dataset    Dataset split name, e.g. dw, sp, nova, neutron (default: dw)"
            echo "  --setting    Hint setting: 0=ETE, 1=schema hints, 2=all hints (default: 0)"
            echo "  --workers    Concurrent requests to the vLLM server (default: 16)"
            echo "  --timestamp  Resume an existing run folder output/...-log-<timestamp>"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

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

if [ -z "${LLM_BASE_URL}" ]; then
    echo "LLM_BASE_URL is not set; point it at the vLLM server, e.g. http://localhost:8000/v1"; exit 1
fi

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

echo "================================"
echo "XiYan-SQL on Beaver ${DATASET} Dataset"
echo "Setting ${SETTING}"
echo "================================"
echo "Model: $MODEL"
echo "Endpoint: $LLM_BASE_URL"
echo "Output Path: $OUTPUT_DIR"
echo "================================"

echo ""
echo "Step 1: Generating Predictions"
echo "=============================="
cd "$SCRIPT_DIR"
python execute.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --q_fn "dev_sampled" \
    --num_workers "$NUM_WORKERS" \
    $HINTS

echo ""
echo ""
echo "Step 2: Unify Predictions"
echo "=============================="
GOLD_FILE="${DATA_DIR}/${DATASET}/dev_sampled.json"

python ../utils/unify.py \
    --baseline xiyansql \
    --input_dir "$OUTPUT_DIR" \
    --gold_file "$GOLD_FILE" \
    --dataset "$DATASET"

echo "========================================"
echo "XiYan-SQL Generation Complete!"
echo "Results saved to: $OUTPUT_DIR"
