#!/bin/bash
set -e

# ============================================================================
# Few-shot Evaluation Pipeline for Beaver
# Usage: ./run.sh --model gpt-5-mini --dataset dw --setting 0
# ============================================================================

# --- Argument Parsing ---
MODEL="gpt-5-mini"
DATASET="dw"
SETTING=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)   MODEL="$2";   shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --setting) SETTING="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./run.sh --model <model> --dataset <dataset> --setting <0|1|2>"
            echo ""
            echo "Arguments:"
            echo "  --model    LLM model name (default: gpt-5-mini)"
            echo "  --dataset  Dataset split name, e.g. dw, sp, nova, neutron (default: dw)"
            echo "  --setting  Hint setting: 0=ETE, 1=schema hints, 2=all hints (default: 0)"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="$(cd "${SCRIPT_DIR}/../../data" && pwd)"

# --- Load .env ---
if [ -f "${EVAL_DIR}/../.env" ]; then
    set -a
    source "${EVAL_DIR}/../.env"
    set +a
fi

# Setup arguments for parallel_execute.py
HINTS=""
if [ "$SETTING" -eq 1 ]; then
    HINTS="--gold_tables --mapping --join_keys"
elif [ "$SETTING" -eq 2 ]; then
    HINTS="--gold_tables --mapping --join_keys --knowledge --decomp"
fi

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
OUTPUT_DIR="${SCRIPT_DIR}/output/${MODEL}-beaver-${DATASET}-setting${SETTING}-log-${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "================================"
echo "Few-Shot on Beaver ${DATASET} Dataset"
echo "Setting ${SETTING}"
echo "================================"
echo "Model: $MODEL"
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
    $HINTS

echo ""
echo ""
echo "Step 2: Unify Predictions"
echo "=============================="
GOLD_FILE="${DATA_DIR}/${DATASET}/dev_sampled.json"

python ../utils/unify.py \
    --baseline fewshot \
    --input_dir "$OUTPUT_DIR" \
    --gold_file "$GOLD_FILE" \
    --dataset "$DATASET"

echo "========================================"
echo "Few-Shot Generation Complete!"
echo "Results saved to: $OUTPUT_DIR"
