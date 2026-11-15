#!/bin/bash

# ============================================================================
# DIN-SQL Evaluation Pipeline for Beaver
# Usage: ./run.sh --model gpt-5.2 --dataset dw --setting 0
# ============================================================================

# --- Argument Parsing ---
MODEL="gpt-5.2"
DATASET="dw"
SETTING=0
TIMESTAMP=""
NUM_WORKERS=20

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)   MODEL="$2";   shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --setting) SETTING="$2"; shift 2 ;;
        --workers)   NUM_WORKERS="$2"; shift 2 ;;
        --timestamp) TIMESTAMP="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./run.sh --model <model> --dataset <dataset> --setting <0|1|2> [--workers <n>] [--timestamp <ts>]"
            echo ""
            echo "Arguments:"
            echo "  --model    LLM model name (default: gpt-5.2)"
            echo "  --dataset  Dataset split name, e.g. dw, sp, nova, neutron (default: dw)"
            echo "  --setting  Hint setting: 0=ETE, 1=schema hints, 2=all hints (default: 0)"
            echo "  --workers    Questions processed in parallel (processes) (default: 20)"
            echo "  --timestamp  Resume an existing run folder ...-log-<timestamp>"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ! "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--workers must be a positive integer"; exit 1
fi
MODEL_TAG="${MODEL//\//_}"

# Map setting to internal option number (Python scripts use 1/2/3)
OPTION=$((SETTING + 1))

# --- Load .env ---
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${script_dir}/../../.env" ]; then
    set -a
    source "${script_dir}/../../.env"
    set +a
fi

# --- Check API keys ---
if [ -z "$OPENAI_API_KEY" ] && [ -z "$OPENROUTER_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ] && [ -z "$LLM_BASE_URL" ]; then
    echo "Error: No LLM API key found. Set OPENAI_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, or LLM_BASE_URL in .env"
    exit 1
fi

# --- Derived Variables ---
DEV="beaver_${DATASET}_opt${OPTION}"

# Data paths
BEAVER_QUESTIONS="../../data/${DATASET}/dev_sampled.json"
BEAVER_TABLES="../../data/${DATASET}/dev_tables.json"

TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_PATH="output/${MODEL_TAG}-beaver-${DATASET}-setting${SETTING}-log-${TIMESTAMP}"

echo "========================================================================"
echo "DINSQL on ${DATASET} - Setting ${SETTING} (option ${OPTION})"
echo "Model: ${MODEL}"
echo "Workers: ${NUM_WORKERS}"
echo "========================================================================"

# Step 1: Preprocess data
echo ""
echo "[Step 1/3] Preprocessing Beaver data for Setting ${SETTING}..."
echo "------------------------------------------------------------------------"

python3 preprocessed_data/beaver_preprocess_v2.py \
    --option ${OPTION} \
    --questions_file ${BEAVER_QUESTIONS} \
    --tables_file ${BEAVER_TABLES} \
    --dataset ${DATASET}

if [ $? -ne 0 ]; then
    echo "Error during preprocessing. Exiting."
    exit 1
fi

# Step 2: Run DINSQL
echo ""
echo "[Step 2/3] Running DINSQL SQL generation..."
echo "------------------------------------------------------------------------"
echo "Using preprocessed data from: preprocessed_data/${DEV}/"

python3 DIN-SQL-beaver-v2.py \
    --dev ${DEV} \
    --model ${MODEL} \
    --temperature 0 \
    --n 1 \
    --processes ${NUM_WORKERS} \
    --output_dir $OUTPUT_PATH

if [ $? -ne 0 ]; then
    echo "Error during SQL generation. Exiting."
    exit 1
fi

# Step 3: Unify results
echo ""
echo "[Step 3/3] Unifying generated SQLs..."
echo "------------------------------------------------------------------------"

python3 ../utils/unify.py \
    --baseline dinsql \
    --input_dir $OUTPUT_PATH \
    --gold_file "../../data/${DATASET}/dev_sampled.json" \
    --dataset ${DATASET}

echo ""
echo "========================================================================"
echo "DIN-SQL on ${DATASET} Setting ${SETTING} Complete!"
echo "Model: ${MODEL}"
echo "========================================================================"
