#!/bin/bash

# ============================================================================
# DAIL-SQL Evaluation Pipeline for Beaver
# Usage: ./run.sh --model gpt-5.2 --dataset dw --setting 0
# ============================================================================

# --- Argument Parsing ---
MODEL="gpt-5.2"
DATASET="dw"
SETTING=0
TIMESTAMP=""
NUM_WORKERS=16

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
            echo "  --workers    Questions processed in parallel (processes) (default: 16)"
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

# Map setting to internal option number (Python scripts use 1/2/3): option 1 = setting 0 (retrieved tables,
# no hints), option 2 = setting 1 (gold tables, mapping, join keys), option 3 = setting 2 (all hints)
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
COMMENT="beaver_${DATASET}_opt${OPTION}"

# Data paths
BEAVER_QUESTIONS="../../data/${DATASET}/dev_sampled.json"
BEAVER_TABLES="../../data/${DATASET}/dev_tables.json"

TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d-%H%M%S")}"
RUN_NAME="${MODEL_TAG}-beaver-${DATASET}-setting${SETTING}-log-${TIMESTAMP}"
# One results folder per run (earlier runs of the same model, dataset and setting used to share one)
RESULTS_DIR="postprocessed_data/${COMMENT}_${DEV}_CTX-200/RESULTS_MODEL-${RUN_NAME}-SQL"
mkdir -p "${script_dir}/${RESULTS_DIR}"
# Save everything this script prints to the run folder (appended when a run is resumed); Python output
# unbuffered so it lands in order
export PYTHONUNBUFFERED=1
exec > >(tee -a "${script_dir}/${RESULTS_DIR}/run.log") 2>&1

echo "========================================"
echo "Running DAILSQL on ${DATASET} - Setting ${SETTING} (option ${OPTION})"
echo "Model: ${MODEL}"
echo "========================================"

# Step 1: Preprocess beaver data
echo ""
echo "Step 1: Preprocessing Beaver data"
cd ${script_dir}
python preprocessed_data/beaver_preprocess.py \
    --option ${OPTION} \
    --questions_file ${BEAVER_QUESTIONS} \
    --tables_file ${BEAVER_TABLES} \
    --dataset ${DATASET}

if [ $? -ne 0 ]; then
    echo "Error during preprocessing. Exiting."
    exit 1
fi

# Step 2: Generate questions with prompts
echo ""
echo "Step 2: Generating questions with DAIL-SQL prompts..."
python generate_question.py --dev ${DEV} --model ${MODEL} --tokenizer ${MODEL} --prompt_repr SQL --comment ${COMMENT} --processes 1

# Step 3: Query LLM
echo ""
echo "Step 3: Querying LLM to generate SQL..."
python ask_llm.py --model ${MODEL} --n 1 --question postprocessed_data/${COMMENT}_${DEV}_CTX-200 \
    --results_dir ${RESULTS_DIR} --processes ${NUM_WORKERS}

# Step 4: Postprocess results and execute on MySQL
echo ""
echo "Step 4: Postprocessing and executing SQL on MySQL..."
DEV_FILE="preprocessed_data/${DEV}/${DEV}_preprocessed.json"
TABLES_FILE=${BEAVER_TABLES}

python postprocessed_data/beaver_postprocess.py \
    --dataset ${DATASET} \
    --option ${OPTION} \
    --model ${MODEL} \
    --comment ${COMMENT} \
    --dev_file ${DEV_FILE} \
    --dev ${DEV} \
    --tables_file ${TABLES_FILE} \
    --results_dir ${RESULTS_DIR}

# Step 5: Unify results
echo ""
echo "Step 5: Unifying generated SQLs..."
python ../utils/unify.py \
    --baseline dailsql \
    --input_dir ${RESULTS_DIR} \
    --run_name $RUN_NAME \
    --gold_file ../../data/${DATASET}/dev_sampled.json \
    --dataset ${DATASET}

echo ""
echo "========================================"
echo "DAILSQL ${DATASET} Setting ${SETTING} Complete!"
echo "Model: ${MODEL}"
echo "========================================"
