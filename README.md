# 🦫 BEAVER: An Enterprise Benchmark for Text-to-SQL

[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue?logo=huggingface)](https://huggingface.co/collections/beaverbench/beaver-dataset)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv)](https://arxiv.org/abs/2409.02038)

This repository contains the evaluation code for the **BEAVER** text-to-SQL benchmark. It includes (i) four text-to-SQL methods: **ReFoRCE**, **DAIL-SQL**, **DIN-SQL**, and **Few-shot** (ii) two evaluation modes: **coarse-grained evaluation** (execution accuracy) and **fine-grained evaluation** across five subtasks critical to text-to-SQL.

If you find our data, code, or the paper helpful, please cite the paper:
```
@article{chen2024beaver,
  title={BEAVER: an enterprise benchmark for text-to-sql},
  author={Chen, Peter Baile and Yang, Devin and Li, Weiyue and Wenz, Fabian and Zhang, Yi and Tatbul, Nesime and Cafarella, Michael and Demiralp, {\c{C}}a{\u{g}}atay and Stonebraker, Michael},
  journal={arXiv preprint arXiv:2409.02038},
  year={2024}
}
```

## Repository Structure

```
├── .env                                 # Credential file (API keys + MySQL database password)
├── data/                                # Dataset files (e.g. metadata, questions, tables)
│   ├── dw/                              # `dw` dataset
│   │   ├── example.json                 # Few-shot examples
│   │   └── ...                 
│   └── ...                              # Other datasets
├── eval/                                # Evaluation methods and scripts
│   ├── reforce/                         # ReFoRCE evaluation pipeline
│   ├── fewshot/                         # Few-shot evaluation pipeline
│   ├── dailsql/                         # DAIL-SQL evaluation pipeline
│   ├── dinsql/                          # DIN-SQL evaluation pipeline
│   ├── evaluate_ex_acc.py               # Script for computing execution accuracy
│   └── evaluate_subtasks.py             # Script for subtask evaluation
└── retrieve/                            # Table retrieval
```

## Getting Started

### Credentials (`.env` file)

All API keys and MySQL credentials are managed through a single `.env` file at the root directory.

The `.env` file should contain:
```
# LLM API Keys
OPENAI_API_KEY=xxx
OPENROUTER_API_KEY=xxx

# MySQL Credentials (shared across all databases)
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=xxx

# Optional (ReFoRCE only): if LLM_BASE_URL is set, ReFoRCE sends ALL requests to this
# OpenAI-compatible endpoint (e.g., vLLM) and ignores OPENAI_API_KEY / OPENROUTER_API_KEY.
# Always set LLM_API_KEY (any placeholder if the endpoint needs none).
LLM_BASE_URL=xxx
LLM_API_KEY=xxx
LLM_RPM=30   # cap on requests per minute, for rate-limited endpoints
```

### Data Pre-processing
Our [gated dataset](https://huggingface.co/collections/beaverbench/beaver-dataset) is hoted on Hugging Face. You must be **authenticated** to access it, which means logging in through your CLI.

```bash
python data/download_hf.py --sample [sample_size]
```

For each of the four datasets (`dw`, `nova`, `neutron`, `dw_real`), this script automatically downloads the datasets from Hugging Face and generates:
* `dev.json`: The full set of questions.
* `dev_tables.json`: The full set of tables.
* `dev_sampled.json`: A sampled subset of questions of size `sample_size` (default `100`) since running on the full dataset can be computationally expensive

You should also follow the instruction [here](https://huggingface.co/datasets/beaverbench/beaver-table#getting-started) to setup the MySQL databases.

### Table retrieval
The following command applies a retrieve-then-rerank pipeline to retrieve tables in `dev_tables.json` that are semantically relevant to questions in `dev_sampled.json`, guiding downstream text-to-SQL generation. The pipeline uses a dense embedding model `embed_model` for retrieval and an optional reranker model `rerank_model` for improved ordering. If `rerank_model` is omitted, the pipeline performs retrieval only, without the reranking step.

```
python retrieve/retrieve.py --dataset [dataset] --embed_provider local --rerank_model Qwen/Qwen3-Reranker-8B
```

The command takes the following arguments:
* `dataset`: one of `dw`, `dw_real`, `neutron`, `nova`
* `embed_model` (default `Qwen/Qwen3-Embedding-8B`): the dense embedding model for the retrieval step
* `embed_k` (default `50`): the number of tables returned by the retrieval step
* `embed_provider`: `local` (GPU required) or `openrouter` (requires a valid `OPENROUTER_API_KEY`)
* `rerank_model` (default `None`): the optional reranker model for the reranking step; the paper used `Qwen/Qwen3-Reranker-8B`
* `rerank_k` (default `15`): the number of tables returned by the reranking step

Note: the retrieval step outputs `retrieved_tables.json`, and the reranking step (if enabled) outputs `reranked_tables.json`. While generating SQL, the text-to-SQL method uses `reranked_tables.json` if it exists and otherwise `retrieved_tables.json`.

## Text-to-SQL methods

We consider four text-to-SQL methods:
1. ReFoRCE (adapted from [this official ReFoRCE implementation](https://github.com/Snowflake-Labs/ReFoRCE/tree/o3/methods/ReFoRCE)): an agentic method with candidate generation, majority voting, and column exploration.
2. DIN-SQL (adapted from [this Spider2 implementation](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite/baselines/dinsql)): a method with query decomposition and self-correction
3. DAIL-SQL (adapted from [this Spider2 implementation](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite/baselines/dailsql)): a method with example selection
4. Few-shot: a method with static in-context examples

### Environment

You can setup the package environment using either `conda` or `venv`.

```bash
cd eval

# Using `conda`
conda create -n beaver-eval python=3.10 -y
conda activate beaver-eval

# Using `venv`
python3 -m venv beaver-eval
source beaver-eval/bin/activate

pip install -r requirements.txt

# To run the DAIL‑SQL method, execute the following additional commands:
python dailsql/nltk_downloader.py
python -m spacy download en_core_web_sm
```

### SQL generation

All methods can be executed using the `run.sh` script in their respective folders.
```bash
cd eval/[method]
./run.sh --model [model] --dataset [dataset] --setting {0,1,2}
```
- `model`: the LLM for SQL generation (e.g., `gpt-5-mini`)
- `dataset`: one of `dw`, `dw_real`, `neutron`, `nova`
- `setting=0` *(default setting)*: Standard end-to-end setting with no hints. Base information with only the top-k tables retrieved from [table retrieval](#table-retrieval).
- `setting=1`: With hints for three schema-linking subtasks. Includes gold tables, column mapping, and join keys.
- `setting=2`: With hints for all five subtasks. Includes three subtasks in `setting=1`, domain knowledge, and subquery decomposition.

ReFoRCE also accepts two optional arguments:
```bash
cd eval/reforce
./run.sh --model [model] --dataset [dataset] --setting {0,1,2} --workers [workers] --timestamp [timestamp]
```
- `workers`: number of questions processed in parallel, each running 8 votes (default `16`). Lower it if you hit rate limits.
- `timestamp`: resume an interrupted run by reusing its output folder (the `YYYYMMDD-HHMMSS` suffix of `output/...-log-[timestamp]`).

## Evaluation

We evaluate text-to-SQL methods in two complementary modes:
1. **Coarse-grained evaluation** using execution accuracy. This is the standard metric, but an all-or-nothing signal makes it hard to diagnose where a method failed, especially when producing a correct SQL involves solving multiple *compounded* challenges, such as domain knowledge and complex query construction.
2. **Fine-grained evaluation** across five subtasks critical to text-to-SQL performance, enabling more targeted error analysis.

### Execution accuracy
Execution accuracy (`ex_acc`) is 1 when the generated SQL returns the same execution result as the gold SQL, and 0 otherwise.
```bash
cd eval

# Using `conda`
conda activate beaver-eval

# Using `venv`
source beaver-eval/bin/activate

python evaluate_ex_acc.py --dataset [dataset] --input_dir unified-output/[method]/[run_name]
```
<!-- *(This produces `summary_ex_acc.json` in the `input_dir`)* -->

### Subtask evaluation
We assess five subtasks: multi-table retrieval, join key detection, column mapping, domain knowledge extraction, and query decomposition.

For each subtask (except query decomposition), we compare information extracted from the generated SQL (using a specified LLM `model`) with annotated information from the gold SQL. Query decomposition is evaluated via LLM-as-a-judge: the `model` scores how closely the structure of the generated SQL matches the gold query’s decomposition based on human‑designed rubrics.

```bash
python evaluate_subtasks.py --dataset [dataset] --model [model] --input_dir unified-output/[method]/[run_name]
```
<!-- *(This produces `summary_subtasks.json` and detailed outputs in folder `subtask_eval/` in the `input_dir`)* -->
