# Project Setup Guide

## Environment Setup

1. Create a new conda environment called 'data_decomposer':
`conda create --name data_decomposer python=3.10`

2. Activate the environment: `conda activate data_decomposer`

3. Install requirements: `pip install -r requirements.txt`

# Project Architecture Overview

## Repository Structure

The repository is organized into the following main directories:

- `core/`: Core pipeline interfaces and system setup
  - `base_implementation.py`: Abstract base class for all implementations
  - `config.py`: Configuration management 
  - `factory.py`: Factory pattern for creating implementation instances

- `data/`: Data storage for input datasets

- `data_processing/`: Data processing and question generation scripts
  - Jupyter notebooks for generating questions from different data types (passages, tables, etc.)

- `implementations/`: Contains different system implementations
  - `symphony/`: Symphony implementation with data decomposition and execution
  - `ReSP/`: Retrieval-enhanced Structured Processing implementation
  - `XMODE/`: Cross-modal data handling implementation
  - `baseline/`: Baseline implementation for comparison

- `results/`: Results storage for evaluation outputs

- `results_v2/`: Extended results storage with additional metrics

- `scripts/`: Command-line tools and utilities
  - `auto_extract_embeddings.py`: Extract embeddings from data using a GPT embedding model
  - `build_index.py`: Build search indices for data retrieval
  - `run_query.py`: Run queries against the system
  - `train.py`: Training a T5-based autoencoder model
  - `extract_embeddings.py`: Extracting embeddings from the trained T5-based autoencoder model
  - `passage_embedd_and_index.py`: Process and index passage data
  - `build_representation_index.py`: Build indices for cross-modal representations
  - `csv_to_sqlite.py`: Convert CSV data to SQLite database format

- `tests/`: Test suite for validating system functionality

## Usage

### Processing a Single Query

To process a query with ground truth answer for source relevance scoring:

```bash
python main.py --config config.yaml --ground-truth-answer "Ground truth answer text" "Your query here"
```

### Running Evaluation on Multiple Queries

To evaluate the system against a dataset of queries and ground truth answers:

```bash
python evaluate_qa.py --config config.yaml --gt-file path/to/groundtruth.csv --output results.json
```

The ground truth file should be a CSV with columns: `question`, `answer`, `text`, `table`. Where:
- `question`: The query to process
- `answer`: The ground truth answer
- `text`: Comma-separated list of expected text source files
- `table`: Comma-separated list of expected table source files

Example:
```csv
"question","answer","text","table"
"What is the mechanism of action for Cetuximab?","Cetuximab is an EGFR binding FAB, targeting the EGFR in humans.","None","drugbank-targets"
```

## Passage QA Pipeline (OpenRouter)

All passage QA generation and processing scripts use OpenRouter for LLM calls. Place your key in `.env` at the repo root as `OPENROUTER_API_KEY`.

- End-to-end runner (workers decide mode):
  - Sequential (workers=1):
    - `python data_processing/run_passage_pipeline.py --num-questions 100 --workers 1`
  - Parallel (workers>1):
    - `python data_processing/run_passage_pipeline.py --workers 4 --num-questions 100`
  - Optional: restrict sources considered to the first N passages for debugging: `--limit N`
  - Passage source for each generation attempt is randomly sampled from the available passage set.
  - Optional reproducibility: `--seed 42`
  - Default model is `openai/gpt-5.2` (override with `--model`)
  - Outputs are CSV files:
    - all generated QA rows: `--generated-file` (default `data_processing/passage_generated.csv`)
    - processed QA rows: `--processed-file` (default `data_processing/passage_processed.csv`)
  - Each output row includes:
    - `question_id`: unique ID to reference the question
    - `short_answer`: direct short answer
    - `answer_reasoning`: brief supporting reasoning
  - Files are flushed to disk every 20 written rows during execution.

- Individual stages remain available:
  - Generation only: `python data_processing/generate_passage_questions.py --model openai/gpt-5.2`
  - Processing only: `python data_processing/process_passage_questions.py --model openai/gpt-5.2`

# Acknowledgements

While building the benchmark and implementing the three methods, I used Github Co-pilot as an assistive tool. I primary used Co-pilot for assisting in writing boiler plate code for functions that I planned, designed and architected myself.
