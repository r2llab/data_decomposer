# Project Setup Guide

## Environment Setup

1. Create a new conda environment called 'data_decomposer':
`conda create --name data_decomposer python=3.10`

2. Activate the environment: `conda activate data_decomposer`

3. Install requirements: `pip install -r requirements.txt`

# Symphony Architecture Overview

## Repository Structure

The repository is organized into the following main directories:

- `symphony/`: Main package directory containing all core components
  - `core/`: Core pipeline and interfaces
  - `embeddings/`: Embedding models and utilities
  - `discovery/`: Search and retrieval components
  - `decomposition/`: Query decomposition logic
  - `execution/`: Query execution engine
  - `utils/`: Shared utility functions

- `scripts/`: Command-line tools and utilities
  - `auto_extract_embeddings.py`: Extract embeddings from data using a GPT embedding model
  - `build_index.py`: Build search indices
  - `run_query.py`: Run queries against the system
  - `train.py`: Training a T5-based autoencoder model
  - `extract_embeddings.py`: Extracting embeddings from the trained T5-based autoencoder model

- `data/`: Data storage
  - `traindev_tables_tok/`: Tokenized table data from Open Table-and-Text Question Answering (OTT-QA)
  - `traindev_request_tok/`: Tokenized text data from Open Table-and-Text Question Answering (OTT-QA)

- `checkpoints/`: Model checkpoints for AutoEncoder embedding model
- `embeddings/`: Generated embeddings
- `index/`: Built search indices
- `tests/`: Test suite

The main interface for interacting with Symphony's components is through `core/pipeline.py`, which orchestrates the flow between different components.

## Component Overview

### Dataset (`embeddings/dataset.py`)
The `CrossModalDataset` class handles both tabular and text data, providing a unified interface for loading and preprocessing data. It supports loading from directories and serializing items into a format suitable for embedding.

### Embedding (`embeddings/`)
Symphony implements two embedding approaches:
- `AutoEmbedder`: Uses OpenAI's embedding models (text-embedding-3-small by default) for production use
- `SymphonyAutoEncoder`: A custom T5-based autoencoder model for research and offline use

### Discovery (`discovery/`)
The discovery component implements semantic search using vector indices. It combines both semantic similarity and keyword matching for improved retrieval, with configurable boosting of keyword matches.

### Decomposition (`decomposition/`)
The decomposer breaks down complex queries into simpler sub-queries that can be executed independently. It uses OpenAI's API for natural language understanding and query planning.

### Execution (`execution/`)
The executor processes decomposed queries against the retrieved context. It handles both direct lookups and more complex reasoning tasks using the OpenAI API.

### Aggregation (`execution/aggregator.py`)
The aggregator combines results from multiple sub-queries into a coherent final response, ensuring consistency and handling any conflicts in the intermediate results.

# Running Main

```
python main.py --config config.yaml "your query here"
```

# Data Decomposer with Source Relevance Scoring

This project implements a query processing system that can evaluate relevance of retrieved sources against a ground truth answer.

## Key Features

- Process natural language queries and extract information from various data sources
- Calculate relevance scores between retrieved sources and a ground truth answer
- Support for multiple implementation frameworks (Symphony, ReSP)
- Evaluation tools for performance metrics

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

## Source Relevance Scoring

The system now calculates how relevant each retrieved document is to the ground truth answer. This addresses the limitation of simple source comparison, as the system might use equivalent sources with different names or draw information from sources that contain the same data.

The relevance score is calculated using string similarity between each retrieved document's content and the ground truth answer. The system tracks:

- Average relevance score across all retrieved documents
- Maximum relevance score (best match)
- Individual scores for each document

These metrics help evaluate if the system is retrieving relevant information regardless of source names.

## Implementation Details

The source relevance scoring is implemented by:

1. Passing the ground truth answer to the implementation's `process_query` method
2. Intercepting document retrieval to calculate similarity between document content and ground truth
3. Accumulating scores for all retrieved documents
4. Returning the aggregate metrics in the result

This approach works with both Symphony and ReSP implementations.


Sample query over the db:
```
sqlite3 data/drugbank.db "SELECT COUNT(*) FROM drugbank_drug;"
```
