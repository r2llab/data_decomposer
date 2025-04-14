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

## Component Overview

The system is built around multiple implementations that share a common interface but offer different approaches to data handling and query processing:

1. **Core Infrastructure**:
   - Base implementation interface that defines the contract for all implementations
   - Configuration management for flexible system setup
   - Factory pattern for creating implementation instances

2. **Symphony Implementation**:
   - Focuses on data decomposition and structured execution
   - Separates discovery, decomposition, and execution phases
   - Uses vector embeddings for similarity search

3. **ReSP Implementation**:
   - Retrieval-enhanced Structured Processing 
   - Combines retrieval with structured reasoning
   - Specialized handling for different data modalities

4. **XMODE Implementation**:
   - Cross-modal data handling capabilities
   - Unified representation for text and tabular data

5. **Evaluation Framework**:
   - Comprehensive metrics for answer quality
   - Source relevance scoring
   - Cost tracking and efficiency analysis

## Key Features

1. **Multi-Modal Data Support**:
   - Process both textual and tabular data
   - Cross-modal querying capabilities
   - Unified representation for heterogeneous data sources

2. **Relevance Scoring**:
   - Source relevance tracking against ground truth answers
   - Precision, recall, and F1 metrics for source selection
   - Text similarity scoring for answer evaluation

3. **Cost Tracking**:
   - Detailed tracking of API usage and costs
   - Model-specific and endpoint-specific breakdowns
   - Query-level cost summaries

4. **Flexible Evaluation**:
   - Multiple metrics including ROUGE, string similarity, and LLM-based scoring
   - Source overlap analysis
   - Performance benchmarking across implementations

5. **Modular Design**:
   - Common interface for all implementations
   - Pluggable components for embedding, retrieval, and reasoning
   - Extensible architecture for adding new implementations

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

## Implementation Details

The system incorporates several key implementation features:

1. **Vector Embedding and Indexing**:
   - Both implementations use vector embeddings for semantic search
   - Supports pre-computing embeddings for efficient retrieval
   - Uses specialized indices for handling different data modalities

2. **Source Relevance Scoring**:
   - Computes similarity between retrieved content and ground truth answers
   - Supports different content types including text and dataframes
   - Implements both average and maximum relevance metrics

3. **Pipeline Architecture**:
   - Symphony uses a three-stage pipeline: discovery, decomposition, execution
   - ReSP implements a retrieval-reasoning-generation workflow
   - Both track document sources and provide detailed metadata

4. **Evaluation Methodology**:
   - Uses multiple metrics to evaluate answer quality
   - Implements both automated scoring and LLM-based evaluation
   - Provides detailed reports for both individual queries and aggregate results

5. **Extensibility**:
   - Common base class for all implementations
   - Standardized input/output formats
   - Flexible configuration for tuning system parameters

