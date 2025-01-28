# Project Setup Guide

## Environment Setup

1. Create a new conda environment called 'symphony':
`conda create --name symphony python=3.10`

2. Activate the environment: `conda activate symphony`

3. Install requirements: `pip install -r requirements.txt`


# GPT Embedding Model

Command to extract embeddings:
```
python scripts/auto_extract_embeddings.py \
    --data_dir data/ \
    --output_dir embeddings/ \
    --api_key openai-api-key \
    --batch_size 32
```

Command to build the index:
```
python scripts/build_index.py \
    --data-dir data/ \
    --embeddings embeddings/embeddings.npy \
    --output-dir index/ \
    --api-key openai-api-key
```

Command to run a query against the indexed dataset:
```
python scripts/run_query.py \
    --index-dir index/ \
    --api-key openai-api-key
```

# Cross-Modal Representation Learning

Command to train the model:
```
python scripts/train.py \
    --data-dir data/ \
    --output-dir checkpoints/ \
    --batch-size 32 \
    --epochs 10 \
    --lr 1e-4
```

Command to extract embeddings using the trained model:
```
python scripts/extract_embeddings.py \
    --checkpoint checkpoints/best_model.pt \
    --data-dir data/ \
    --output-dir embeddings/
```

Command to build the search index:
```
python scripts/build_index.py \
    --embeddings embeddings/embeddings.pt \
    --data-dir data/ \
    --output-dir index/
```