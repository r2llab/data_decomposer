from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from vector_index import QwenLocalEmbedder

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SearchableField,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )
    _AZURE_SEARCH_SDK_AVAILABLE = True
except ModuleNotFoundError:
    AzureKeyCredential = Any  # type: ignore[assignment]
    ResourceNotFoundError = Exception  # type: ignore[assignment]
    SearchClient = Any  # type: ignore[assignment]
    SearchIndexClient = Any  # type: ignore[assignment]
    HnswAlgorithmConfiguration = Any  # type: ignore[assignment]
    SearchField = Any  # type: ignore[assignment]
    SearchFieldDataType = Any  # type: ignore[assignment]
    SearchIndex = Any  # type: ignore[assignment]
    SearchableField = Any  # type: ignore[assignment]
    SimpleField = Any  # type: ignore[assignment]
    VectorSearch = Any  # type: ignore[assignment]
    VectorSearchProfile = Any  # type: ignore[assignment]
    _AZURE_SEARCH_SDK_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_TABLES_DIR = REPO_ROOT / "data" / "Pharma" / "drugbank-tables"
DEFAULT_INDEX_NAME = "drug_bank_data_lake_tables"
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"
VECTOR_FIELD = "content_vector"
VECTOR_PROFILE_NAME = "vector-profile"
VECTOR_ALGORITHM_NAME = "hnsw-default"


def _require_azure_search_sdk() -> None:
    if _AZURE_SEARCH_SDK_AVAILABLE:
        return
    raise ModuleNotFoundError(
        "Missing dependency `azure-search-documents`. "
        "Install it with: python -m pip install azure-search-documents"
    )


@dataclass
class TableMetadataDoc:
    id: str
    doc_type: str
    title: str
    source: str
    table_name: str
    content: str
    metadata_json: str
    row_count: int
    column_count: int


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip().lower())
    return cleaned.strip("_") or "table"


def normalize_table_name(raw_name: str) -> str:
    return _slugify(raw_name)


def _safe_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if len(text) > 120:
        return text[:117] + "..."
    return text


def _count_csv_rows(path: Path) -> int:
    row_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        next(handle, None)  # header
        for _ in handle:
            row_count += 1
    return row_count


def _build_table_doc(path: Path, tables_dir: Path, sample_rows: int) -> TableMetadataDoc:
    table_name = normalize_table_name(path.stem)
    relative_source = str(path.relative_to(tables_dir))

    sample_df = pd.read_csv(path, nrows=sample_rows)
    row_count = _count_csv_rows(path)
    column_count = int(len(sample_df.columns))

    columns_metadata: list[dict[str, Any]] = []
    for col in sample_df.columns:
        examples = [_safe_value(v) for v in sample_df[col].head(3).tolist()]
        examples = [x for x in examples if x]
        columns_metadata.append(
            {
                "name": str(col),
                "dtype": str(sample_df[col].dtype),
                "examples": examples,
            }
        )

    content_lines = [
        f"Table name: {table_name}",
        f"Source file: {relative_source}",
        f"Row count: {row_count}",
        f"Column count: {column_count}",
        "Columns:",
    ]
    for col in columns_metadata:
        examples = ", ".join(col["examples"]) if col["examples"] else "n/a"
        content_lines.append(f"- {col['name']} ({col['dtype']}), examples: {examples}")
    content = "\n".join(content_lines)

    metadata = {
        "table_name": table_name,
        "source_file": relative_source,
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns_metadata,
    }

    return TableMetadataDoc(
        id=f"table-{table_name}",
        doc_type="table_metadata",
        title=f"Schema for {table_name}",
        source=relative_source,
        table_name=table_name,
        content=content,
        metadata_json=json.dumps(metadata, ensure_ascii=False),
        row_count=row_count,
        column_count=column_count,
    )


def iter_table_docs(tables_dir: Path, sample_rows: int) -> Iterator[TableMetadataDoc]:
    csv_files = sorted(tables_dir.glob("*.csv"))
    for path in tqdm(csv_files, desc="Building table metadata", unit="table"):
        try:
            yield _build_table_doc(path, tables_dir=tables_dir, sample_rows=sample_rows)
        except Exception as exc:
            LOGGER.warning("Skipping '%s' due to error: %s", path, exc)


def _build_index(name: str, vector_dim: int) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(
            name="doc_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="table_name",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="metadata_json", type=SearchFieldDataType.String),
        SimpleField(name="row_count", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
        SimpleField(name="column_count", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
        SearchField(
            name=VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dim,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=VECTOR_ALGORITHM_NAME)],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=VECTOR_ALGORITHM_NAME,
            )
        ],
    )
    return SearchIndex(name=name, fields=fields, vector_search=vector_search)


def ensure_search_index(
    index_client: SearchIndexClient,
    index_name: str,
    vector_dim: int,
    recreate: bool = False,
) -> None:
    if recreate:
        try:
            index_client.delete_index(index_name)
            LOGGER.info("Deleted existing Azure Search index '%s'", index_name)
        except ResourceNotFoundError:
            pass

    existing = None
    try:
        existing = index_client.get_index(index_name)
    except ResourceNotFoundError:
        existing = None

    if existing is not None:
        existing_dim = None
        for field in existing.fields:
            if field.name == VECTOR_FIELD:
                existing_dim = getattr(field, "vector_search_dimensions", None)
                break
        if existing_dim != vector_dim:
            raise RuntimeError(
                f"Azure index '{index_name}' already exists with vector dim={existing_dim}, "
                f"but current model dim is {vector_dim}. Use --recreate-index."
            )
        LOGGER.info("Azure Search index '%s' already exists", index_name)
        return

    index_client.create_index(_build_index(index_name, vector_dim))
    LOGGER.info("Created Azure Search index '%s' (vector dim=%d)", index_name, vector_dim)


def _resolve_azure_settings(endpoint: Optional[str], api_key: Optional[str]) -> tuple[str, str]:
    resolved_endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
    resolved_key = api_key or os.getenv("AZURE_SEARCH_API_KEY")
    if not resolved_endpoint:
        raise RuntimeError(
            "Azure Search endpoint is missing. Set AZURE_SEARCH_ENDPOINT or pass --azure-endpoint."
        )
    if not resolved_key:
        raise RuntimeError(
            "Azure Search API key is missing. Set AZURE_SEARCH_API_KEY or pass --azure-api-key."
        )
    return resolved_endpoint, resolved_key


def _doc_to_search_doc(doc: TableMetadataDoc, vector: np.ndarray) -> Dict[str, Any]:
    return {
        "id": doc.id,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "source": doc.source,
        "table_name": doc.table_name,
        "content": doc.content,
        "metadata_json": doc.metadata_json,
        "row_count": doc.row_count,
        "column_count": doc.column_count,
        VECTOR_FIELD: vector.astype(np.float32, copy=False).tolist(),
    }


def _upload_with_retries(
    search_client: SearchClient,
    docs: Sequence[Dict[str, Any]],
    max_retries: int = 5,
) -> None:
    for attempt in range(max_retries):
        try:
            results = search_client.upload_documents(list(docs))
            failed = [result for result in results if not result.succeeded]
            if failed:
                msg = "; ".join(f"{entry.key}: {entry.error_message}" for entry in failed[:3])
                raise RuntimeError(f"{len(failed)} documents failed indexing: {msg}")
            return
        except Exception as exc:
            if attempt + 1 == max_retries:
                raise
            delay = min(2**attempt, 20)
            LOGGER.warning("Upload attempt %d failed: %s. Retrying in %ss", attempt + 1, exc, delay)
            time.sleep(delay)


def _chunked(items: Sequence[TableMetadataDoc], chunk_size: int) -> Iterator[Sequence[TableMetadataDoc]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def run(
    *,
    tables_dir: Path,
    index_name: str,
    embedding_model: str,
    azure_endpoint: Optional[str],
    azure_api_key: Optional[str],
    sample_rows: int,
    max_tables: Optional[int],
    embed_batch_size: int,
    upload_batch_size: int,
    recreate_index: bool,
    device: str,
    dtype: str,
) -> None:
    _require_azure_search_sdk()
    load_dotenv()
    endpoint, key = _resolve_azure_settings(azure_endpoint, azure_api_key)
    credential = AzureKeyCredential(key)

    docs = list(iter_table_docs(tables_dir=tables_dir, sample_rows=sample_rows))
    if max_tables is not None:
        docs = docs[:max_tables]
    if not docs:
        raise RuntimeError(f"No table metadata docs found under {tables_dir}")
    LOGGER.info("Prepared %d table metadata docs.", len(docs))

    embedder = QwenLocalEmbedder(
        model_name=embedding_model,
        max_length=1024,
        device=device,
        dtype=dtype,
    )

    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    ensure_search_index(
        index_client=index_client,
        index_name=index_name,
        vector_dim=embedder.dimension,
        recreate=recreate_index,
    )
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    pending: list[Dict[str, Any]] = []
    processed = 0
    progress = tqdm(total=len(docs), desc="Embedding/uploading tables", unit="table")
    for batch in _chunked(docs, embed_batch_size):
        vectors = embedder.embed([doc.content for doc in batch])
        for doc, vector in zip(batch, vectors, strict=True):
            pending.append(_doc_to_search_doc(doc, vector))
            if len(pending) >= upload_batch_size:
                _upload_with_retries(search_client, pending)
                pending = []
        processed += len(batch)
        progress.update(len(batch))
    if pending:
        _upload_with_retries(search_client, pending)
    progress.close()
    LOGGER.info("Completed table metadata indexing: %d docs uploaded to '%s'.", processed, index_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DrugBank table metadata Azure Search index.")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-key", default=None)
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--max-tables", type=int, default=None)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--upload-batch-size", type=int, default=128)
    parser.add_argument("--recreate-index", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run(
        tables_dir=Path(args.tables_dir).expanduser().resolve(),
        index_name=args.index_name,
        embedding_model=args.embedding_model,
        azure_endpoint=args.azure_endpoint,
        azure_api_key=args.azure_api_key,
        sample_rows=args.sample_rows,
        max_tables=args.max_tables,
        embed_batch_size=args.embed_batch_size,
        upload_batch_size=args.upload_batch_size,
        recreate_index=args.recreate_index,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    raise SystemExit(code)
