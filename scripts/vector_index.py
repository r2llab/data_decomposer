from __future__ import annotations

import argparse
import hashlib
import inspect
import logging
import math
import os
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
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
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_TARGETS_DIR = REPO_ROOT / "data" / "Pharma" / "pubmed-targets"
DEFAULT_GROUPED_PASSAGES_FILE = REPO_ROOT / "grouped_passages_by_drug.json"
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"
DEFAULT_INDEX_NAME = "drug_bank_data_lake"

VECTOR_FIELD = "content_vector"
VECTOR_PROFILE_NAME = "vector-profile"
VECTOR_ALGORITHM_NAME = "hnsw-default"
POINT_ID_NAMESPACE = uuid.UUID("5f43b7b6-ecaf-4c80-82ef-45ca5f4dbbd2")


@dataclass
class IndexedRecord:
    key: str
    pubmed_id: str
    filename: str
    content: str
    drug_names: list[str]


def _build_doc_key(pubmed_id: str) -> str:
    return f"passage-{uuid.uuid5(POINT_ID_NAMESPACE, pubmed_id).hex}"


def _hash_shard(value: str, num_shards: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_shards


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _load_grouped_passage_lookup(path: Path) -> dict[str, list[str]]:
    """Map passage filename -> sorted list of drug names from grouped_passages_by_drug.json."""
    if not path.exists():
        LOGGER.warning(
            "Grouped passage map not found at '%s'. Continuing without drug-name metadata.",
            path,
        )
        return {}

    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning(
            "Failed to parse grouped passage file '%s': %s. Continuing without drug-name metadata.",
            path,
            exc,
        )
        return {}

    if not isinstance(payload, Mapping):
        LOGGER.warning(
            "Grouped passage file '%s' is not a JSON object. Continuing without drug-name metadata.",
            path,
        )
        return {}

    filename_to_drugs: dict[str, set[str]] = {}
    for drug_name, filenames in payload.items():
        if not isinstance(drug_name, str):
            continue
        if not isinstance(filenames, list):
            continue
        for filename in filenames:
            if not isinstance(filename, str):
                continue
            normalized = Path(filename).name
            if not normalized:
                continue
            filename_to_drugs.setdefault(normalized, set()).add(drug_name.strip())

    return {
        filename: sorted(drugs)
        for filename, drugs in filename_to_drugs.items()
        if drugs
    }


def _list_target_files(targets_dir: Path) -> list[Path]:
    if not targets_dir.exists():
        raise FileNotFoundError(f"PubMed targets directory does not exist: {targets_dir}")
    files = sorted([path for path in targets_dir.glob("Target-*") if path.is_file()])
    return files


def _select_shard_files(
    files: Sequence[Path],
    *,
    num_shards: int,
    shard_id: int,
    shard_strategy: str,
) -> list[Path]:
    if num_shards <= 1:
        return list(files)

    if shard_strategy == "hash":
        return [path for path in files if _hash_shard(path.name, num_shards) == shard_id]
    if shard_strategy == "contiguous":
        start = (len(files) * shard_id) // num_shards
        end = (len(files) * (shard_id + 1)) // num_shards
        return list(files[start:end])
    raise ValueError("Unsupported shard strategy. Expected one of: hash, contiguous")


def _read_passage(path: Path, max_passage_chars: int) -> Optional[str]:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        LOGGER.warning("Failed to read passage file '%s': %s", path, exc)
        return None

    if not content:
        return None
    if max_passage_chars > 0:
        content = content[:max_passage_chars]
    return content


def _iter_passage_records(
    files: Sequence[Path],
    *,
    filename_to_drugs: Mapping[str, list[str]],
    max_passage_chars: int,
) -> Iterator[IndexedRecord]:
    for path in files:
        content = _read_passage(path, max_passage_chars=max_passage_chars)
        if not content:
            continue

        pubmed_id = path.name
        yield IndexedRecord(
            key=_build_doc_key(pubmed_id),
            pubmed_id=pubmed_id,
            filename=path.name,
            content=content,
            drug_names=filename_to_drugs.get(path.name, []),
        )


class QwenLocalEmbedder:
    """Local GPU embedding wrapper for Qwen-family embedding models."""

    def __init__(
        self,
        model_name: str,
        max_length: int,
        normalize: bool = True,
        device: str = "auto",
        dtype: str = "bf16",
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.device = device
        self.dtype_name = dtype

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype '{dtype}'. Use one of {tuple(dtype_map)}.")
        torch_dtype = dtype_map[dtype]

        if device == "auto" and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            device_map: str | Dict[str, int] | None = "auto"
            self.inference_device: Optional[str] = None
            LOGGER.info(
                "CUDA available (%s GPUs). Loading model with device_map=auto",
                torch.cuda.device_count(),
            )
        else:
            device_map = None
            if device == "auto":
                self.inference_device = "cpu"
                LOGGER.warning("CUDA is unavailable; embedding will run on CPU and be much slower.")
            else:
                self.inference_device = device
                LOGGER.info("Loading embedding model on explicit device '%s'", device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        if self.inference_device is not None:
            self.model.to(self.inference_device)
        self.model.eval()

        self._has_custom_encode = callable(getattr(self.model, "encode", None))
        self._encode_signature = (
            inspect.signature(self.model.encode) if self._has_custom_encode else None
        )
        self.dimension = self._infer_dimension()
        LOGGER.info("Embedding dimension detected: %d", self.dimension)

    def _infer_dimension(self) -> int:
        probe = self.embed(["Embedding dimension probe"])
        if probe.ndim != 2 or probe.shape[0] != 1:
            raise RuntimeError("Unexpected embedding shape while probing model dimension")
        return int(probe.shape[1])

    def _encode_custom(self, texts: Sequence[str]) -> np.ndarray:
        if self._encode_signature is None:
            raise RuntimeError("Custom encode signature is unavailable")

        kwargs: dict[str, Any] = {}
        params = self._encode_signature.parameters
        if "batch_size" in params:
            kwargs["batch_size"] = len(texts)
        if "max_length" in params:
            kwargs["max_length"] = self.max_length
        if "convert_to_numpy" in params:
            kwargs["convert_to_numpy"] = True
        if "normalize_embeddings" in params:
            kwargs["normalize_embeddings"] = self.normalize

        embeddings = self.model.encode(list(texts), **kwargs)
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if self.normalize and "normalize_embeddings" not in params:
            matrix = _l2_normalize(matrix)
        return matrix

    def _encode_transformers(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

        if self.inference_device is not None:
            first_device = torch.device(self.inference_device)
        else:
            first_device = next(self.model.parameters()).device
        encoded = {name: tensor.to(first_device) for name, tensor in encoded.items()}

        with torch.inference_mode():
            output = self.model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).type_as(token_embeddings)
            pooled = (token_embeddings * attention_mask).sum(dim=1)
            denom = attention_mask.sum(dim=1).clamp(min=1e-6)
            pooled = pooled / denom
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.float().cpu().numpy()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self._has_custom_encode:
            return self._encode_custom(texts)
        return self._encode_transformers(texts)


def _build_index(name: str, vector_dim: int) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="pubmed_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="drug_names",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SimpleField(name="char_count", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
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

    try:
        index_client.create_index(_build_index(index_name, vector_dim))
        LOGGER.info("Created Azure Search index '%s' (vector dim=%d)", index_name, vector_dim)
        return
    except Exception as exc:
        # Another worker may have created the index concurrently.
        LOGGER.warning("Index creation raised '%s'; rechecking existing index.", exc)

    try:
        existing = index_client.get_index(index_name)
    except ResourceNotFoundError:
        raise RuntimeError(
            f"Failed to create Azure Search index '{index_name}' and it does not exist afterward."
        )

    existing_dim = None
    for field in existing.fields:
        if field.name == VECTOR_FIELD:
            existing_dim = getattr(field, "vector_search_dimensions", None)
            break
    if existing_dim != vector_dim:
        raise RuntimeError(
            f"Azure index '{index_name}' exists after create race but vector dim={existing_dim}, "
            f"expected {vector_dim}."
        )
    LOGGER.info("Azure Search index '%s' already exists after concurrent creation", index_name)


def _to_search_doc(record: IndexedRecord, embedding: np.ndarray) -> Dict[str, Any]:
    return {
        "id": record.key,
        "pubmed_id": record.pubmed_id,
        "filename": record.filename,
        "source_type": "public_abstract",
        "content": record.content,
        "drug_names": record.drug_names,
        "char_count": len(record.content),
        VECTOR_FIELD: embedding.astype(np.float32, copy=False).tolist(),
    }


def _upload_with_retries(
    search_client: SearchClient,
    docs: list[Dict[str, Any]],
    max_retries: int = 5,
) -> None:
    if not docs:
        return

    for attempt in range(max_retries):
        try:
            results = search_client.upload_documents(docs)
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


def _chunked(iterable: Iterable[IndexedRecord], size: int) -> Iterator[list[IndexedRecord]]:
    chunk: list[IndexedRecord] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def ingest_to_azure(
    records: Iterable[IndexedRecord],
    embedder: QwenLocalEmbedder,
    search_client: Optional[SearchClient],
    *,
    embed_batch_size: int,
    upload_batch_size: int,
    upload_workers: int,
    expected_total: Optional[int],
    progress_label: str,
) -> tuple[int, float]:
    upload_buffer: list[Dict[str, Any]] = []
    processed = 0
    started = time.perf_counter()
    upload_pool = (
        ThreadPoolExecutor(max_workers=max(1, upload_workers))
        if search_client is not None and upload_workers > 1
        else None
    )
    pending_uploads: list[Future[None]] = []
    max_pending = max(2, upload_workers * 2)

    def submit_upload(docs: list[Dict[str, Any]]) -> None:
        if search_client is None:
            return
        if upload_pool is None:
            _upload_with_retries(search_client, docs)
            return

        pending_uploads.append(upload_pool.submit(_upload_with_retries, search_client, docs))
        if len(pending_uploads) >= max_pending:
            pending_uploads.pop(0).result()

    progress = tqdm(
        total=expected_total,
        desc=progress_label,
        unit="doc",
        dynamic_ncols=True,
    )

    for batch in _chunked(records, embed_batch_size):
        vectors = embedder.embed([rec.content for rec in batch])
        for rec, vector in zip(batch, vectors, strict=True):
            upload_buffer.append(_to_search_doc(rec, vector))
            if search_client is not None and len(upload_buffer) >= upload_batch_size:
                submit_upload(list(upload_buffer))
                upload_buffer.clear()
        processed += len(batch)
        progress.update(len(batch))

    if search_client is not None and upload_buffer:
        submit_upload(list(upload_buffer))

    for future in pending_uploads:
        future.result()
    if upload_pool is not None:
        upload_pool.shutdown(wait=True)

    progress.close()
    elapsed = time.perf_counter() - started
    return processed, elapsed


def _resolve_azure_settings(
    endpoint: Optional[str],
    api_key: Optional[str],
) -> tuple[str, str]:
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


def _service_storage_stats(index_client: SearchIndexClient) -> tuple[Optional[int], Optional[int]]:
    """Return Azure Search service storage usage/quota in bytes when available."""
    try:
        stats = index_client.get_service_statistics()
    except Exception as exc:
        LOGGER.warning("Unable to fetch Azure Search service stats: %s", exc)
        return None, None

    if not isinstance(stats, dict):
        return None, None
    counters = stats.get("counters", {})
    if not isinstance(counters, dict):
        return None, None

    storage = counters.get("storage_size_counter", {})
    if not isinstance(storage, dict):
        return None, None

    usage = storage.get("usage")
    quota = storage.get("quota")
    try:
        parsed_usage = int(usage) if usage is not None else None
    except (TypeError, ValueError):
        parsed_usage = None
    try:
        parsed_quota = int(quota) if quota is not None else None
    except (TypeError, ValueError):
        parsed_quota = None
    return parsed_usage, parsed_quota


def run(
    *,
    index_name: str = DEFAULT_INDEX_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    targets_dir: Path = DEFAULT_TARGETS_DIR,
    grouped_passages_file: Path = DEFAULT_GROUPED_PASSAGES_FILE,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    sample_size: int = 256,
    sample_only: bool = False,
    limit: Optional[int] = None,
    max_length: int = 1024,
    max_passage_chars: int = 20_000,
    embed_batch_size: int = 96,
    upload_batch_size: int = 256,
    upload_workers: int = 16,
    recreate_index: bool = False,
    no_upload: bool = False,
    device: str = "auto",
    dtype: str = "bf16",
    num_shards: int = 1,
    shard_id: Optional[int] = None,
    shard_strategy: str = "hash",
) -> None:
    load_dotenv()
    if shard_id is None:
        shard_id = 0
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")
    if shard_strategy not in {"hash", "contiguous"}:
        raise ValueError("--shard-strategy must be one of: hash, contiguous")

    endpoint, key = _resolve_azure_settings(azure_endpoint, azure_api_key)
    credential = AzureKeyCredential(key)

    LOGGER.info("Loading embedding model: %s", embedding_model)
    embedder = QwenLocalEmbedder(
        embedding_model,
        max_length=max_length,
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

    usage, quota = _service_storage_stats(index_client)
    if usage is not None and quota is not None:
        LOGGER.info(
            "Azure Search storage usage: %.2f / %.2f MiB",
            usage / (1024 * 1024),
            quota / (1024 * 1024),
        )

    search_client: Optional[SearchClient]
    if no_upload:
        LOGGER.warning("Upload disabled via --no-upload. Embeddings will not be sent to Azure.")
        search_client = None
    else:
        if usage is not None and quota is not None and usage >= quota:
            raise RuntimeError(
                "Azure Search storage quota is already exhausted. "
                "Delete existing indexes/documents or upgrade your service tier before uploading."
            )
        search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    all_files = _list_target_files(targets_dir)
    shard_files = _select_shard_files(
        all_files,
        num_shards=num_shards,
        shard_id=shard_id,
        shard_strategy=shard_strategy,
    )
    if limit is not None:
        shard_files = shard_files[:limit]
    expected_total = len(shard_files)

    if expected_total == 0:
        LOGGER.warning("No passage files selected for indexing. Nothing to do.")
        return

    LOGGER.info("Passage source: %s", targets_dir)
    LOGGER.info("Index name: %s", index_name)
    LOGGER.info("Total selected passage files: %d", expected_total)
    if num_shards > 1:
        LOGGER.info(
            "Shard mode enabled: shard %d/%d (strategy=%s, files=%d)",
            shard_id + 1,
            num_shards,
            shard_strategy,
            len(shard_files),
        )

    filename_to_drugs = _load_grouped_passage_lookup(grouped_passages_file)
    if filename_to_drugs:
        LOGGER.info(
            "Loaded grouped passage metadata for %d passages from %s",
            len(filename_to_drugs),
            grouped_passages_file,
        )

    sample_count = 0
    sample_limit = 0
    if sample_size > 0:
        sample_limit = min(sample_size, expected_total)
        sample_records = _iter_passage_records(
            shard_files[:sample_limit],
            filename_to_drugs=filename_to_drugs,
            max_passage_chars=max_passage_chars,
        )
        sample_count, sample_seconds = ingest_to_azure(
            sample_records,
            embedder,
            search_client,
            embed_batch_size=embed_batch_size,
            upload_batch_size=upload_batch_size,
            upload_workers=upload_workers,
            expected_total=sample_limit,
            progress_label="Sample embedding/upload"
            if search_client
            else "Sample embedding (no upload)",
        )
        if sample_count == 0:
            raise RuntimeError("Sample run produced 0 indexed documents.")

        docs_per_second = sample_count / sample_seconds if sample_seconds > 0 else math.inf
        projected_seconds = expected_total / docs_per_second if docs_per_second > 0 else math.inf
        LOGGER.info(
            "Sample complete: %d docs in %.2fs (%.2f docs/s). Projected full runtime: %s for %d docs.",
            sample_count,
            sample_seconds,
            docs_per_second,
            _format_eta(projected_seconds),
            expected_total,
        )

        if sample_only:
            return

    remaining_files = shard_files[sample_limit:]
    if not remaining_files:
        LOGGER.info("Nothing left to process after sample.")
        return

    records = _iter_passage_records(
        remaining_files,
        filename_to_drugs=filename_to_drugs,
        max_passage_chars=max_passage_chars,
    )
    indexed, seconds = ingest_to_azure(
        records,
        embedder,
        search_client,
        embed_batch_size=embed_batch_size,
        upload_batch_size=upload_batch_size,
        upload_workers=upload_workers,
        expected_total=len(remaining_files),
        progress_label="Full embedding/upload"
        if search_client
        else "Full embedding (no upload)",
    )
    LOGGER.info(
        "Full run complete: indexed %d docs in %.2fs (%.2f docs/s).",
        indexed,
        seconds,
        indexed / seconds if seconds > 0 else float("inf"),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed DrugBank-linked public PubMed abstracts with a Qwen-family embedding model "
            "and index them into Azure AI Search."
        )
    )
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--targets-dir", default=str(DEFAULT_TARGETS_DIR))
    parser.add_argument("--grouped-passages-file", default=str(DEFAULT_GROUPED_PASSAGES_FILE))
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-key", default=None)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-passage-chars", type=int, default=20_000)
    parser.add_argument("--embed-batch-size", type=int, default=96)
    parser.add_argument("--upload-batch-size", type=int, default=256)
    parser.add_argument("--upload-workers", type=int, default=16)
    parser.add_argument("--recreate-index", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--shard-strategy", default="hash", choices=("hash", "contiguous"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _build_worker_cmd(args: argparse.Namespace, shard_id: int, worker_device: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--index-name",
        args.index_name,
        "--embedding-model",
        args.embedding_model,
        "--targets-dir",
        args.targets_dir,
        "--grouped-passages-file",
        args.grouped_passages_file,
        "--sample-size",
        str(args.sample_size),
        "--max-length",
        str(args.max_length),
        "--max-passage-chars",
        str(args.max_passage_chars),
        "--embed-batch-size",
        str(args.embed_batch_size),
        "--upload-batch-size",
        str(args.upload_batch_size),
        "--upload-workers",
        str(args.upload_workers),
        "--dtype",
        args.dtype,
        "--num-shards",
        str(args.num_shards),
        "--shard-id",
        str(shard_id),
        "--shard-strategy",
        args.shard_strategy,
        "--device",
        worker_device,
        "--log-level",
        args.log_level,
    ]

    if args.azure_endpoint:
        cmd.extend(["--azure-endpoint", args.azure_endpoint])
    if args.azure_api_key:
        cmd.extend(["--azure-api-key", args.azure_api_key])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.sample_only:
        cmd.append("--sample-only")
    if args.recreate_index and shard_id == 0:
        cmd.append("--recreate-index")
    if args.no_upload:
        cmd.append("--no-upload")
    return cmd


def _run_sharded_default(args: argparse.Namespace) -> None:
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        raise RuntimeError(
            "Multi-GPU shard mode requested, but no CUDA GPUs were detected. "
            "Pass --num-shards 1 for CPU/single-process mode."
        )

    shard_count = args.num_shards
    if shard_count <= 1:
        return

    if gpu_count < shard_count:
        LOGGER.warning(
            "Requested %d shards but only %d GPUs detected; some GPUs will run multiple shards.",
            shard_count,
            gpu_count,
        )

    LOGGER.info("Launching %d shard workers across %d detected GPUs", shard_count, gpu_count)
    procs: list[tuple[int, int, subprocess.Popen[str]]] = []
    try:
        for shard_id in range(shard_count):
            assigned_gpu = shard_id % gpu_count
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
            env["VECTOR_INDEX_FORCE_OS_EXIT"] = "1"

            worker_device = "cuda:0" if args.device == "auto" else args.device
            cmd = _build_worker_cmd(args, shard_id=shard_id, worker_device=worker_device)
            LOGGER.info(
                "Starting shard %d/%d on GPU %d",
                shard_id + 1,
                shard_count,
                assigned_gpu,
            )
            procs.append((shard_id, assigned_gpu, subprocess.Popen(cmd, env=env)))

        failed = 0
        for shard_id, assigned_gpu, proc in procs:
            return_code = proc.wait()
            if return_code != 0:
                failed += 1
                LOGGER.error(
                    "Shard %d on GPU %d exited with code %d",
                    shard_id,
                    assigned_gpu,
                    return_code,
                )

        if failed:
            raise RuntimeError(f"{failed} shard workers failed")

        LOGGER.info("All shard workers completed successfully.")
    finally:
        for _, _, proc in procs:
            if proc.poll() is None:
                proc.terminate()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if args.shard_id is None and args.num_shards > 1:
        _run_sharded_default(args)
        return

    run(
        index_name=args.index_name,
        embedding_model=args.embedding_model,
        targets_dir=Path(args.targets_dir).expanduser().resolve(),
        grouped_passages_file=Path(args.grouped_passages_file).expanduser().resolve(),
        azure_endpoint=args.azure_endpoint,
        azure_api_key=args.azure_api_key,
        sample_size=args.sample_size,
        sample_only=args.sample_only,
        limit=args.limit,
        max_length=args.max_length,
        max_passage_chars=args.max_passage_chars,
        embed_batch_size=args.embed_batch_size,
        upload_batch_size=args.upload_batch_size,
        upload_workers=args.upload_workers,
        recreate_index=args.recreate_index,
        no_upload=args.no_upload,
        device=args.device,
        dtype=args.dtype,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        shard_strategy=args.shard_strategy,
    )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1

    if os.getenv("VECTOR_INDEX_FORCE_OS_EXIT", "0") == "1":
        os._exit(exit_code)
    sys.exit(exit_code)
