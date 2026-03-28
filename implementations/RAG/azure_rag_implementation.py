import json
import logging
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import HnswAlgorithmConfiguration
from azure.search.documents.indexes.models import SearchField
from azure.search.documents.indexes.models import SearchFieldDataType
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.indexes.models import SearchableField
from azure.search.documents.indexes.models import SimpleField
from azure.search.documents.indexes.models import VectorSearch
from azure.search.documents.indexes.models import VectorSearchProfile
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import OpenAI

from core.base_implementation import BaseImplementation
from core.cost_tracker import CostTracker


class AzureSearchRAGImplementation(BaseImplementation):
    """RAG implementation backed by Azure AI Search with OpenRouter embeddings/generation."""

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _resolve_config_or_env(self, config_key: str, env_key: str) -> Optional[str]:
        value = self.config.get(config_key)
        if isinstance(value, str):
            stripped = value.strip()
            # YAML !env fallback returns the env var name itself when unset.
            if stripped and stripped != env_key:
                return stripped
        env_value = os.getenv(env_key)
        return env_value.strip() if isinstance(env_value, str) and env_value.strip() else None

    def initialize(self) -> None:
        load_dotenv()

        self.search_endpoint = self._resolve_config_or_env(
            "azure_search_endpoint", "AZURE_SEARCH_ENDPOINT"
        )
        self.search_api_key = self._resolve_config_or_env(
            "azure_search_api_key", "AZURE_SEARCH_API_KEY"
        )
        self.openrouter_api_key = self._resolve_config_or_env(
            "openrouter_api_key", "OPENROUTER_API_KEY"
        )

        if not self.search_endpoint or not self.search_api_key:
            raise ValueError(
                "Azure Search credentials are required. Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY."
            )
        if not self.search_endpoint.startswith("http"):
            raise ValueError(
                "AZURE_SEARCH_ENDPOINT must be a full URL like 'https://<service>.search.windows.net'."
            )
        if not self.openrouter_api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY in environment or config."
            )

        self.index_name = self.config.get("index_name", "pharma-rag-openrouter")
        self.data_path = Path(self.config.get("data_path", "data/Pharma"))

        self.embedding_model = self.config.get(
            "embedding_model", "openai/text-embedding-3-small"
        )
        self.generation_model = self.config.get("generation_model", "openai/gpt-5.2")
        self.verbose = bool(self.config.get("verbose", False))
        self.quiet_logs = bool(self.config.get("quiet_logs", True))
        self.show_usage_summary = bool(self.config.get("show_usage_summary", False))

        if self.quiet_logs:
            for logger_name in [
                "azure",
                "azure.core",
                "azure.core.pipeline.policies.http_logging_policy",
                "httpx",
                "openai",
            ]:
                logging.getLogger(logger_name).setLevel(logging.WARNING)

        self.embedding_batch_size = int(self.config.get("embedding_batch_size", 32))
        self.embedding_max_retries = int(self.config.get("embedding_max_retries", 3))
        self.embedding_retry_backoff_seconds = float(
            self.config.get("embedding_retry_backoff_seconds", 1.0)
        )
        self.embedding_max_chars = int(self.config.get("embedding_max_chars", 12000))
        self.upload_batch_size = int(self.config.get("upload_batch_size", 200))

        self.k = int(self.config.get("k", 5))
        self.min_score = float(self.config.get("min_score", 0.0))
        self.use_hybrid_search = bool(self.config.get("use_hybrid_search", True))
        self.vector_field = self.config.get("vector_field")
        self.content_field = self.config.get("content_field", "content")
        self.source_field = self.config.get("source_field")
        self.source_type_field = self.config.get("source_type_field")
        self.chunk_index_field = self.config.get("chunk_index_field", "chunk_index")

        self.text_chunk_size = int(self.config.get("text_chunk_size", 1800))
        self.text_chunk_overlap = int(self.config.get("text_chunk_overlap", 200))
        self.max_table_rows = int(self.config.get("max_table_rows", 240))
        self.table_rows_per_chunk = int(self.config.get("table_rows_per_chunk", 80))
        self.max_files = int(self.config.get("max_files", 0))
        self.max_docs = int(self.config.get("max_docs", 0))

        self.auto_build_index = bool(self.config.get("build_index_on_startup", False))
        self.force_recreate_index = bool(self.config.get("force_recreate_index", False))

        self.cost_tracker = CostTracker()
        self.cost_tracker.reset_query_stats()

        credential = AzureKeyCredential(self.search_api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=credential,
        )
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=credential,
        )

        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_api_key,
        )

        self.embedding_calls = 0
        self.embedding_tokens = 0
        self.chat_calls = 0
        self.chat_tokens = 0
        self.id_field = "id"
        self.select_fields: List[str] = [self.id_field, self.content_field]

        index_exists = self._index_exists()
        if self.force_recreate_index and index_exists:
            self._log(f"Deleting existing Azure index: {self.index_name}")
            self.index_client.delete_index(self.index_name)
            index_exists = False

        if not index_exists:
            if not self.auto_build_index:
                self._log(
                    f"Index '{self.index_name}' does not exist. "
                    "Set build_index_on_startup=true to create and load it automatically."
                )
            else:
                self._log(f"Creating and loading Azure index '{self.index_name}'...")
                self.build_index()
        else:
            self._log(f"Loaded Azure index: {self.index_name}")
            self._refresh_index_schema()

    def _index_exists(self) -> bool:
        return any(idx.name == self.index_name for idx in self.index_client.list_indexes())

    def _create_index(self, embedding_dimensions: int) -> None:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=embedding_dimensions,
                vector_search_profile_name="content-vector-profile",
            ),
            SearchableField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="source_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="chunk_index",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="file_path",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[
                VectorSearchProfile(
                    name="content-vector-profile",
                    algorithm_configuration_name="hnsw",
                )
            ],
        )

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )
        self.index_client.create_or_update_index(index)
        self._log(
            f"Azure index '{self.index_name}' created/updated with vector dimensions={embedding_dimensions}."
        )

    def _refresh_index_schema(self) -> None:
        index = self.index_client.get_index(self.index_name)
        fields = index.fields
        field_names = [f.name for f in fields]
        field_set = set(field_names)

        key_field = next((f.name for f in fields if getattr(f, "key", False)), "id")
        self.id_field = key_field

        vector_candidates = [
            f.name for f in fields if getattr(f, "vector_search_dimensions", None)
        ]
        if self.vector_field and self.vector_field in field_set:
            chosen_vector = self.vector_field
        elif "content_vector" in field_set:
            chosen_vector = "content_vector"
        elif "dense_vector" in field_set:
            chosen_vector = "dense_vector"
        elif vector_candidates:
            chosen_vector = vector_candidates[0]
        else:
            raise RuntimeError(
                f"Index '{self.index_name}' does not contain a vector field."
            )
        self.vector_field = chosen_vector

        if self.content_field not in field_set:
            self.content_field = "content" if "content" in field_set else field_names[0]

        if self.source_field and self.source_field in field_set:
            source_field = self.source_field
        else:
            source_field = next(
                (n for n in ["source", "source_id", "title", "file_path"] if n in field_set),
                None,
            )
        self.source_field = source_field

        if self.source_type_field and self.source_type_field in field_set:
            source_type_field = self.source_type_field
        else:
            source_type_field = next(
                (n for n in ["source_type", "doc_type", "table_name"] if n in field_set),
                None,
            )
        self.source_type_field = source_type_field

        if self.chunk_index_field not in field_set:
            self.chunk_index_field = None

        select_fields = [self.id_field, self.content_field]
        if self.source_field:
            select_fields.append(self.source_field)
        if self.source_type_field:
            select_fields.append(self.source_type_field)
        if self.chunk_index_field:
            select_fields.append(self.chunk_index_field)
        self.select_fields = select_fields

        self._log(
            f"Azure index schema resolved: vector_field={self.vector_field}, "
            f"content_field={self.content_field}, source_field={self.source_field}, "
            f"source_type_field={self.source_type_field}, key_field={self.id_field}"
        )

    def _normalize_model_name(self, model_name: str) -> str:
        return model_name.split("/")[-1]

    def _extract_message_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                else:
                    text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
            return "\n".join(parts).strip()
        return str(content).strip()

    def _prepare_embedding_text(self, text: str) -> str:
        prepared = (text or "").strip()
        if self.embedding_max_chars > 0 and len(prepared) > self.embedding_max_chars:
            prepared = prepared[: self.embedding_max_chars]
        return prepared

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        prepared_texts = [self._prepare_embedding_text(text) for text in texts]
        last_error: Optional[Exception] = None

        for attempt in range(1, self.embedding_max_retries + 1):
            try:
                response = self.openrouter_client.embeddings.create(
                    model=self.embedding_model,
                    input=prepared_texts,
                    encoding_format="float",
                )
                self.embedding_calls += 1

                usage = getattr(response, "usage", None)
                if usage:
                    total_tokens = int(
                        getattr(usage, "total_tokens", 0)
                        or getattr(usage, "prompt_tokens", 0)
                    )
                    self.embedding_tokens += total_tokens

                self.cost_tracker.track_embedding_call(
                    model=self._normalize_model_name(self.embedding_model),
                    input_count=len(prepared_texts),
                )

                data = getattr(response, "data", None)
                if not data:
                    raise RuntimeError("Embedding response missing data.")

                vectors: List[List[float]] = []
                for item in data:
                    embedding = getattr(item, "embedding", None)
                    if embedding is None:
                        raise RuntimeError("Embedding item is missing vector data.")
                    vectors.append(list(embedding))

                if len(vectors) != len(prepared_texts):
                    raise RuntimeError(
                        f"Embedding vector count mismatch: expected {len(prepared_texts)}, got {len(vectors)}"
                    )
                return vectors
            except Exception as exc:
                last_error = exc
                if attempt >= self.embedding_max_retries:
                    break
                sleep_seconds = self.embedding_retry_backoff_seconds * attempt
                time.sleep(sleep_seconds)

        raise RuntimeError(
            f"Failed embedding batch of {len(prepared_texts)} texts after "
            f"{self.embedding_max_retries} attempts: {last_error}"
        )

    def _chunk_text(self, text: str) -> List[str]:
        content = text.strip()
        if not content:
            return []
        if len(content) <= self.text_chunk_size:
            return [content]

        chunks: List[str] = []
        step = max(1, self.text_chunk_size - self.text_chunk_overlap)
        start = 0
        while start < len(content):
            end = min(len(content), start + self.text_chunk_size)
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(content):
                break
            start += step
        return chunks

    def _is_binary_file(self, path: Path) -> bool:
        try:
            with open(path, "rb") as handle:
                sample = handle.read(2048)
            if b"\x00" in sample:
                return True
            # Treat files with many non-printable bytes as binary.
            non_text = sum(
                1 for byte in sample if byte < 9 or (13 < byte < 32) or byte > 126
            )
            return len(sample) > 0 and (non_text / len(sample)) > 0.30
        except Exception:
            return True

    def _read_text_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")

    def _table_to_documents(self, file_path: Path, relative_path: str) -> List[Dict[str, Any]]:
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            self._log(f"Skipping table {relative_path}: {exc}")
            return []

        if df.empty:
            return []

        limited = df.head(self.max_table_rows) if self.max_table_rows > 0 else df
        columns = [str(c) for c in limited.columns]

        docs: List[Dict[str, Any]] = []
        chunk_idx = 0
        for start in range(0, len(limited), self.table_rows_per_chunk):
            chunk_df = limited.iloc[start : start + self.table_rows_per_chunk]
            row_lines = chunk_df.fillna("").astype(str).agg(" | ".join, axis=1).tolist()
            text = (
                f"Table Source: {relative_path}\n"
                f"Columns: {', '.join(columns)}\n"
                f"Rows:\n" + "\n".join(row_lines)
            )
            docs.append(
                {
                    "content": text,
                    "source": relative_path,
                    "source_type": "table",
                    "chunk_index": chunk_idx,
                    "file_path": str(file_path),
                }
            )
            chunk_idx += 1
            if self.max_docs > 0 and len(docs) >= self.max_docs:
                break
        return docs

    def _text_to_documents(self, file_path: Path, relative_path: str) -> List[Dict[str, Any]]:
        if self._is_binary_file(file_path):
            return []

        try:
            text = self._read_text_file(file_path)
        except Exception as exc:
            self._log(f"Skipping text {relative_path}: {exc}")
            return []

        chunks = self._chunk_text(text)
        docs: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "content": f"Text Source: {relative_path}\n{chunk}",
                    "source": relative_path,
                    "source_type": "text",
                    "chunk_index": idx,
                    "file_path": str(file_path),
                }
            )
            if self.max_docs > 0 and len(docs) >= self.max_docs:
                break
        return docs

    def _iter_source_files(self) -> Iterable[Path]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        count = 0
        for root, _, files in os.walk(self.data_path):
            for name in files:
                if name.endswith(".zip"):
                    continue
                path = Path(root) / name
                yield path
                count += 1
                if self.max_files > 0 and count >= self.max_files:
                    return

    def _collect_documents(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for file_path in self._iter_source_files():
            rel = str(file_path.relative_to(self.data_path))
            if file_path.suffix.lower() == ".csv":
                new_docs = self._table_to_documents(file_path, rel)
            else:
                new_docs = self._text_to_documents(file_path, rel)

            for doc in new_docs:
                raw_key = f"{doc['source']}::{doc['chunk_index']}::{doc['source_type']}"
                doc_id = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()
                doc["id"] = doc_id
                docs.append(doc)
                if self.max_docs > 0 and len(docs) >= self.max_docs:
                    return docs
        return docs

    def _upload_documents(self, docs: List[Dict[str, Any]]) -> None:
        for start in range(0, len(docs), self.upload_batch_size):
            batch_docs = docs[start : start + self.upload_batch_size]
            upload_results = self.search_client.upload_documents(documents=batch_docs)
            failed = [r for r in upload_results if not r.succeeded]
            if failed:
                raise RuntimeError(
                    f"Failed uploading {len(failed)} docs to Azure Search index '{self.index_name}'."
                )

    def _embed_documents_with_fallback(
        self, docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        prepared_docs: List[Dict[str, Any]] = []
        skipped_docs = 0

        for start in range(0, len(docs), self.embedding_batch_size):
            batch = docs[start : start + self.embedding_batch_size]
            try:
                vectors = self._embed_texts([d["content"] for d in batch])
                for item, vector in zip(batch, vectors):
                    payload = dict(item)
                    payload["content_vector"] = vector
                    prepared_docs.append(payload)
            except Exception as batch_exc:
                self._log(
                    f"Batch embedding failed at offset {start} (size={len(batch)}): {batch_exc}. "
                    "Falling back to per-document embedding."
                )
                for item in batch:
                    try:
                        vector = self._embed_texts([item["content"]])[0]
                        payload = dict(item)
                        payload["content_vector"] = vector
                        prepared_docs.append(payload)
                    except Exception as item_exc:
                        skipped_docs += 1
                        self._log(
                            f"Skipping doc id={item.get('id')} source={item.get('source')}: {item_exc}"
                        )

        return {
            "prepared_docs": prepared_docs,
            "skipped_docs": skipped_docs,
        }

    def build_index(self) -> Dict[str, Any]:
        docs = self._collect_documents()
        if not docs:
            raise RuntimeError(f"No indexable documents found in {self.data_path}")

        probe_vector: Optional[List[float]] = None
        for probe_doc in docs[: min(20, len(docs))]:
            try:
                probe_vector = self._embed_texts([probe_doc["content"]])[0]
                break
            except Exception:
                continue
        if probe_vector is None:
            raise RuntimeError("Failed to produce a probe embedding for index creation.")

        try:
            self._create_index(embedding_dimensions=len(probe_vector))
        except HttpResponseError as exc:
            raise RuntimeError(
                f"Failed creating index '{self.index_name}': {exc.message}"
            ) from exc

        # Recreate client in case index was newly created.
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.search_api_key),
        )
        self._refresh_index_schema()

        embed_summary = self._embed_documents_with_fallback(docs)
        prepared_docs = embed_summary["prepared_docs"]
        skipped_docs = int(embed_summary["skipped_docs"])
        if not prepared_docs:
            raise RuntimeError(
                "No documents were successfully embedded for upload."
            )

        self._upload_documents(prepared_docs)

        summary = {
            "index_name": self.index_name,
            "documents_collected": len(docs),
            "documents_indexed": len(prepared_docs),
            "documents_skipped": skipped_docs,
            "source_path": str(self.data_path),
            "embedding_model": self.embedding_model,
        }
        self._log(
            f"Indexed {len(prepared_docs)} documents into Azure Search '{self.index_name}'."
        )
        return summary

    def load_index(self) -> None:
        if not self._index_exists():
            raise RuntimeError(
                f"Azure index '{self.index_name}' does not exist. Build it first."
            )
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.search_api_key),
        )
        self._refresh_index_schema()

    def _search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_vector = self._embed_texts([query])[0]
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields=self.vector_field,
        )

        search_text = query if self.use_hybrid_search else None
        results = self.search_client.search(
            search_text=search_text,
            vector_queries=[vector_query],
            top=top_k,
            select=self.select_fields,
        )

        matches: List[Dict[str, Any]] = []
        for result in results:
            score = float(result.get("@search.score", 0.0))
            if score < self.min_score:
                continue
            matches.append(
                {
                    "id": result.get(self.id_field),
                    "content": result.get(self.content_field, ""),
                    "source": result.get(self.source_field, "") if self.source_field else "",
                    "source_type": (
                        result.get(self.source_type_field, "")
                        if self.source_type_field
                        else ""
                    ),
                    "chunk_index": (
                        result.get(self.chunk_index_field, 0)
                        if self.chunk_index_field
                        else 0
                    ),
                    "relevance_score": score,
                }
            )
        return matches

    def _parse_answer_payload(self, raw_text: str) -> Dict[str, str]:
        text = (raw_text or "").strip()
        short_answer = ""
        answer_reasoning = ""

        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict):
                short_answer = str(
                    parsed.get("short_answer") or parsed.get("answer") or ""
                ).strip()
                answer_reasoning = str(
                    parsed.get("answer_reasoning") or parsed.get("reasoning") or ""
                ).strip()

        if not short_answer and text:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                short_answer = lines[0]
                answer_reasoning = " ".join(lines[1:]).strip()

        if not short_answer:
            short_answer = "No reliable short answer was generated."

        answer = short_answer
        if answer_reasoning:
            answer = f"{short_answer}\n\nReasoning: {answer_reasoning}"

        return {
            "short_answer": short_answer,
            "answer_reasoning": answer_reasoning,
            "answer": answer,
        }

    def _generate_answer(self, query: str, matches: List[Dict[str, Any]]) -> Dict[str, str]:
        if not matches:
            short_answer = (
                "I could not find relevant information in the index to answer this question."
            )
            return {
                "short_answer": short_answer,
                "answer_reasoning": "No retrieved context passed the relevance threshold.",
                "answer": (
                    f"{short_answer}\n\nReasoning: "
                    "No retrieved context passed the relevance threshold."
                ),
            }

        context_blocks = []
        for idx, match in enumerate(matches, start=1):
            context_blocks.append(
                f"[{idx}] source={match['source']} score={match['relevance_score']:.4f}\n{match['content']}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        response = self.openrouter_client.chat.completions.create(
            model=self.generation_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise biomedical QA assistant. "
                        "Answer using only the retrieved context when possible. "
                        "If context is insufficient, say so clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\n"
                        f"Retrieved context:\n{context}\n\n"
                        "Return JSON only with keys `short_answer` and `answer_reasoning`.\n"
                        "Keep short_answer explicit and concise.\n"
                        "Keep answer_reasoning brief and grounded in the provided context."
                    ),
                },
            ],
            temperature=0.2,
        )

        self.chat_calls += 1
        usage = getattr(response, "usage", None)
        prompt_tokens = 0
        completion_tokens = 0
        if usage:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
            completion_tokens = int(getattr(usage, "completion_tokens", 0))
            self.chat_tokens += int(getattr(usage, "total_tokens", 0))

        self.cost_tracker.track_chat_completion_call(
            model=self._normalize_model_name(self.generation_model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={"component": "azure_rag", "query": query},
        )

        content = response.choices[0].message.content if response.choices else ""
        parsed = self._parse_answer_payload(self._extract_message_text(content))
        return parsed

    def process_query(
        self, query: str, ground_truth_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        del ground_truth_answer  # Not used in this implementation.
        self.cost_tracker.reset_query_stats()

        if not self._index_exists():
            raise RuntimeError(
                f"Azure index '{self.index_name}' does not exist. "
                "Enable build_index_on_startup or run build_index()."
            )
        if not self.vector_field:
            self._refresh_index_schema()

        matches = self._search(query, top_k=self.k)
        answer_payload = self._generate_answer(query, matches)

        cost_summary = self.cost_tracker.get_query_summary()
        document_sources = list(dict.fromkeys(m["source"] for m in matches if m["source"]))

        top_score = max((m["relevance_score"] for m in matches), default=0.0)
        confidence = min(1.0, max(0.0, top_score))

        return {
            "answer": answer_payload["answer"],
            "short_answer": answer_payload["short_answer"],
            "answer_reasoning": answer_payload["answer_reasoning"],
            "confidence": confidence,
            "source_type": "azure_search",
            "source": document_sources[0] if document_sources else None,
            "document_sources": document_sources,
            "retrieval_results": matches,
            "cost_metrics": {
                "total_cost": float(cost_summary["query_cost"]),
                "total_tokens": int(cost_summary["query_tokens"]),
                "api_calls": int(cost_summary["query_calls"]),
                "model_breakdown": {
                    model: {
                        "cost": float(stats["cost"]),
                        "tokens": int(stats["tokens"]),
                        "calls": int(stats["calls"]),
                    }
                    for model, stats in cost_summary["models"].items()
                },
                "endpoint_breakdown": {
                    endpoint: {
                        "cost": float(stats["cost"]),
                        "tokens": int(stats["tokens"]),
                        "calls": int(stats["calls"]),
                    }
                    for endpoint, stats in cost_summary["endpoints"].items()
                },
                "raw_counts": {
                    "embedding_calls": self.embedding_calls,
                    "embedding_tokens": self.embedding_tokens,
                    "chat_calls": self.chat_calls,
                    "chat_tokens": self.chat_tokens,
                },
            },
        }

    def cleanup(self) -> None:
        if not self.show_usage_summary:
            return
        summary = self.cost_tracker.get_cost_summary()
        print("\nAzure RAG usage summary:")
        print(f"Total cost: ${summary['total_cost']:.6f}")
        print(f"Total tokens: {summary['total_tokens']}")
        print(f"Total API calls: {summary['total_calls']}")
