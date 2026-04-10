#!/usr/bin/env python3
"""DrugBank dual-index RAG + SQL workflow for answering passage questions."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from table_metadata_index import normalize_table_name
from vector_index import QwenLocalEmbedder


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_TABLE_INDEX_NAME = "drug_bank_data_lake_tables"
DEFAULT_PASSAGE_INDEX_NAME = "drug_bank_data_lake"
DEFAULT_API_VERSION = "2024-07-01"
DEFAULT_TOP_K = 5
DEFAULT_SQL_LIMIT = 50
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"
DEFAULT_NL2SQL_MODEL = "openai/gpt-5"
DEFAULT_ANSWER_MODEL = "openai/gpt-5"
DEFAULT_INPUT_CSV = REPO_ROOT / "questions_final" / "passage_processed.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "questions_final_test" / "passage_processed_rag_answers.csv"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "drugbank.db"
DEFAULT_PUBMED_TABLE_MAP = REPO_ROOT / "data" / "Pharma" / "pubmed-drugbank-tables.gt"


@dataclass
class OpenRouterConfig:
    client: OpenAI
    headers: dict[str, str]


@dataclass
class RetrievalResult:
    id: str
    title: str
    source: str
    content: str
    table_name: str | None
    metadata_json: str | None
    pubmed_id: str | None
    filename: str | None
    score: float | None


def build_openrouter_config() -> OpenRouterConfig:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY for OpenRouter.")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    headers: dict[str, str] = {}
    site_url = os.getenv("OPENROUTER_SITE_URL")
    site_name = os.getenv("OPENROUTER_SITE_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name
    return OpenRouterConfig(client=client, headers=headers)


def request_json(method: str, url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise RuntimeError(f"Azure Search request failed ({exc.code}): {error_body}") from exc
    if not response_body:
        return {}
    return json.loads(response_body)


def search_index(
    query_vector: list[float],
    endpoint: str,
    index_name: str,
    api_version: str,
    api_key: str,
    top_k: int,
    vector_field: str,
    select_fields: str,
) -> list[RetrievalResult]:
    url = f"{endpoint}/indexes/{index_name}/docs/search?api-version={api_version}"
    payload = {
        "search": "*",
        "top": top_k,
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": query_vector,
                "fields": vector_field,
                "k": top_k,
            }
        ],
        "select": select_fields,
    }
    response = request_json("POST", url, api_key, payload)
    results: list[RetrievalResult] = []
    for item in response.get("value", []):
        results.append(
            RetrievalResult(
                id=str(item.get("id", "")),
                title=str(item.get("title", "")),
                source=str(item.get("source", "")),
                content=str(item.get("content", "")),
                table_name=item.get("table_name"),
                metadata_json=item.get("metadata_json"),
                pubmed_id=item.get("pubmed_id"),
                filename=item.get("filename"),
                score=item.get("@search.score"),
            )
        )
    return results


def load_pubmed_table_mapping(path: Path) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or "," not in line:
                continue
            pubmed_id, table_file = line.split(",", 1)
            table_stem = Path(table_file.strip()).stem
            table_name = normalize_table_name(table_stem)
            mapping.setdefault(pubmed_id.strip(), set()).add(table_name)
    return {k: sorted(v) for k, v in mapping.items() if v}


def build_passage_context(results: list[RetrievalResult]) -> str:
    if not results:
        return "No relevant passage documents found."
    lines: list[str] = []
    for idx, result in enumerate(results, start=1):
        identifier = result.pubmed_id or result.filename or result.id
        lines.append(f"Passage {idx} ({identifier}):\n{result.content.strip()}")
    return "\n\n".join(lines)


def build_table_context(
    results: list[RetrievalResult],
    linked_tables: list[str],
    live_table_columns: dict[str, list[str]],
) -> str:
    lines: list[str] = []
    if not results and not linked_tables:
        return "No relevant table metadata found."

    for idx, result in enumerate(results, start=1):
        table_name = result.table_name or "n/a"
        lines.append(f"Retrieved table {idx} ({table_name}):\n{result.content.strip()}")

    if linked_tables:
        lines.append("Tables linked to retrieved passages:")
        for table_name in linked_tables:
            lines.append(f"- {table_name}")

    candidate_tables = sorted(set(linked_tables + [r.table_name for r in results if r.table_name]))
    if candidate_tables:
        lines.append("\nLive SQLite schema for candidate tables:")
        for table_name in candidate_tables:
            cols = live_table_columns.get(table_name, [])
            if cols:
                lines.append(f"- {table_name}: {', '.join(cols)}")
            else:
                lines.append(f"- {table_name}: (table not present in SQLite DB)")

    return "\n".join(lines)


def parse_sql_from_response(text: str) -> str:
    code_block = re.search(r"```(?:sql)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    return text.strip()


def _chat_completion_with_retries(
    config: OpenRouterConfig,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_retries: int = 5,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "extra_headers": config.headers,
                "model": model,
                "messages": messages,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = config.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt + 1 == max_retries:
                raise
            time.sleep(min(2**attempt, 10))
    if last_exc:
        raise last_exc
    raise RuntimeError("OpenRouter call failed with no response")


def generate_sql(
    question: str,
    table_context: str,
    passage_context: str,
    config: OpenRouterConfig,
    model: str,
    sql_limit: int,
) -> str:
    system_prompt = (
        "You are an expert data analyst for DrugBank. Generate one SQLite SQL query that helps answer "
        "the question using available DrugBank tables. Return only SQL in a fenced code block."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Table context:\n{table_context}\n\n"
        f"Passage context:\n{passage_context}\n\n"
        "Constraints:\n"
        "- Use SQLite syntax.\n"
        "- Only use table/column names that exist in the table context or live schema context.\n"
        f"- Add LIMIT {sql_limit} unless the user explicitly asks for all rows.\n"
        "- If uncertain, choose the most directly relevant table and produce the best-effort query."
    )
    content = _chat_completion_with_retries(
        config,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return parse_sql_from_response(content)


def get_table_names(db_path: str) -> list[str]:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()


def get_table_columns(db_path: str, table_names: list[str]) -> dict[str, list[str]]:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        columns: dict[str, list[str]] = {}
        for table_name in table_names:
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns[table_name] = [str(row[1]) for row in cursor.fetchall()]
        return columns
    finally:
        connection.close()


def quote_table_names(sql: str, table_names: list[str]) -> str:
    quoted_sql = sql
    for table_name in sorted(table_names, key=len, reverse=True):
        double_quoted = rf'""{re.escape(table_name)}""'
        quoted_sql = re.sub(double_quoted, f'"{table_name}"', quoted_sql)
        unquoted_pattern = rf'(?<!")\b{re.escape(table_name)}\b(?!")'
        quoted_sql = re.sub(unquoted_pattern, f'"{table_name}"', quoted_sql)
    return quoted_sql


def execute_sql(db_path: str, sql: str, fetch_limit: int) -> dict[str, Any]:
    result: dict[str, Any] = {"sql": sql, "rows": [], "columns": []}
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(fetch_limit)
        result["columns"] = [desc[0] for desc in cursor.description or []]
        result["rows"] = rows
    finally:
        connection.close()
    return result


def synthesize_answer(
    question: str,
    sql_result: dict[str, Any],
    table_context: str,
    passage_context: str,
    config: OpenRouterConfig,
    model: str,
) -> dict[str, str]:
    system_prompt = (
        "You are a biomedical data assistant. Use SQL results, table context, and passage context to answer. "
        "Respond with JSON containing keys: direct_answer and reasoning_answer."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"SQL execution result:\n{json.dumps(sql_result, ensure_ascii=False)}\n\n"
        f"Table context:\n{table_context}\n\n"
        f"Passage context:\n{passage_context}\n\n"
        "Answer based on evidence from both structured and unstructured sources when possible."
    )
    content = _chat_completion_with_retries(
        config,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    try:
        parsed = json.loads(content)
        direct = str(parsed.get("direct_answer", "")).strip()
        reasoning = str(parsed.get("reasoning_answer", "")).strip()
    except json.JSONDecodeError:
        direct = content.strip().splitlines()[0] if content.strip() else ""
        reasoning = content.strip()
    return {"direct_answer": direct, "reasoning_answer": reasoning}


def _ensure_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    if "," in text:
        return [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    return [text]


def _extract_pubmed_ids(results: Iterable[RetrievalResult]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for result in results:
        candidate = (result.pubmed_id or result.filename or "").strip()
        if not candidate:
            continue
        if candidate in seen:
            continue
        out.append(candidate)
        seen.add(candidate)
    return out


def _extract_table_names(results: Iterable[RetrievalResult]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for result in results:
        table_name = (result.table_name or "").strip()
        if not table_name:
            continue
        normalized = normalize_table_name(table_name)
        if normalized in seen:
            continue
        out.append(normalized)
        seen.add(normalized)
    return out


def _chunked(sequence: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def _infer_question_family(question_id: object) -> str:
    if question_id is None:
        return ""
    text = str(question_id).strip().lower()
    if not text:
        return ""
    match = re.match(r"[a-zA-Z]+", text)
    return match.group(0).lower() if match else ""


def _should_use_tables(row: pd.Series) -> bool:
    explicit_tables = _ensure_list(row.get("table_id"))
    if explicit_tables:
        return True
    family = _infer_question_family(row.get("question_id"))
    if family in {"passage", "passagehop"}:
        return False
    if "table" in family:
        return True
    # Keep previous behavior for unknown formats.
    return True


@dataclass
class QuestionResult:
    idx: int
    system_answer: str
    direct_answer: str
    reasoning_answer: str
    sql_executed: str
    tables_used: str
    passages_used: str
    sql_skipped: bool


def _process_one_question(
    *,
    idx: int,
    question: str,
    query_vector: list[float],
    endpoint: str,
    api_version: str,
    api_key: str,
    table_index_name: str,
    passage_index_name: str,
    vector_field: str,
    top_k: int,
    sql_limit: int,
    pubmed_table_map: dict[str, list[str]],
    table_columns: dict[str, list[str]],
    table_names: list[str],
    db_path: str,
    openrouter: OpenRouterConfig,
    nl2sql_model: str,
    answer_model: str,
    requires_table_reasoning: bool,
) -> QuestionResult:
    table_results: list[RetrievalResult] = []
    if requires_table_reasoning:
        table_results = search_index(
            query_vector=query_vector,
            endpoint=endpoint,
            index_name=table_index_name,
            api_version=api_version,
            api_key=api_key,
            top_k=top_k,
            vector_field=vector_field,
            select_fields="id,title,source,table_name,content,metadata_json",
        )
    passage_results = search_index(
        query_vector=query_vector,
        endpoint=endpoint,
        index_name=passage_index_name,
        api_version=api_version,
        api_key=api_key,
        top_k=top_k,
        vector_field=vector_field,
        select_fields="id,pubmed_id,filename,content,drug_names",
    )

    retrieved_pubmed_ids = _extract_pubmed_ids(passage_results)
    linked_tables_set: set[str] = set()
    if requires_table_reasoning:
        for pubmed_id in retrieved_pubmed_ids:
            for table_name in pubmed_table_map.get(pubmed_id, []):
                linked_tables_set.add(table_name)
    retrieved_table_names = _extract_table_names(table_results)
    candidate_tables = sorted(set(retrieved_table_names).union(linked_tables_set))

    table_context = (
        build_table_context(
            results=table_results,
            linked_tables=sorted(linked_tables_set),
            live_table_columns=table_columns,
        )
        if requires_table_reasoning
        else "No table context required for this passage-only question."
    )
    passage_context = build_passage_context(passage_results)
    sql_skipped = False
    quoted_sql = ""
    sql_result: dict[str, Any]
    if requires_table_reasoning:
        sql = generate_sql(
            question=question,
            table_context=table_context,
            passage_context=passage_context,
            config=openrouter,
            model=nl2sql_model,
            sql_limit=sql_limit,
        )
        quoted_sql = quote_table_names(sql, table_names)

        try:
            sql_result = execute_sql(db_path, quoted_sql, fetch_limit=sql_limit)
        except sqlite3.Error as exc:
            sql_result = {"sql": quoted_sql, "error": str(exc), "rows": [], "columns": []}
    else:
        sql_skipped = True
        sql_result = {
            "sql": "",
            "rows": [],
            "columns": [],
            "note": "SQL generation/execution skipped for passage-only question.",
        }

    answer_payload = synthesize_answer(
        question=question,
        sql_result=sql_result,
        table_context=table_context,
        passage_context=passage_context,
        config=openrouter,
        model=answer_model,
    )

    combined_answer = answer_payload["direct_answer"]
    if answer_payload["reasoning_answer"]:
        combined_answer = f"{combined_answer}\n\n{answer_payload['reasoning_answer']}".strip()

    return QuestionResult(
        idx=idx,
        system_answer=combined_answer,
        direct_answer=answer_payload["direct_answer"],
        reasoning_answer=answer_payload["reasoning_answer"],
        sql_executed=quoted_sql,
        tables_used=", ".join(candidate_tables) if requires_table_reasoning else "",
        passages_used=", ".join(retrieved_pubmed_ids),
        sql_skipped=sql_skipped,
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Answer DrugBank questions using dual-index RAG + SQL.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--pubmed-table-map", default=str(DEFAULT_PUBMED_TABLE_MAP))
    parser.add_argument("--endpoint", default=os.getenv("AZURE_SEARCH_ENDPOINT"))
    parser.add_argument("--service-name", default=os.getenv("AZURE_SEARCH_SERVICE"))
    parser.add_argument("--api-key", default=os.getenv("AZURE_SEARCH_API_KEY"))
    parser.add_argument("--table-index-name", default=os.getenv("AZURE_SEARCH_TABLE_INDEX", DEFAULT_TABLE_INDEX_NAME))
    parser.add_argument("--passage-index-name", default=os.getenv("AZURE_SEARCH_PASSAGE_INDEX", DEFAULT_PASSAGE_INDEX_NAME))
    parser.add_argument("--vector-field", default="content_vector")
    parser.add_argument("--api-version", default=os.getenv("AZURE_SEARCH_API_VERSION", DEFAULT_API_VERSION))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--nl2sql-model", default=os.getenv("OPENROUTER_NL2SQL_MODEL", DEFAULT_NL2SQL_MODEL))
    parser.add_argument("--answer-model", default=os.getenv("OPENROUTER_ANSWER_MODEL", DEFAULT_ANSWER_MODEL))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--sql-limit", type=int, default=DEFAULT_SQL_LIMIT)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N questions.")
    parser.add_argument("--question-workers", type=int, default=4, help="Parallel workers per file for question processing.")
    parser.add_argument("--embed-batch-size", type=int, default=128, help="Batch size for local query embedding.")
    parser.add_argument("--save-every", type=int, default=50, help="Persist intermediate CSV every N completed rows.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Provide --api-key or AZURE_SEARCH_API_KEY.", file=sys.stderr)
        return 1

    endpoint = args.endpoint
    if not endpoint:
        if not args.service_name:
            print("Provide --endpoint or --service-name.", file=sys.stderr)
            return 1
        endpoint = f"https://{args.service_name}.search.windows.net"

    if not os.path.exists(args.input_csv):
        print(f"Input CSV not found: {args.input_csv}", file=sys.stderr)
        return 1
    if not os.path.exists(args.db_path):
        print(f"SQLite DB not found: {args.db_path}", file=sys.stderr)
        return 1

    try:
        openrouter = build_openrouter_config()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    embedder = QwenLocalEmbedder(
        model_name=args.embedding_model,
        max_length=1024,
        device=args.device,
        dtype=args.dtype,
    )

    pubmed_table_map = load_pubmed_table_mapping(Path(args.pubmed_table_map))
    table_names = get_table_names(args.db_path)
    table_columns = get_table_columns(args.db_path, table_names)

    df = pd.read_csv(args.input_csv)
    if "question" not in df.columns:
        print("Input CSV must contain a 'question' column.", file=sys.stderr)
        return 1
    if args.limit is not None:
        df = df.head(args.limit).copy()

    questions = df["question"].astype(str).tolist()
    requires_table_reasoning = [_should_use_tables(df.iloc[i]) for i in range(len(df))]
    total_questions = len(questions)

    embed_started = time.perf_counter()
    query_vectors: list[list[float]] = []
    for batch_questions in tqdm(_chunked(questions, max(1, args.embed_batch_size)), desc="Embedding questions", unit="batch"):
        batch_vectors = embedder.embed(batch_questions)
        query_vectors.extend([vector.astype(float).tolist() for vector in batch_vectors])
    embed_seconds = time.perf_counter() - embed_started
    if embed_seconds > 0:
        print(
            f"Embedded {total_questions} questions in {embed_seconds:.2f}s "
            f"({total_questions / embed_seconds:.2f} q/s)"
        )
    answers: list[str] = ["" for _ in questions]
    direct_answers: list[str] = ["" for _ in questions]
    reasoning_answers: list[str] = ["" for _ in questions]
    sql_executed: list[str] = ["" for _ in questions]
    tables_used: list[str] = ["" for _ in questions]
    passages_used: list[str] = ["" for _ in questions]
    sql_skipped_flags: list[bool] = [False for _ in questions]

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rag_started = time.perf_counter()
    completed = 0
    progress = tqdm(total=total_questions, desc="Answering questions", unit="question")
    with ThreadPoolExecutor(max_workers=max(1, args.question_workers)) as pool:
        futures: dict[Any, int] = {}
        for i, (question, query_vector, use_tables) in enumerate(
            zip(questions, query_vectors, requires_table_reasoning, strict=True)
        ):
            future = pool.submit(
                _process_one_question,
                idx=i,
                question=question,
                query_vector=query_vector,
                endpoint=endpoint,
                api_version=args.api_version,
                api_key=args.api_key,
                table_index_name=args.table_index_name,
                passage_index_name=args.passage_index_name,
                vector_field=args.vector_field,
                top_k=args.top_k,
                sql_limit=args.sql_limit,
                pubmed_table_map=pubmed_table_map,
                table_columns=table_columns,
                table_names=table_names,
                db_path=args.db_path,
                openrouter=openrouter,
                nl2sql_model=args.nl2sql_model,
                answer_model=args.answer_model,
                requires_table_reasoning=use_tables,
            )
            futures[future] = i

        for future in as_completed(futures):
            row_idx = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # Preserve progress and continue on failures.
                result = QuestionResult(
                    idx=row_idx,
                    system_answer=f"[ERROR] {exc}",
                    direct_answer="",
                    reasoning_answer=f"{exc}",
                    sql_executed="",
                    tables_used="",
                    passages_used="",
                    sql_skipped=False,
                )

            answers[result.idx] = result.system_answer
            direct_answers[result.idx] = result.direct_answer
            reasoning_answers[result.idx] = result.reasoning_answer
            sql_executed[result.idx] = result.sql_executed
            tables_used[result.idx] = result.tables_used
            passages_used[result.idx] = result.passages_used
            sql_skipped_flags[result.idx] = result.sql_skipped

            completed += 1
            progress.update(1)

            if completed % max(1, args.save_every) == 0 or completed == total_questions:
                df["system_answer"] = answers
                df["direct_system_answer"] = direct_answers
                df["reasoning_system_answer"] = reasoning_answers
                df["sql_executed"] = sql_executed
                df["tables_used"] = tables_used
                df["passages_used"] = passages_used
                df["sql_skipped"] = sql_skipped_flags
                df.to_csv(args.output_csv, index=False)

    progress.close()
    rag_seconds = time.perf_counter() - rag_started
    throughput = (total_questions / rag_seconds) if rag_seconds > 0 else float("inf")
    print(
        f"Wrote {len(df)} answers to {args.output_csv}. "
        f"RAG stage: {total_questions} questions in {rag_seconds:.2f}s ({throughput:.2f} q/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
