#!/usr/bin/env python3
"""ReSP baseline for DrugBank questions (reasoner + retriever + summarizer loop)."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from rag_answer_questions import (
    DEFAULT_ANSWER_MODEL,
    DEFAULT_API_VERSION,
    DEFAULT_DB_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INPUT_CSV,
    DEFAULT_NL2SQL_MODEL,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_PASSAGE_INDEX_NAME,
    DEFAULT_PUBMED_TABLE_MAP,
    DEFAULT_SQL_LIMIT,
    DEFAULT_TABLE_INDEX_NAME,
    DEFAULT_TOP_K,
    OpenRouterConfig,
    QwenLocalEmbedder,
    RetrievalResult,
    _chat_completion_with_retries,
    _chunked,
    _extract_pubmed_ids,
    _extract_table_names,
    _should_use_tables,
    build_openrouter_config,
    build_passage_context,
    build_table_context,
    execute_sql,
    generate_sql,
    get_table_columns,
    get_table_names,
    load_pubmed_table_mapping,
    quote_table_names,
    search_index,
    synthesize_answer,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_RESP_MAX_ITERS = 2
DEFAULT_RESP_MAX_MEMORY_ITEMS = 8
DEFAULT_RESP_SUMMARY_MAX_ITEMS = 3
DEFAULT_RESP_SUMMARY_MAX_CHARS = 500


@dataclass
class RespDecision:
    needs_more_evidence: bool
    sub_question: str
    focus: str  # one of: passage, table, both
    rationale: str


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
    resp_iterations_used: int
    resp_trace_json: str


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16].rstrip() + "\n...[truncated]"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _normalise_focus(value: object, *, requires_tables: bool) -> str:
    text = _safe_text(value).lower()
    if text in {"passage", "passages", "text", "document", "documents"}:
        return "passage"
    if text in {"table", "tables", "sql", "schema", "structured"}:
        return "table" if requires_tables else "passage"
    if text in {"both", "all", "any", "mixed"}:
        return "both" if requires_tables else "passage"
    return "both" if requires_tables else "passage"


def _summarize_passages(
    results: list[RetrievalResult],
    *,
    max_items: int,
    max_chars: int,
) -> str:
    if not results:
        return "No relevant passages retrieved."
    lines: list[str] = []
    for idx, result in enumerate(results[: max(1, max_items)], start=1):
        identifier = result.pubmed_id or result.filename or result.id
        snippet = _truncate(_safe_text(result.content).replace("\n", " "), max_chars)
        lines.append(f"Passage {idx} ({identifier}): {snippet}")
    return "\n".join(lines)


def _summarize_tables(
    results: list[RetrievalResult],
    *,
    max_items: int,
    max_chars: int,
) -> str:
    if not results:
        return "No relevant table metadata retrieved."
    lines: list[str] = []
    for idx, result in enumerate(results[: max(1, max_items)], start=1):
        table_name = _safe_text(result.table_name) or "n/a"
        snippet = _truncate(_safe_text(result.content).replace("\n", " "), max_chars)
        lines.append(f"Table {idx} ({table_name}): {snippet}")
    return "\n".join(lines)


def _reason_next_step(
    *,
    question: str,
    global_memory: list[str],
    local_pathway: list[dict[str, Any]],
    current_iteration: int,
    max_iterations: int,
    requires_tables: bool,
    config: OpenRouterConfig,
    model: str,
) -> RespDecision:
    memory_block = "\n\n".join(global_memory[-8:])
    local_block = "\n".join(
        [
            (
                f"- Iter {entry.get('iteration')}: {entry.get('sub_question')} "
                f"(focus={entry.get('focus')}, passages={entry.get('passage_hits')}, tables={entry.get('table_hits')})"
            )
            for entry in local_pathway[-8:]
        ]
    )

    focus_hint = "Use one of: passage, both." if not requires_tables else "Use one of: passage, table, both."
    system_prompt = (
        "You are a retrieval planner for biomedical QA. Decide whether one more retrieval step is needed. "
        "Return strict JSON only with keys: needs_more_evidence (boolean), sub_question (string), "
        "focus (string), rationale (string). "
        + focus_hint
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Current iteration: {current_iteration}/{max_iterations}\n\n"
        f"Global evidence memory:\n{memory_block or '[empty]'}\n\n"
        f"Local pathway:\n{local_block or '[empty]'}\n\n"
        "If evidence already seems sufficient, set needs_more_evidence=false. Return JSON only."
    )

    content = _chat_completion_with_retries(
        config,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    payload = _extract_json_object(content)
    if not isinstance(payload, dict):
        return RespDecision(
            needs_more_evidence=False,
            sub_question="",
            focus="both" if requires_tables else "passage",
            rationale="Reasoner returned non-JSON; stopping.",
        )

    return RespDecision(
        needs_more_evidence=bool(payload.get("needs_more_evidence", False)),
        sub_question=_safe_text(payload.get("sub_question")),
        focus=_normalise_focus(payload.get("focus"), requires_tables=requires_tables),
        rationale=_safe_text(payload.get("rationale")),
    )


def _merge_unique(existing: list[RetrievalResult], incoming: list[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[str] = {item.id for item in existing if _safe_text(item.id)}
    merged = list(existing)
    for item in incoming:
        item_id = _safe_text(item.id)
        if item_id and item_id in seen:
            continue
        merged.append(item)
        if item_id:
            seen.add(item_id)
    return merged


def _resp_retrieve_loop(
    *,
    question: str,
    query_vector: list[float],
    endpoint: str,
    api_version: str,
    api_key: str,
    table_index_name: str,
    passage_index_name: str,
    vector_field: str,
    top_k: int,
    requires_table_reasoning: bool,
    config: OpenRouterConfig,
    reasoner_model: str,
    max_iters: int,
    max_memory_items: int,
    summary_max_items: int,
    summary_max_chars: int,
) -> tuple[list[RetrievalResult], list[RetrievalResult], list[str], list[dict[str, Any]], list[dict[str, Any]], int]:
    all_passage_results: list[RetrievalResult] = []
    all_table_results: list[RetrievalResult] = []
    global_memory: list[str] = []
    local_pathway: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    current_sub_question = question
    current_focus = "both" if requires_table_reasoning else "passage"
    iterations_used = 0

    for iteration in range(1, max_iters + 1):
        iterations_used = iteration
        use_passages = current_focus in {"passage", "both"}
        use_tables = requires_table_reasoning and current_focus in {"table", "both"}

        passage_results: list[RetrievalResult] = []
        table_results: list[RetrievalResult] = []

        if use_passages:
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
            all_passage_results = _merge_unique(all_passage_results, passage_results)

        if use_tables:
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
            all_table_results = _merge_unique(all_table_results, table_results)

        passage_summary = _summarize_passages(
            passage_results,
            max_items=summary_max_items,
            max_chars=summary_max_chars,
        )
        table_summary = _summarize_tables(
            table_results,
            max_items=summary_max_items,
            max_chars=summary_max_chars,
        )

        memory_entry = "\n".join(
            [
                f"Iteration: {iteration}",
                f"Sub-question: {current_sub_question}",
                f"Focus: {current_focus}",
                f"Passage summary:\n{passage_summary}",
                f"Table summary:\n{table_summary}",
            ]
        )
        global_memory.append(memory_entry)
        if len(global_memory) > max_memory_items:
            global_memory = global_memory[-max_memory_items:]

        local_entry = {
            "iteration": iteration,
            "sub_question": current_sub_question,
            "focus": current_focus,
            "passage_hits": len(passage_results),
            "table_hits": len(table_results),
        }
        local_pathway.append(local_entry)
        if len(local_pathway) > max_memory_items:
            local_pathway = local_pathway[-max_memory_items:]

        trace_step: dict[str, Any] = {
            "iteration": iteration,
            "sub_question": current_sub_question,
            "focus": current_focus,
            "passages": [
                (result.pubmed_id or result.filename or result.id)
                for result in passage_results
            ],
            "tables": [result.table_name or result.id for result in table_results],
        }

        if iteration >= max_iters:
            trace_step["reasoner"] = {
                "needs_more_evidence": False,
                "rationale": "max_iterations_reached",
            }
            trace.append(trace_step)
            break

        decision = _reason_next_step(
            question=question,
            global_memory=global_memory,
            local_pathway=local_pathway,
            current_iteration=iteration,
            max_iterations=max_iters,
            requires_tables=requires_table_reasoning,
            config=config,
            model=reasoner_model,
        )
        trace_step["reasoner"] = {
            "needs_more_evidence": decision.needs_more_evidence,
            "sub_question": decision.sub_question,
            "focus": decision.focus,
            "rationale": decision.rationale,
        }
        trace.append(trace_step)

        if not decision.needs_more_evidence:
            break

        next_sub_question = decision.sub_question
        if not next_sub_question:
            break

        current_sub_question = next_sub_question
        current_focus = decision.focus
        if not requires_table_reasoning and current_focus == "table":
            current_focus = "passage"

    return (
        all_passage_results,
        all_table_results,
        global_memory,
        local_pathway,
        trace,
        iterations_used,
    )


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
    reasoner_model: str,
    nl2sql_model: str,
    answer_model: str,
    requires_table_reasoning: bool,
    max_iters: int,
    max_memory_items: int,
    summary_max_items: int,
    summary_max_chars: int,
) -> QuestionResult:
    (
        passage_results,
        table_results,
        global_memory,
        local_pathway,
        trace,
        iterations_used,
    ) = _resp_retrieve_loop(
        question=question,
        query_vector=query_vector,
        endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
        table_index_name=table_index_name,
        passage_index_name=passage_index_name,
        vector_field=vector_field,
        top_k=top_k,
        requires_table_reasoning=requires_table_reasoning,
        config=openrouter,
        reasoner_model=reasoner_model,
        max_iters=max_iters,
        max_memory_items=max_memory_items,
        summary_max_items=summary_max_items,
        summary_max_chars=summary_max_chars,
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

    if global_memory:
        memory_block = "\n\n".join(global_memory[-4:])
        passage_context = f"{passage_context}\n\nReSP memory:\n{memory_block}".strip()

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
        resp_iterations_used=iterations_used,
        resp_trace_json=json.dumps(trace, ensure_ascii=False),
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Answer DrugBank questions using ReSP baseline.")
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
    parser.add_argument("--reasoner-model", default=os.getenv("OPENROUTER_REASONER_MODEL", DEFAULT_ANSWER_MODEL))
    parser.add_argument("--nl2sql-model", default=os.getenv("OPENROUTER_NL2SQL_MODEL", DEFAULT_NL2SQL_MODEL))
    parser.add_argument("--answer-model", default=os.getenv("OPENROUTER_ANSWER_MODEL", DEFAULT_ANSWER_MODEL))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--sql-limit", type=int, default=DEFAULT_SQL_LIMIT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-workers", type=int, default=4)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--max-iters", type=int, default=DEFAULT_RESP_MAX_ITERS)
    parser.add_argument("--max-memory-items", type=int, default=DEFAULT_RESP_MAX_MEMORY_ITEMS)
    parser.add_argument("--summary-max-items", type=int, default=DEFAULT_RESP_SUMMARY_MAX_ITEMS)
    parser.add_argument("--summary-max-chars", type=int, default=DEFAULT_RESP_SUMMARY_MAX_CHARS)
    args = parser.parse_args()

    if args.max_iters <= 0 or args.max_iters > 2:
        print("--max-iters must be between 1 and 2", file=sys.stderr)
        return 1

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
    for batch_questions in tqdm(
        _chunked(questions, max(1, args.embed_batch_size)),
        desc="Embedding questions",
        unit="batch",
    ):
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
    resp_iterations_used: list[int] = [0 for _ in questions]
    resp_trace_json: list[str] = ["" for _ in questions]

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rag_started = time.perf_counter()
    completed = 0
    progress = tqdm(total=total_questions, desc="ReSP answering", unit="question")
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
                reasoner_model=args.reasoner_model,
                nl2sql_model=args.nl2sql_model,
                answer_model=args.answer_model,
                requires_table_reasoning=use_tables,
                max_iters=args.max_iters,
                max_memory_items=args.max_memory_items,
                summary_max_items=args.summary_max_items,
                summary_max_chars=args.summary_max_chars,
            )
            futures[future] = i

        for future in as_completed(futures):
            row_idx = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = QuestionResult(
                    idx=row_idx,
                    system_answer=f"[ERROR] {exc}",
                    direct_answer="",
                    reasoning_answer=f"{exc}",
                    sql_executed="",
                    tables_used="",
                    passages_used="",
                    sql_skipped=False,
                    resp_iterations_used=0,
                    resp_trace_json=json.dumps(
                        [
                            {
                                "iteration": 0,
                                "error": str(exc),
                            }
                        ],
                        ensure_ascii=False,
                    ),
                )

            answers[result.idx] = result.system_answer
            direct_answers[result.idx] = result.direct_answer
            reasoning_answers[result.idx] = result.reasoning_answer
            sql_executed[result.idx] = result.sql_executed
            tables_used[result.idx] = result.tables_used
            passages_used[result.idx] = result.passages_used
            sql_skipped_flags[result.idx] = result.sql_skipped
            resp_iterations_used[result.idx] = result.resp_iterations_used
            resp_trace_json[result.idx] = result.resp_trace_json

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
                df["resp_iterations_used"] = resp_iterations_used
                df["resp_trace_json"] = resp_trace_json
                df.to_csv(args.output_csv, index=False)

    progress.close()
    rag_seconds = time.perf_counter() - rag_started
    throughput = (total_questions / rag_seconds) if rag_seconds > 0 else float("inf")
    print(
        f"Wrote {len(df)} answers to {args.output_csv}. "
        f"ReSP stage: {total_questions} questions in {rag_seconds:.2f}s ({throughput:.2f} q/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
