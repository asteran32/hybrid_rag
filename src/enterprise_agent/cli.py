import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from enterprise_agent.embeddings import configured_embedder
from enterprise_agent.generation.chat import GenerationUnavailable
from enterprise_agent.generation.contracts import AnswerRequest
from enterprise_agent.generation.runtime import configured_scope, run_answer
from enterprise_agent.ingestion.loader import SOURCES, load_documents
from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import ChunkSettings, SearchFilter
from enterprise_agent.retrieval.service import (
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
    build_retriever,
)
from enterprise_agent.storage.db import build_lexical_index, connect, initialize
from enterprise_agent.storage.embeddings import index_embeddings
from enterprise_agent.storage.search import SearchStore
from enterprise_agent.validation.entailment import VerificationUnavailable


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="enterprise-agent")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db")
    load = commands.add_parser("ingest")
    load.add_argument("--input", required=True, type=Path)
    load.add_argument("--source-type", choices=SOURCES)
    scope = load.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=positive, default=100)
    scope.add_argument("--all-documents", action="store_true")
    load.add_argument("--chunk-size", type=positive, default=1200)
    load.add_argument("--overlap", type=int, default=160)
    load.add_argument("--batch-size", type=positive, default=32)
    load.add_argument("--lexical-index", action="store_true")
    index = commands.add_parser("build-index")
    index.add_argument(
        "--kind", choices=("keyword", "dense", "both", "pageindex"), default="keyword"
    )
    index.add_argument(
        "--input",
        type=Path,
        help="optional source directory for PageIndex field boundaries",
    )
    index.add_argument("--max-chunks", type=positive, default=100)
    index.add_argument("--batch-size", type=positive, default=32)
    search = commands.add_parser("search")
    answer = commands.add_parser(
        "answer", help="generate an answer with verified source citations"
    )
    for command, method, top_k in ((search, "keyword", 10), (answer, "hybrid", 5)):
        command.add_argument("query")
        command.add_argument(
            "--method",
            choices=("keyword", "dense", "hybrid", "pageindex")
            if command is search
            else ("keyword", "dense", "hybrid"),
            default=method,
        )
        command.add_argument("--top-k", type=positive, default=top_k)
        command.add_argument("--source-type", choices=SOURCES)
        command.add_argument(
            "--metadata", default="{}", help="JSONB containment filter"
        )
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--questions", required=True, type=Path)
    evaluation.add_argument("--limit", type=positive, default=20)
    evaluation.add_argument("--chunk-depth", type=positive, default=50)
    evaluation.add_argument("--dataset-version", required=True)
    evaluation.add_argument("--output", required=True, type=Path)
    for command in (search, answer, evaluation):
        command.add_argument("--rrf-constant", type=int, default=60)
        command.add_argument("--candidates", type=positive, default=50)
    return p


def run_evaluation(db: psycopg.Connection, args: argparse.Namespace) -> None:
    # Only the evaluation command imports or reads questions and relevance labels.
    from enterprise_agent.evaluation.runner import (
        evaluate,
        load_questions,
        question_file_sha256,
        write_report,
    )
    from enterprise_agent.storage.snapshot import corpus_snapshot

    if args.output.resolve() == args.questions.resolve():
        raise ValueError(
            "evaluation output must be a separate file from the benchmark questions"
        )
    questions = load_questions(args.questions, args.limit)
    provider = configured_embedder()
    with db.transaction():
        db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        snapshot = corpus_snapshot(db, provider.spec.key)
        gold = set().union(*(q.expected_doc_ids for q in questions))
        present = {
            r["document_id"]
            for r in db.execute(
                "SELECT document_id FROM documents WHERE document_id = ANY(%s::text[])",
                (list(gold),),
            )
        }
        store = SearchStore(db)
        keyword = KeywordRetriever(store)
        dense = DenseRetriever(store, provider)
        hybrid = HybridRetriever(
            keyword, dense, candidates=args.candidates, constant=args.rrf_constant
        )
        results = evaluate(
            {"C0": keyword, "C1": dense, "C2": hybrid},
            questions,
            chunk_depth=args.chunk_depth,
        )
    config = {
        "dataset_version": args.dataset_version,
        "snapshot": snapshot,
        "embedding": provider.spec.model_dump(),
        "questions_file": args.questions.name,
        "questions_sha256": question_file_sha256(args.questions),
        "question_limit": args.limit,
        "question_count": len(questions),
        "chunk_depth": args.chunk_depth,
        "rrf_constant": args.rrf_constant,
        "hybrid_candidate_depth": max(args.candidates, args.chunk_depth),
        "methods": {
            "C0": "english FTS, OR, ts_rank_cd",
            "C1": "exact cosine",
            "C2": "RRF(C0,C1)",
        },
        "questions_with_any_gold_in_corpus": sum(
            bool(q.expected_doc_ids & present) for q in questions
        ),
        "gold_document_ids_in_corpus": len(present),
        "distinct_gold_document_ids": len(gold),
    }
    write_report(
        args.output,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "configuration": config,
            "results": results,
        },
    )
    print(
        json.dumps(
            {name: result["summary"] for name, result in results.items()}, indent=2
        )
    )


def search(db: psycopg.Connection, args: argparse.Namespace) -> None:
    filters = SearchFilter(
        source_type=args.source_type,
        metadata=json.loads(args.metadata),
        scope=configured_scope() if args.method == "pageindex" else None,
    )
    retriever = build_retriever(
        SearchStore(db),
        args.method,
        candidates=args.candidates,
        constant=args.rrf_constant,
    )
    hits = retriever.search(args.query, args.top_k, filters)
    print(json.dumps([hit.model_dump() for hit in hits], ensure_ascii=False, indent=2))


def main() -> int:
    load_dotenv(Path.cwd() / ".env")
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        with connect() as db:
            if args.command == "init-db":
                initialize(db)
            elif args.command == "ingest":
                settings = ChunkSettings(size=args.chunk_size, overlap=args.overlap)
                stats = ingest(
                    db,
                    load_documents(args.input, args.source_type),
                    settings,
                    limit=None if args.all_documents else args.limit,
                    batch_size=args.batch_size,
                )
                if args.lexical_index:
                    build_lexical_index(db)
                print(stats.model_dump_json())
                return int(stats.errors > 0)
            elif args.command == "build-index":
                if args.kind == "pageindex":
                    from enterprise_agent.storage.pageindex import build_pageindex

                    result = build_pageindex(db, args.input, args.batch_size)
                    print(json.dumps(result))
                    return int(result["errors"] > 0)
                if args.kind in ("keyword", "both"):
                    build_lexical_index(db)
                if args.kind in ("dense", "both"):
                    count = index_embeddings(
                        db,
                        configured_embedder(),
                        max_chunks=args.max_chunks,
                        batch_size=args.batch_size,
                    )
                    print(json.dumps({"embeddings_written": count}))
            elif args.command == "search":
                search(db, args)
            elif args.command == "answer":
                request = AnswerRequest(
                    question=args.query,
                    method=args.method,
                    top_k=args.top_k,
                    source_type=args.source_type,
                    metadata=json.loads(args.metadata),
                    candidates=args.candidates,
                    rrf_constant=args.rrf_constant,
                )
                print(run_answer(db, request).model_dump_json(indent=2))
            elif args.command == "evaluate":
                run_evaluation(db, args)
        return 0
    except (
        ValueError,
        OSError,
        psycopg.Error,
        GenerationUnavailable,
        VerificationUnavailable,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
