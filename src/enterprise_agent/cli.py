import argparse
import json
import logging
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from enterprise_agent.embeddings import configured_embedder
from enterprise_agent.ingestion.loader import SOURCES, load_documents
from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import ChunkSettings, SearchFilter
from enterprise_agent.retrieval.service import (
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
)
from enterprise_agent.storage.db import build_lexical_index, connect, initialize
from enterprise_agent.storage.embeddings import index_embeddings
from enterprise_agent.storage.search import SearchStore


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
        "--kind", choices=("keyword", "dense", "both"), default="keyword"
    )
    index.add_argument("--max-chunks", type=positive, default=100)
    index.add_argument("--batch-size", type=positive, default=32)
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument(
        "--method", choices=("keyword", "dense", "hybrid"), default="keyword"
    )
    search.add_argument("--top-k", type=positive, default=10)
    search.add_argument("--source-type", choices=SOURCES)
    search.add_argument("--metadata", default="{}", help="JSONB containment filter")
    search.add_argument("--rrf-constant", type=int, default=60)
    search.add_argument("--candidates", type=positive, default=50)
    return p


def search(db: psycopg.Connection, args: argparse.Namespace) -> None:
    filters = SearchFilter(
        source_type=args.source_type, metadata=json.loads(args.metadata)
    )
    store = SearchStore(db)
    keyword = KeywordRetriever(store)
    retriever = keyword
    if args.method != "keyword":
        dense = DenseRetriever(store, configured_embedder())
        retriever = (
            dense
            if args.method == "dense"
            else HybridRetriever(
                keyword, dense, candidates=args.candidates, constant=args.rrf_constant
            )
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
        return 0
    except (ValueError, OSError, psycopg.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
