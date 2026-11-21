import argparse
import logging
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from enterprise_agent.ingestion.loader import SOURCES, load_documents
from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import ChunkSettings
from enterprise_agent.storage.db import build_lexical_index, connect, initialize


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
    return p


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
        return 0
    except (ValueError, OSError, psycopg.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
