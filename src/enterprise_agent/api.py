import logging
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from enterprise_agent.generation.chat import GenerationUnavailable
from enterprise_agent.generation.contracts import AnswerRequest, GroundedAnswer
from enterprise_agent.generation.runtime import run_answer
from enterprise_agent.storage.db import connect


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(Path.cwd() / ".env")
    yield


app = FastAPI(title="Enterprise Knowledge Agent", lifespan=lifespan)


@app.post("/answer", response_model=GroundedAnswer)
def answer(request: AnswerRequest) -> GroundedAnswer:
    try:
        with connect() as db:
            return run_answer(db, request)
    except (GenerationUnavailable, ValueError, OSError, psycopg.Error) as exc:
        logging.getLogger(__name__).warning(
            "Answer service unavailable (%s)", type(exc).__name__
        )
        raise HTTPException(
            status_code=503, detail="Answer service unavailable"
        ) from None
