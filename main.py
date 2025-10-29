import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

# Load environment variables from .env in project root
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

_index = None
_query_engine = None


def init_index():
    global _index, _query_engine

    # PDF path: default to file in project root, can override via env var PDF_PATH
    default_pdf = os.path.join(os.path.dirname(__file__), "constitution of pakistan.pdf")
    pdf_path = os.environ.get("PDF_PATH", default_pdf)

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # Require OpenAI API key via environment variable (do not hardcode secrets)
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )

    # Allow overriding model names via environment variables
    llm_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    embed_model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

    # Configure LlamaIndex to use OpenAI LLM and embeddings
    Settings.llm = OpenAI(model=llm_model)
    Settings.embed_model = OpenAIEmbedding(model=embed_model)

    # Load document and build index
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    _index = VectorStoreIndex.from_documents(documents)
    _query_engine = _index.as_query_engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_index()
    yield
    # Shutdown (cleanup if needed)


app = FastAPI(title="Llama RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryRequest):
    global _query_engine
    if _query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    try:
        response = _query_engine.query(req.query)
        return {"answer": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))