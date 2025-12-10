import json
import pathlib
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

DATA_PATH = pathlib.Path("data/processed.jsonl")
DB_DIR = pathlib.Path("data/chroma_db")

def load_docs() -> List[dict]:
    docs = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def build_vector_store():
    docs_json = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    from langchain.schema import Document

    docs = [
        Document(
            page_content=d["content"],
            metadata={"id": d["id"], "title": d["title"], "source_url": d["source_url"]}
        )
        for d in docs_json
    ]
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    vectordb.persist()

def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )

if __name__ == "__main__":
    build_vector_store()