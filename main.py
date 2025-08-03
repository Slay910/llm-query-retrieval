from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List
import openai
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore

# Multi-format document loaders
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredFileLoader
)

from pinecone import Pinecone, ServerlessSpec

# Initialize FastAPI
app = FastAPI()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "policy-index"

openai.api_key = OPENAI_API_KEY

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Request model
class QARequest(BaseModel):
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# File loader based on extension
def load_and_chunk_document(file_path):
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path)
    elif ext in ["eml", "msg"]:
        loader = UnstructuredEmailLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path)

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Embed & upsert into Pinecone
def embed_and_upsert(docs):
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeStore.from_documents(docs, embedder, index_name=INDEX_NAME)

# Answer question from stored vectors
def answer_query(question):
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeStore(index_name=INDEX_NAME, embedding=embedder)
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Context:
    {context}

    Q: {question}
    A:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a legal assistant answering questions from policy documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# Endpoint: file upload + question list
@app.post("/api/v1/hackrx/run", response_model=QAResponse)
async def run_query(questions: List[str], file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        chunks = load_and_chunk_document(tmp_path)
        embed_and_upsert(chunks)
        answers = [answer_query(q) for q in questions]
        os.remove(tmp_path)
        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
