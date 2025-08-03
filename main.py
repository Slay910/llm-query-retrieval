from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import openai
import pinecone
import requests
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore

# Initialize app
app = FastAPI()

# Load env keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "policy-index"

openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

# Request & Response Schemas
class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# Helper: Load + Chunk PDF
def load_and_chunk_pdf(pdf_url):
    local_path = "/tmp/temp.pdf"
    with open(local_path, 'wb') as f:
        f.write(requests.get(pdf_url).content)
    loader = PyMuPDFLoader(local_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Helper: Embed & Upsert
def embed_and_upsert(docs):
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeStore.from_documents(docs, embedder, index_name=INDEX_NAME)

# Helper: Retrieve & Answer
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
            {"role": "system", "content": "You are a legal assistant answering questions from policies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# Endpoint
@app.post("/api/v1/hackrx/run", response_model=QAResponse)
def run_query(req: QARequest):
    try:
        chunks = load_and_chunk_pdf(req.documents)
        embed_and_upsert(chunks)
        answers = [answer_query(q) for q in req.questions]
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
