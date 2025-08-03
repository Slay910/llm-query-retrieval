from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

app = FastAPI()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENV
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") or "us-east-1"

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "query-retrieval"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )

index = pc.Index(index_name)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        file_path = f"temp_files/{file.filename}"
        os.makedirs("temp_files", exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif file.filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        vectorstore = LangchainPinecone.from_documents(
            texts,
            embeddings,
            index_name=index_name,
            namespace="docs"
        )

        return {"message": "File uploaded and processed successfully."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/query")
async def query_llm(query: str = Form(...)):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
            namespace="docs"
        )
        docs = vectorstore.similarity_search(query, k=5)

        llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

        return {"response": response}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
