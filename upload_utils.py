import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_uploaded_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    filename = os.path.basename(pdf_path)
    for doc in documents:
        doc.metadata["source"] = filename

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

    splitter = SemanticChunker(
        embeddings
    )

    docs = splitter.split_documents(
        documents
    )

    vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True)

    vectorstore.add_documents(docs)
    vectorstore.save_local("vectorstore")

    return filename
