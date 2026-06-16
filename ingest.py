import os
import ssl

# Temporary SSL fix for macOS certificate issue
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# ============================================================
# STEP 1: LOAD PDF
# ============================================================
# PyPDFLoader reads each page of the PDF as a separate "Document".
# Each Document has: page_content (the text) + metadata (page number, etc.)


documents = []

for filename in os.listdir("data"):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join("data",filename)

        loader = PyPDFLoader(pdf_path)

        pdf_docs = loader.load()

        # Add source metadata
        for doc in pdf_docs:

            doc.metadata["source"] = filename

        documents.extend(pdf_docs)

print(
    f"Loaded {len(documents)} pages from all PDFs"
)
# ============================================================
# STEP 2: SPLIT INTO CHUNKS
# ============================================================
# Why split? LLMs have limited context windows, and smaller chunks
# allow more precise retrieval. 
#
# chunk_size=500  → each chunk is ~500 characters
# chunk_overlap=100 → chunks overlap by 100 chars so we don't
#                      lose meaning at chunk boundaries
#
# RecursiveCharacterTextSplitter tries to split at paragraphs first,
# then sentences, then words — keeping text as meaningful as possible.

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = SemanticChunker(
    embeddings
)

docs = splitter.split_documents(
    documents
)
print(f"Split into {len(docs)} chunks")

# ============================================================
# STEP 3: SAVE TO DISK
# ============================================================
# Save so we don't have to re-process the PDF every time.
# Creates a "vectorstore/" folder with index files.
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

vectorstore.save_local("vectorstore")
print("Vector database saved to ./vectorstore")