import os
import sys
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# CONFIG
# -----------------------------
print(f"Using Python executable: {sys.executable}")

# Load environment variables from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in environment variables (.env file).")

INDEX_NAME = "my-vector-index"          # Pinecone index name
DB_FAISS_PATH = "vector_store/db_faiss" # Path to FAISS store
EMBED_DIM = 384                         # all-MiniLM-L6-v2 embedding size
REGION = "us-east-1"                    # Pinecone region
CLOUD = "aws"
# -----------------------------

# 1️⃣ Initialize embedding model (needed for loading FAISS)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2️⃣ Load FAISS vectorstore
print("📂 Loading FAISS vectorstore...")
faiss_store = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)
print("✅ FAISS index loaded successfully.")

# 3️⃣ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# 4️⃣ Check or create Pinecone index
print("🔍 Checking Pinecone indexes...")
existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    print(f"🛠️ Creating Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )
    print("✅ Index created successfully.")
else:
    print(f"ℹ️ Pinecone index '{INDEX_NAME}' already exists.")

# 5️⃣ Connect to the Pinecone index
index = pc.Index(INDEX_NAME)
print(f"🔗 Connected to index: {INDEX_NAME}")

# 6️⃣ Extract documents and stored embeddings from FAISS
print("📑 Extracting documents and vectors from FAISS...")

valid_docs = []
for doc_id in faiss_store.index_to_docstore_id.values():
    doc = faiss_store.docstore.search(doc_id)
    if doc is not None:
        valid_docs.append(doc)

texts = [doc.page_content for doc in valid_docs]
metadatas = [doc.metadata for doc in valid_docs]

# Number of vectors stored in FAISS
n_vectors = faiss_store.index.ntotal

# Retrieve stored embeddings directly from FAISS (no recomputation)
stored_vectors = faiss_store.index.reconstruct_n(0, n_vectors)

print(f"✅ Retrieved {len(texts)} documents and {len(stored_vectors)} embeddings from FAISS.")

# 7️⃣ Prepare vectors for upload
print("📦 Preparing vectors for Pinecone upload...")
vectors_to_upsert = [
    (str(i), stored_vectors[i].tolist(), metadatas[i])
    for i in range(n_vectors)
]

# 8️⃣ Upload to Pinecone in batches
batch_size = 100
print(f"🚀 Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    try:
        index.upsert(vectors=batch)
        print(f"✅ Upserted batch {i // batch_size + 1}/{(len(vectors_to_upsert) // batch_size) + 1}")
    except Exception as e:
        print(f"⚠️ Error in batch {i // batch_size + 1}: {e}")

print("🎉 Upload complete! Your FAISS embeddings are now in Pinecone.")
