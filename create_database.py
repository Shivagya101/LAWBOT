import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

# --- New Imports to fix FAISS initialization ---
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2

# --- Step 1: Load raw PDF documents (No change here) ---
DATA_PATH = "data/"
DB_FAISS_PATH = "vector_store/db_faiss"

def load_pdf_files(data_path):
    """
    Loads all PDF documents from a specified directory.
    """
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# --- Step 2: Define the Hierarchical Chunking Strategy (No change here) ---
def get_retriever(vectorstore, documents):
    """
    Initializes the ParentDocumentRetriever with a parent-child splitting strategy.
    """
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print("Adding documents to the retriever...")
    retriever.add_documents(documents, ids=None)
    print("Documents added successfully.")
    
    return retriever

# --- Step 3: Create Vector Embeddings (No change here) ---
def get_embedding_model():
    """
    Initializes the HuggingFace embedding model.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading PDF documents...")
    documents = load_pdf_files(DATA_PATH)
    print(f"Loaded {len(documents)} pages from PDF files.")

    embedding_model = get_embedding_model()

    # --- Step 4: Create and Store Embeddings using the Parent-Child Strategy ---
    # --- MODIFICATION START ---
    # The original code created an empty FAISS object incorrectly.
    # We now initialize it with the necessary components so it can accept documents later.

    # 1. Get the embedding size from the model
    embedding_size = embedding_model._client.get_sentence_embedding_dimension()

    # 2. Define a FAISS index. IndexFlatL2 is a standard choice.
    index = IndexFlatL2(embedding_size)

    # 3. Instantiate the components FAISS needs to accept new documents.
    faiss_docstore = InMemoryDocstore()
    index_to_docstore_id = {}

    # 4. Create the FAISS vector store with the required components.
    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=faiss_docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    # --- MODIFICATION END ---

    retriever = get_retriever(vectorstore, documents)

    print(f"Saving FAISS index to {DB_FAISS_PATH}...")
    retriever.vectorstore.save_local(DB_FAISS_PATH)
    print("Vector database created and saved successfully.")