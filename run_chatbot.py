import os
# If you keep credentials in a .env file, try to load them into the environment at runtime.
try:
    # python-dotenv is listed in requirements.txt; this import is optional in environments where
    # environment variables are set externally (CI, system env, etc.).
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        # No .env file found in the project hierarchy.
        pass
except Exception:
    # If python-dotenv isn't installed or fails to load, we continue — the script will still
    # read variables from the actual environment.
    print("python-dotenv not available; relying on environment variables")
    pass
from typing import List, TypedDict, Annotated, Sequence
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
import operator

# Global counter for how many times the LLM is called.
llm_call_count = 0


def invoke_chain(chain, payload: dict):
    """Invoke a chain/llm and increment the global llm_call_count.

    Returns the raw response from chain.invoke(payload).
    """
    global llm_call_count
    resp = chain.invoke(payload)
    llm_call_count += 1
    return resp

# --- Configuration (No changes here) ---
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vector_store/db_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# --- IMPORTANT: Add the path to your PDF data for the retriever setup ---
DATA_PATH = "data/" 

# --- LLM and Retriever Setup (Reused and adapted from previous scripts) ---

def load_chat_model() -> ChatHuggingFace:
    if not HF_TOKEN:
        raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in your environment.")
    endpoint = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",
        temperature=0.1, # Lower temperature for more deterministic grading/rewriting
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=endpoint)

def get_retriever():
    """
    Loads the FAISS vector store and returns a retriever.
    This function now assumes the ParentDocumentRetriever setup was used.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # For this example, we load the FAISS index created by the previous script.
    # In a real application, you would also persist and load the 'docstore'.
    # Since the docstore was in-memory, we can't load it directly.
    # The ParentDocumentRetriever is best used when loaded and passed around,
    # but for this script, we'll use the child-chunk retriever directly.
    # The logic in the graph will handle fetching the full context.
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})

# --- LangGraph Agent Definition ---

# 1. Define the State
# The state is the "memory" of our agent. It's a dictionary that gets passed between nodes.
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List
    iterations: int
    context_too_large: bool

# 2. Define the Nodes
# Each node is a function that performs an action and updates the state.

def retrieve(state):
    """
    Retrieve documents from the vector store.
    """
    print("---NODE: RETRIEVE---")
    question = state["question"]
    # Support multiple retriever APIs (langchain has several: get_relevant_documents, retrieve, etc.)
    try:
        documents = retriever.get_relevant_documents(question)
    except AttributeError:
        try:
            documents = retriever.retrieve(question)
        except AttributeError:
            # Fallback: if the retriever implements an invoke-style API used by the graph
            documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    This is our "critic" node.
    """
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    # Simple grading prompt that returns 'yes' or 'no' in plain text
    grade_prompt = PromptTemplate(
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "If the document contains keywords or clear answers related to the user question, reply with 'yes'.\n"
            "Otherwise, reply with 'no'.\n\n"
            "Retrieved document:\n{document}\n\nUser question:\n{question}\n\nAnswer:" 
        ),
        input_variables=["document", "question"],
    )
    
    # Use the chat LLM directly (no structured output). We'll parse 'yes'/'no' from the model text.
    chain = grade_prompt | llm
    
    filtered_docs = []
    for d in documents:
        resp = invoke_chain(chain, {"question": question, "document": d.page_content})
        print(f"Grading response: {getattr(resp, 'content', str(resp))}")
        # The LLM returns text; normalize and check for 'yes' or 'no'
        text = getattr(resp, "content", str(resp)).strip().lower()
        is_yes = False
        if text.startswith("yes") or " yes" in f" {text} ":
            is_yes = True

        if is_yes:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    
    return {"documents": filtered_docs}

def generate(state):
    """
    Generate an answer using the retrieved documents.
    """
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT, input_variables=["context", "question"]
    )
    
    rag_chain = prompt | llm
    generation = invoke_chain(rag_chain, {"context": documents, "question": question})
    # Ensure we return the text content when available
    print(f"Generation response: {getattr(generation, 'content', str(generation))}")
    return {"generation": getattr(generation, "content", str(generation))}

def rewrite_query(state):
    """
    Transform the query to produce a better question.
    """
    print("---NODE: REWRITE QUERY---")
    question = state["question"]
    
    # Prompt
    system = """You are a query re-writer. Given a user question, your task is to rephrase it to be more
    aligned with the language and terminology found in legal and policy documents.
    Do not answer the question, only rewrite it."""
    
    rewrite_prompt = PromptTemplate(
        template="Original question: {question}",
        input_variables=["question"],
    )
    
    rewriter_chain = rewrite_prompt | llm
    rewritten_question = invoke_chain(rewriter_chain, {"question": question})
    print(f"Rewritten question: {getattr(rewritten_question, 'content', str(rewritten_question))}")
    return {"question": getattr(rewritten_question, "content", str(rewritten_question))}

def check_context_size(state):
    """
    Check if the combined context of retrieved documents is too large for the LLM.
    """
    print("---NODE: CHECK CONTEXT SIZE---")
    documents = state["documents"]
    # Simple token count estimation (adjust limit as needed for your model)
    # A more robust method would use the model's tokenizer.
    CONTEXT_LIMIT = 3000 
    total_tokens = sum(len(doc.page_content.split()) for doc in documents)
    
    if total_tokens > CONTEXT_LIMIT:
        print(f"---CONTEXT OVERFLOW: {total_tokens} tokens > {CONTEXT_LIMIT}---")
        return {"context_too_large": True}
    else:
        print(f"---CONTEXT OK: {total_tokens} tokens <= {CONTEXT_LIMIT}---")
        return {"context_too_large": False}

def summarize_map(state):
    """
    Summarize each document individually if the context is too large.
    This is the "Map" part of Map-Reduce.
    """
    print("---NODE: SUMMARIZE MAP---")
    question = state["question"]
    documents = state["documents"]

    summarizer_prompt = PromptTemplate(
        template="Summarize the key points in the following text that are relevant to the user's query: '{question}'\n\nText: {document}",
        input_variables=["question", "document"],
    )
    summarizer_chain = summarizer_prompt | llm

    # Summarize each document in parallel (or sequentially)
    summaries = [getattr(invoke_chain(summarizer_chain, {"question": question, "document": doc.page_content}), "content", str(invoke_chain(summarizer_chain, {"question": question, "document": doc.page_content}))) for doc in documents]
    
    # Create new Document objects from the summaries
    summary_docs = [
        Document(page_content=summary, metadata=(getattr(doc, "metadata", {}) or {}))
        for summary, doc in zip(summaries, documents)
    ]
    return {"documents": summary_docs}

# 3. Define the Edges (The control flow)

def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate the question.
    """
    print("---EDGE: DECIDE TO GENERATE---")
    documents = state["documents"]
    iterations = state.get("iterations", 0)
    
    if not documents or iterations >= 3: # If no relevant docs or max retries reached
        if iterations >= 3:
            print("---DECISION: MAX RETRIES REACHED, ENDING---")
        else:
            print("---DECISION: NO RELEVANT DOCUMENTS, REWRITING QUERY---")
        return "rewrite"
    else:
        print("---DECISION: RELEVANT DOCUMENTS FOUND, PROCEEDING TO GENERATE---")
        return "generate"

def decide_context_path(state):
    """
    Decides whether to use the full context or apply Map-Reduce.
    """
    print("---EDGE: DECIDE CONTEXT PATH---")
    if state["context_too_large"]:
        return "summarize"
    else:
        return "direct_generate"

# --- Build the Graph ---

# Initialize LLM and Retriever globally for the graph nodes
llm = load_chat_model()
retriever = get_retriever()
CUSTOM_PROMPT = """Use ONLY the information in the context to answer the user's question. If the answer is not in the context, say you don't know. Do not make anything up.

Context: {context}
Question: {question}
Answer:"""

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("check_context_size", check_context_size)
workflow.add_node("summarize_map", summarize_map)
workflow.add_node("generate", generate)

# Set entry point
workflow.set_entry_point("retrieve")

# Add edges
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "rewrite": "rewrite",
        "generate": "check_context_size",
    },
)
workflow.add_edge("rewrite", "retrieve") # This creates the self-correction loop
workflow.add_conditional_edges(
    "check_context_size",
    decide_context_path,
    {
        "summarize": "summarize_map",
        "direct_generate": "generate",
    },
)
workflow.add_edge("summarize_map", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()

# --- Run the Chatbot ---
if __name__ == "__main__":
    query = input("Enter your query: ").strip()
    if not query:
        raise SystemExit("Empty query. Exiting.")

    # Invoke the graph once (do not stream and then invoke again - that causes two runs)
    inputs = {"question": query, "iterations": 0}
    final_state = app.invoke(inputs)

    # Print only the final answer and the LLM call count
    print("\n\n--- FINAL ANSWER ---")
    print(final_state.get("generation"))
    print(f"\nLLM calls made: {llm_call_count}")

