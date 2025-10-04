import os
import streamlit as st
from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
    ChatHuggingFace,
)
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Configuration
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
DB_FAISS_PATH = "vector_store/db_faiss"

CUSTOM_PROMPT = """
Use ONLY the information in the context to answer the user's question.
If the answer is not in the context, say you don't know. Do not make anything up.

Context:
{context}

Question:
{question}

Answer:
""".strip()

@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db  # FIXED: Actually return the database
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

def make_prompt(template: str) -> PromptTemplate:
    """Create a prompt template"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

@st.cache_resource
def load_chat_model():
    """Load and cache the chat model - FIXED: No parameters needed"""
    if not HF_TOKEN:
        raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in your environment.")
    
    try:
        # Create the endpoint
        endpoint = HuggingFaceEndpoint(
            repo_id=REPO_ID,
            huggingfacehub_api_token=HF_TOKEN,
            task="conversational",
            temperature=0.5,
            max_new_tokens=512,
        )
        # Wrap as chat model
        return ChatHuggingFace(llm=endpoint)
    except Exception as e:
        st.error(f"Failed to load chat model: {e}")
        return None

@st.cache_resource
def initialize_qa_chain():
    """Initialize the QA chain with vector store and chat model"""
    vector_store = get_vectorstore()
    chat_model = load_chat_model()
    
    if vector_store is None:
        st.error("Vector store could not be loaded.")
        return None
    
    if chat_model is None:
        st.error("Chat model could not be loaded.")
        return None
    
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,  # FIXED: Use the loaded model, not function call
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": make_prompt(CUSTOM_PROMPT)},
        )
        return qa_chain
    except Exception as e:
        st.error(f"Failed to initialize QA chain: {e}")
        return None

def process_query(qa_chain, prompt):
    """Process user query and return response with sources"""
    try:
        # FIXED: Use correct method to invoke the chain
        response = qa_chain.invoke({'query': prompt})
        result = response['result']
        source_docs = response['source_documents']
        
        # Format result with sources
        result_with_sources = result + "\n\n**Sources:**\n\n"
        for i, doc in enumerate(source_docs):
            result_with_sources += f"**Source {i+1}:**\n{doc.page_content[:200]}...\n\n"
        
        return result_with_sources
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return f"Sorry, I encountered an error while processing your question: {e}"

def main():
    st.title("🏛️ LawBot - Your Legal Assistant")
    st.markdown("Ask me legal questions based on the loaded documents!")
    
    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    
    # Initialize QA chain
    qa_chain = initialize_qa_chain()
    if qa_chain is None:
        st.error("❌ Failed to initialize the system. Please check your configuration.")
        st.stop()
    
    # Display chat messages
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your legal question here..."):
        # Add user message to chat
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process the query and get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_query(qa_chain, prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    # Check if HF token is available
    if not HF_TOKEN:
        st.error("❌ Please set your HUGGINGFACEHUB_API_TOKEN or HF_TOKEN environment variable.")
        st.markdown("""
        To get a token:
        1. Go to [Hugging Face](https://huggingface.co/)
        2. Create an account and go to Settings > Access Tokens
        3. Create a new token
        4. Set it as an environment variable: `export HF_TOKEN=your_token_here`
        """)
        st.stop()
    
    main()