# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import os

st.set_page_config(page_title="PDF QA Bot", layout="wide")
st.title("📄 PDF Question Answering Bot")

# -----------------------------
# 1️⃣ Upload PDF
# -----------------------------
import tempfile

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name  # <-- this is a valid file path

    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        st.success(f"✅ PDF loaded successfully! Total pages: {len(documents)}")
    except Exception as e:
        st.error(f"❌ Failed to load PDF: {e}")
        st.stop()
else:
    st.info("Please upload a PDF to continue.")
    st.stop()


# -----------------------------
# 2️⃣ Split PDF into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
st.write(f"Total chunks created: {len(chunks)}")

# -----------------------------
# 3️⃣ Load Embeddings and LLM
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    try:
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        llm = pipeline(
            task="text-generation",  # changed to compatible open-source task
            model="google/flan-t5-base",
            max_length=512
        )
        return embedder, llm
    except Exception as e:
        st.error(f"❌ Failed to load LLM: {e}")
        st.stop()

embedder, llm = load_models()
st.success("✅ Models loaded successfully!")

# -----------------------------
# 4️⃣ Create FAISS vector store
# -----------------------------
vectorstore = FAISS.from_documents(chunks, embedder)

# -----------------------------
# 5️⃣ Ask user for a question
# -----------------------------
query = st.text_input("Ask a question from the PDF:")

if query:
    # Step 1: Semantic Search
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    THRESHOLD = 0.8
    relevant_docs = [doc for doc, score in docs_with_scores if score < THRESHOLD]

    # Step 2: Prepare context for LLM
    if relevant_docs:
        context = "\n".join([doc.page_content for doc in relevant_docs])
        st.write("✅ Relevant context found via semantic search.")
    else:
        # Fallback: use the **whole PDF**
        context = "\n".join([doc.page_content for doc in chunks])
        st.write("⚠️ No specific match found. Using full PDF for answering...")

    # Step 3: Generate answer using LLM
    prompt = f"""
Answer the question below using ONLY the context provided.
If the answer is not in the context, say "Answer not found".

Context:
{context}

Question:
{query}

Provide a detailed explanation (at least 10 lines if possible).
"""

    try:
        response = llm(prompt)
        answer = response[0]["generated_text"] if isinstance(response, list) else response
        st.subheader("Answer:")
        st.write(answer)
    except Exception as e:
        st.error(f"❌ Failed to generate answer: {e}")
