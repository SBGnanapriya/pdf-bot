# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

st.set_page_config(page_title="PDF QA Bot", layout="wide")
st.title("📄 PDF Question Answering Bot (Hugging Face LLM)")

# -------------------------------
# 1️⃣ Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if not uploaded_file:
    st.warning("Please upload a PDF to continue.")
    st.stop()

# -------------------------------
# 2️⃣ Load PDF
# -------------------------------
try:
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()
    st.success(f"✅ PDF loaded successfully! Total pages: {len(documents)}")
except Exception as e:
    st.error(f"❌ Failed to load PDF: {e}")
    st.stop()

# -------------------------------
# 3️⃣ Split PDF into chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
st.info(f"Total chunks created: {len(chunks)}")

# -------------------------------
# 4️⃣ Create embeddings
# -------------------------------
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embedder)
st.success("✅ Embeddings created and stored in FAISS!")

# -------------------------------
# 5️⃣ Load Hugging Face LLM
# -------------------------------
try:
    llm = pipeline(
        task="text-generation",  # Correct task
        model="google/flan-t5-base",
        max_length=512
    )
    st.success("✅ LLM loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load LLM: {e}")
    st.stop()

# -------------------------------
# 6️⃣ Ask questions
# -------------------------------
query = st.text_input("Ask a question from the PDF:")

if query:
    # Retrieve similar chunks
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    THRESHOLD = 0.7
    relevant_docs = [doc for doc, score in docs_with_scores if score < THRESHOLD]

    # Combine chunks or fallback
    if not relevant_docs:
        context = "\n".join([doc.page_content for doc in chunks])  # Use full PDF if not matched
    else:
        context = "\n".join([doc.page_content for doc in relevant_docs])

    # Prepare prompt
    prompt = f"""
Answer the following question using ONLY the context below.
If the answer is not present, say "Answer not found".

Context:
{context}

Question:
{query}
"""

    # Generate answer
    response = llm(prompt, max_length=512, do_sample=True, temperature=0.7)
    answer = response[0]["generated_text"]

    st.subheader("Answer:")
    st.write(answer)
