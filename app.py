# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄", layout="wide")
st.title("📄 PDF Q&A Bot with Hugging Face LLM")

# -----------------------------
# Step 1: Upload PDF
# -----------------------------
import tempfile

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name

        # Now load PDF from temporary path
        loader = PyPDFLoader(tmp_pdf_path)
        documents = loader.load()
        st.success(f"✅ PDF loaded successfully! Total pages: {len(documents)}")
    except Exception as e:
        st.error(f"❌ Failed to load PDF: {e}")
        st.stop()
else:
    st.info("Please upload a PDF to continue.")
    st.stop()

# -----------------------------
# Step 2: Split text into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
st.write(f"Total chunks created: {len(chunks)}")

# -----------------------------
# Step 3: Create embeddings + FAISS vectorstore
# -----------------------------
with st.spinner("Creating embeddings... This may take a moment"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
st.success("✅ Embeddings created and stored!")

# -----------------------------
# Step 4: Load Hugging Face LLM
# -----------------------------
with st.spinner("Loading LLM..."):
    try:
        llm = pipeline(
            "text2text-generation",   # Use seq2seq model
            model="google/flan-t5-base",
            device=-1  # CPU, change to 0 if using GPU
        )
    except Exception as e:
        st.error(f"❌ Failed to load LLM: {e}")
        st.stop()

# -----------------------------
# Step 5: Ask question
# -----------------------------
query = st.text_input("Ask any question from the PDF:")
if query:
    # Search for relevant chunks
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)

    # Filter relevant docs
    THRESHOLD = 0.8
    relevant_docs = [doc for doc, score in docs_with_scores if score < THRESHOLD]

    if not relevant_docs:
        st.warning("❌ Answer not found in the PDF.")
    else:
        # Combine chunks
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Generate answer using LLM
        prompt = f"""
Answer the question below using ONLY the context from the PDF.
If the answer is not present, respond with "Answer not found".

Context:
{context}

Question:
{query}
"""
        with st.spinner("Generating answer..."):
            response = llm(prompt, max_length=512, do_sample=True)
            answer = response[0]["generated_text"]

        st.markdown("### ✅ Answer:")
        st.write(answer)
