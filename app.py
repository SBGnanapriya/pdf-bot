# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import tempfile

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄")

st.title("📄 PDF Q&A Bot")
st.write("Upload a PDF and ask any question. The bot will answer in detail or say 'Answer not found' if it's not in the PDF.")

# -------------------------------
# Step 1: Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        st.success(f"✅ PDF loaded successfully! Total pages: {len(documents)}")
    except Exception as e:
        st.error(f"❌ Failed to load PDF: {e}")
        st.stop()
else:
    st.info("Please upload a PDF to continue.")
    st.stop()

# -------------------------------
# Step 2: Split PDF into chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
st.info(f"Total chunks created: {len(chunks)}")

# -------------------------------
# Step 3: Create embeddings & FAISS index
# -------------------------------
with st.spinner("Creating embeddings..."):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
st.success("✅ Embeddings created and stored in FAISS!")

# -------------------------------
# Step 4: Load LLM
# -------------------------------
with st.spinner("Loading language model..."):
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=-1
    )
st.success("✅ LLM loaded successfully!")

# -------------------------------
# Step 5: Ask question
# -------------------------------
query = st.text_input("Ask a question from the PDF:")

if query:
    # -------------------------------
    # Step 6: Semantic search
    # -------------------------------
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    THRESHOLD = 0.9  # High threshold: only pick very relevant chunks
    relevant_docs = [doc for doc, score in docs_with_scores if score < THRESHOLD]

    if not relevant_docs:
        st.warning("❌ Answer not found in the PDF.")
    else:
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # -------------------------------
        # Step 7: Generate answer
        # -------------------------------
        prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present in the context, say 'Answer not found'.

Context:
{context}

Question:
{query}

Answer in detail, at least 10 lines:
"""
        with st.spinner("Generating answer..."):
            output = llm(prompt, max_length=1000, do_sample=True)
            answer = output[0]["generated_text"]

        st.subheader("Answer:")
        st.write(answer)
