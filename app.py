import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline


# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="PDF Question Answering Bot", layout="wide")
st.title("📄 PDF Question Answering Bot (NLP + LLM)")


# ----------------------------
# Load models (cached)
# ----------------------------
@st.cache_resource
def load_models():
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base"
    )

    return embedder, llm


embedder, llm = load_models()


# ----------------------------
# File upload
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    st.success(f"PDF loaded successfully! Pages: {len(documents)}")

    # ----------------------------
    # Split text
    # ----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # ----------------------------
    # Create vector store
    # ----------------------------
    vectorstore = FAISS.from_documents(chunks, embedder)

    st.success("Embeddings created successfully!")

    # ----------------------------
    # Ask question
    # ----------------------------
    query = st.text_input("Ask a question from the PDF:")

    if query:
        # Retrieve similar chunks
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)

        THRESHOLD = 0.6
        query_keywords = set(query.lower().split())
        relevant_docs = []

        for doc, score in docs_with_scores:
            content = doc.page_content.lower()
            keyword_overlap = query_keywords.intersection(content.split())

            if score < THRESHOLD and len(keyword_overlap) >= 1:
                relevant_docs.append(doc)

        # ----------------------------
        # Answer logic
        # ----------------------------
        if not relevant_docs:
            st.error("❌ Answer not found in the PDF.")
        else:
            context = "\n".join(doc.page_content for doc in relevant_docs)

            # Safety check to prevent hallucination
            if len(context.strip()) < 200:
                st.warning("⚠️ Context is too weak to generate a reliable answer.")
            else:
                prompt = f"""
You are a helpful academic assistant.

Answer the question using ONLY the context below.
Explain clearly in at least 8–12 lines.
If the answer is not present in the context, say "Answer not found".

Context:
{context}

Question:
{query}

Answer:
"""

                response = llm(
                    prompt,
                    max_new_tokens=300,
                    do_sample=False
                )

                st.subheader("✅ Answer")
                st.write(response[0]["generated_text"])
