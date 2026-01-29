import streamlit as st
import tempfile
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="PDF QA Bot", layout="wide")
st.title("📄 PDF Question Answering Bot (Open-Source LLM)")


# ----------------------------
# Load models safely
# ----------------------------
@st.cache_resource
def load_models():
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    return embedder, tokenizer, model


embedder, tokenizer, model = load_models()


# ----------------------------
# File upload
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    st.success(f"PDF loaded successfully! Pages: {len(documents)}")

    # ----------------------------
    # Split PDF
    # ----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embedder)
    st.success("Embeddings created!")


    # ----------------------------
    # Question input
    # ----------------------------
    query = st.text_input("Ask a question from the PDF")

    if query:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)

        THRESHOLD = 0.6
        query_words = set(query.lower().split())
        relevant_docs = []

        for doc, score in docs_with_scores:
            content = doc.page_content.lower()
            overlap = query_words.intersection(content.split())
            if score < THRESHOLD and len(overlap) >= 1:
                relevant_docs.append(doc)

        if not relevant_docs:
            st.error("❌ Answer not found in the PDF.")
        else:
            context = "\n".join(doc.page_content for doc in relevant_docs)

            if len(context) < 200:
                st.warning("⚠️ Context too weak to answer reliably.")
            else:
                prompt = f"""
Answer the question using ONLY the context below.
Explain clearly in 8–12 lines.
If the answer is not present, say "Answer not found".

Context:
{context}

Question:
{query}

Answer:
"""

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300
                    )

                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

                st.subheader("✅ Answer")
                st.write(answer)
