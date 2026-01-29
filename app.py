import streamlit as st
import tempfile
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="PDF QA Bot", layout="wide")
st.title("📄 PDF Question Answering Bot (Open-Source LLM)")


# ----------------------------
# Load models
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
# Upload PDF
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    st.success(f"PDF loaded successfully ({len(documents)} pages)")


    # ----------------------------
    # Split PDF
    # ----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embedder)
    st.success("Vector store created!")


    # ----------------------------
    # Question
    # ----------------------------
    query = st.text_input("Ask a question (e.g. overview, explain inheritance, what is object)")

    if query:
        # Always retrieve top chunks
        docs = vectorstore.similarity_search(query, k=5)

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are a helpful assistant answering questions ONLY from the given context.

Rules:
- If the question asks for an overview or summary, summarize the document.
- If the answer is not present in the context, say exactly: "Answer not found in the document."
- Explain clearly in 10–12 lines.
- Do NOT add outside knowledge.

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
                max_new_tokens=350,
                temperature=0.3
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("✅ Answer")
        st.write(answer)
