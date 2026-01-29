import streamlit as st
from pypdf import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("📄 PDF Question Answering Bot (Open Source LLM)")

# -------------------------------
# LOAD MODELS (Cached)
# -------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )
    return embedder, llm

embedder, llm = load_models()

# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# -------------------------------
# TEXT CHUNKING
# -------------------------------
def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# -------------------------------
# FIND MOST RELEVANT CHUNK
# -------------------------------
def get_best_chunk(question, chunks):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)

    similarities = util.cos_sim(question_embedding, chunk_embeddings)[0]
    best_idx = torch.argmax(similarities).item()

    return chunks[best_idx], similarities[best_idx].item()

# -------------------------------
# GENERATE ANSWER
# -------------------------------
def generate_answer(context, question):
    prompt = f"""
You are a helpful teacher.
Answer the question clearly and in detail (at least 10 lines).

Context:
{context}

Question:
{question}

Answer:
"""
    response = llm(prompt)[0]["generated_text"]
    return response

# -------------------------------
# UI
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question from the PDF")

    if question:
        best_chunk, score = get_best_chunk(question, chunks)

        if score < 0.25:
            st.warning("Answer not directly found in PDF. Giving a general explanation.")
            context = text[:1500]  # fallback: general overview
        else:
            context = best_chunk

        answer = generate_answer(context, question)

        st.subheader("📘 Answer")
        st.write(answer)

        with st.expander("🔍 Used Context"):
            st.write(context)
