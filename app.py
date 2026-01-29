import streamlit as st
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("📄 PDF Question Answering Bot (Open Source LLM)")

# ---------------------------------
# LOAD MODELS (CACHED)
# ---------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    llm = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    return embedder, llm

embedder, llm = load_models()

# ---------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# ---------------------------------
# TEXT CHUNKING
# ---------------------------------
def chunk_text(text, chunk_size=400):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# ---------------------------------
# SEMANTIC SEARCH
# ---------------------------------
def get_best_chunk(question, chunks):
    q_emb = embedder.encode(question, convert_to_tensor=True)
    c_embs = embedder.encode(chunks, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, c_embs)[0]
    best_idx = torch.argmax(scores).item()

    return chunks[best_idx], scores[best_idx].item()

# ---------------------------------
# LLM ANSWER
# ---------------------------------
def generate_answer(context, question):
    prompt = f"""
You are a knowledgeable teacher.
Answer in at least 10 lines using simple explanations.

Context:
{context}

Question:
{question}

Answer:
"""
    return llm(prompt)[0]["generated_text"]

# ---------------------------------
# UI
# ---------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question from the PDF")

    if question:
        best_chunk, score = get_best_chunk(question, chunks)

        if score < 0.25:
            st.warning("Exact answer not found. Giving a general explanation.")
            context = text[:1500]
        else:
            context = best_chunk

        answer = generate_answer(context, question)

        st.subheader("📘 Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Context"):
            st.write(context)
