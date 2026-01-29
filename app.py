import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("📄 PDF Question Answering Bot (Open-Source LLM)")

# ---------------------------------
# LOAD MODELS (CACHED)
# ---------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

# ---------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# ---------------------------------
# TEXT CHUNKING
# ---------------------------------
def chunk_text(text, chunk_size=350):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# ---------------------------------
# SEMANTIC SEARCH
# ---------------------------------
def find_context(question, chunks):
    q_emb = embedder.encode(question, convert_to_tensor=True)
    c_embs = embedder.encode(chunks, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, c_embs)[0]
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()

    return chunks[best_idx], best_score

# ---------------------------------
# LLM ANSWER (NO PIPELINE)
# ---------------------------------
def generate_answer(context, question):
    prompt = f"""
You are an expert teacher.
Explain clearly in at least 10 lines.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------
# UI
# ---------------------------------
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    text = extract_text(pdf)
    chunks = chunk_text(text)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question")

    if question:
        context, score = find_context(question, chunks)

        # If question is general (overview, summary, etc.)
        if score < 0.25:
            st.info("General question detected. Using broader context.")
            context = text[:1500]

        answer = generate_answer(context, question)

        st.subheader("📘 Answer")
        st.write(answer)

        with st.expander("🔍 Context used"):
            st.write(context)
