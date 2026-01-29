import streamlit as st
from pypdf import PdfReader
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PDF QA Bot (Open Source LLM)", layout="wide")
st.title("📄 PDF Question Answering Bot (Hugging Face LLM)")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"  # open-source, good explanations
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# ---------------- KEYWORD MATCHING ----------------
def find_relevant_text(question, pdf_text):
    keywords = question.lower().split()
    matched_lines = []

    for line in pdf_text.split("\n"):
        if any(word in line.lower() for word in keywords):
            matched_lines.append(line)

    return "\n".join(matched_lines)

# ---------------- ANSWER GENERATION ----------------
def generate_answer(question, matched_text, full_text):
    if matched_text.strip():
        prompt = f"""
Answer the question using the context below.
Explain clearly in at least 10 lines.

Context:
{matched_text}

Question:
{question}
"""
    else:
        # fallback for overview / general questions
        prompt = f"""
The user asked a general or high-level question.
Use the document below to answer.

Document:
{full_text[:3000]}

Question:
{question}

Instructions:
- Explain in at least 10 lines
- Simple, student-friendly explanation
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.3,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- UI ----------------
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    st.success("PDF uploaded and processed successfully!")

    question = st.text_input("Ask a question about the PDF")

    if st.button("Get Answer") and question.strip():
        with st.spinner("Thinking... 🤔"):
            matched_text = find_relevant_text(question, pdf_text)
            answer = generate_answer(question, matched_text, pdf_text)

        st.subheader("Answer:")
        st.write(answer)
