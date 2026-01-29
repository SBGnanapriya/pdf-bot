import streamlit as st
from pypdf import PdfReader

from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="PDF Question Answer Bot", layout="centered")
st.title("📄 PDF Question Answer Bot")

st.write("Upload a PDF and ask **any question** based ONLY on its content.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ------------------ PDF Processing ------------------
def load_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


@st.cache_resource
def build_qa_chain(pdf_text):
    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(pdf_text)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # LLM
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    return qa_chain


# ------------------ Main Logic ------------------
if uploaded_file:
    with st.spinner("Reading PDF..."):
        pdf_text = load_pdf_text(uploaded_file)

    if len(pdf_text.strip()) == 0:
        st.error("❌ Could not extract text from this PDF.")
    else:
        qa_chain = build_qa_chain(pdf_text)
        st.success("✅ PDF loaded successfully!")

        question = st.text_input("Ask your question:")

        if question:
            with st.spinner("Searching document..."):
                result = qa_chain(question)

            answer = result["result"].strip()

            # Strict NOT FOUND logic
            if (
                len(answer) < 15
                or "not found" in answer.lower()
                or "no information" in answer.lower()
            ):
                st.warning("❌ Answer not found in the document.")
            else:
                st.subheader("📌 Answer (max 10 lines)")
                lines = answer.split("\n")[:10]
                st.write("\n".join(lines))
