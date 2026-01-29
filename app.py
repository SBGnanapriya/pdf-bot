import streamlit as st
from pypdf import PdfReader

from langchain_community.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline

# ---------------- UI ----------------
st.set_page_config(page_title="PDF QA Bot")
st.title("📄 PDF Question Answering Bot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# ---------------- PDF Loader ----------------
def read_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text


@st.cache_resource
def build_chain(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)

    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    return qa


# ---------------- MAIN ----------------
if uploaded_file:
    with st.spinner("Reading PDF..."):
        pdf_text = read_pdf(uploaded_file)

    if not pdf_text.strip():
        st.error("❌ No readable text found in PDF")
    else:
        qa_chain = build_chain(pdf_text)
        st.success("✅ PDF loaded successfully")

        question = st.text_input("Ask any question from the PDF")

        if question:
            with st.spinner("Searching document..."):
                result = qa_chain(question)

            answer = result["result"].strip()

            if len(answer) < 20:
                st.warning("❌ Answer not found in the document")
            else:
                st.subheader("Answer (max 10 lines)")
                st.write("\n".join(answer.split("\n")[:10]))
