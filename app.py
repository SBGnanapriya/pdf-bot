# app.py

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# ------------------------------
# 1️⃣ Streamlit UI
# ------------------------------
st.title("📄 PDF Question-Answer Bot (LLM + NLP)")
st.write("Upload a PDF and ask any question. The bot will try to answer using the PDF content.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")


    # ------------------------------
    # 2️⃣ Load PDF
    # ------------------------------
import tempfile

if uploaded_file:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load PDF using file path
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    st.success(f"PDF loaded successfully! Total pages: {len(documents)}")

    # ------------------------------
    # 3️⃣ Split PDF into chunks
    # ------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"PDF split into {len(chunks)} chunks.")

    # ------------------------------
    # 4️⃣ Create embeddings and FAISS vectorstore
    # ------------------------------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success("Embeddings created and stored in FAISS!")

    # ------------------------------
    # 5️⃣ Initialize Hugging Face LLM
    # ------------------------------
    llm = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

    # ------------------------------
    # 6️⃣ Ask question
    # ------------------------------
    query = st.text_input("Ask a question from the PDF:")

    if query:
        # 6a. Retrieve top 3 similar chunks
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        THRESHOLD = 0.7

        relevant_docs = [
            doc for doc, score in docs_with_scores
            if score < THRESHOLD
        ]

        # 6b. Combine context
        if relevant_docs:
            context = "\n".join([doc.page_content for doc in relevant_docs])
        else:
            # If FAISS found nothing, use the entire PDF text
            context = "\n".join([doc.page_content for doc in chunks])

        # ------------------------------
        # 7️⃣ Prepare prompt for LLM
        # ------------------------------
        prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "Answer not found".

Context:
{context}

Question:
{query}
"""

        # ------------------------------
        # 8️⃣ Get LLM response
        # ------------------------------
        response = llm(prompt)
        answer = response[0]["generated_text"]

        st.subheader("Answer:")
        st.write(answer)

