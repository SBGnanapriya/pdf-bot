# app.py
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("📄 PDF Q&A Bot with LLM")

# 1️⃣ Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_file is not None:

    # Save PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    st.success(f"✅ PDF loaded successfully! Total pages: {len(documents)}")

    # Split PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"Total chunks created: {len(chunks)}")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success("✅ FAISS vectorstore created with embeddings!")

    # Ask question
    query = st.text_input("Ask a question about your PDF:")

    if query:
        # Search for relevant chunks
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
        THRESHOLD = 1.0  # strict similarity
        relevant_docs = [doc for doc, score in docs_with_scores if score < THRESHOLD]

        if not relevant_docs:
            st.warning("❌ Answer not found in the PDF.")
        else:
            # Combine chunks into context
            context = "\n".join([doc.page_content for doc in relevant_docs])
            st.subheader("Retrieved context from PDF:")
            st.write(context)

            # Load LLM (Hugging Face)
            llm = pipeline(
                task="text2text-generation",
                model="google/flan-t5-large",  # larger for better answers
                device=0  # use GPU if available
            )

            # Generate answer using context
            prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "Answer not found".

Context:
{context}

Question:
{query}
"""
            response = llm(prompt, max_length=512, do_sample=True, top_p=0.9, temperature=0.7)
            answer = response[0]["generated_text"]

            st.subheader("📌 Final Answer:")
            st.write(answer)

            st.success(answer)
