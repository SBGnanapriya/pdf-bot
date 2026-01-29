import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="PDF QA Bot", layout="centered")
st.title("📄 PDF Question Answering Bot")
st.write("Upload a PDF and ask any question. The bot will read the PDF and answer intelligently.")

# -----------------------------
# UPLOAD PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:

    # Load PDF
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()
    st.success(f"PDF loaded successfully! Total pages: {len(documents)}")

    # Split PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"PDF split into {len(chunks)} chunks for semantic search.")

    # -----------------------------
    # EMBEDDINGS + VECTORSTORE
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success("Embeddings created and vector store is ready!")

    # -----------------------------
    # LOAD OPEN-SOURCE LLM
    # -----------------------------
    @st.cache_resource
    def load_llm():
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0  # Use -1 for CPU
        )

    llm = load_llm()

    # -----------------------------
    # USER QUESTION
    # -----------------------------
    question = st.text_input("Ask your question:")

    if st.button("Get Answer") and question.strip() != "":
        # Semantic search
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=3)
        THRESHOLD = 1.0  # similarity threshold
        relevant_docs = [doc for doc, score in docs_with_scores if score < THRESHOLD]

        if not relevant_docs:
            st.error("❌ Answer not found in the PDF.")
        else:
            # Combine chunks
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # Generate answer using LLM
            prompt = f"""
Answer the following question based ONLY on the context below.
If the answer is not in the context, say "Answer not found".

Context:
{context}

Question:
{question}
"""

            answer = llm(prompt, max_length=500)[0]["generated_text"]
            st.success(answer)
