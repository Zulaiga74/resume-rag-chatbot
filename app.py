import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tempfile
import os

# CONFIG

load_dotenv()

st.set_page_config(
    page_title="Resume RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

# SIDEBAR

with st.sidebar:

    st.header("📌 About")

    st.write("""
    **Resume RAG Chatbot**

    Technologies Used:
    - Streamlit
    - LangChain
    - Gemini
    - FAISS
    - RAG
    """)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# GEMINI MODEL

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# MAIN UI

st.title("📄 Resume RAG Chatbot")

st.markdown(
    "Upload a resume and ask questions about it."
)

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

# PDF PROCESSING

if uploaded_file:

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

# Create FAISS only once

    if "vector_store" not in st.session_state:

        with st.spinner("Creating Vector Database..."):

            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )

        st.session_state.vector_store = vector_store

    vector_store = st.session_state.vector_store

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    st.success("✅ FAISS Vector Store Ready")

# DISPLAY CHAT HISTORY

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

# CHAT INPUT

    question = st.chat_input(
        "Ask a question about the resume..."
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a resume assistant.

        Answer only from the provided context.

        If the answer is not available in the context,
        respond with:

        "I could not find that information in the resume."

        Context:
        {context}

        Question:
        {question}
        """
    )

    if question:

# Show User Message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

# Generate Answer

        with st.spinner("Analyzing Resume..."):

            results = retriever.invoke(question)

            context = "\n\n".join(
                [doc.page_content for doc in results]
            )

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            response = llm.invoke(final_prompt)

            answer = response.content

# Save Assistant Response

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.write(answer)

# SHOW RETRIEVED CHUNKS

        with st.expander("🔍 View Retrieved Chunks"):

            for i, doc in enumerate(results):

                st.write(f"### Chunk {i+1}")
                st.write(doc.page_content)
                st.write("---")

