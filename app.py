
import os
import shutil
import tempfile
import textwrap

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="NOVA | AI Book Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = "chroma_db"
COLLECTION_NAME = "book_collection"


# ---------------- HTML HELPER ----------------

def render_html(html):
    st.markdown(
        textwrap.dedent(html).strip(),
        unsafe_allow_html=True
    )


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b1020;
    color: #f1f5f9;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

section[data-testid="stSidebar"] {
    background: #10172a;
    border-right: 1px solid #263149;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 700;
    letter-spacing: -1px;
    color: #ffffff;
}

.brand span {
    color: #a78bfa;
}

.sidebar-caption {
    color: #8995ad;
    font-size: 12px;
    margin-top: 4px;
    margin-bottom: 30px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 43px;
    font-weight: 700;
    letter-spacing: -2px;
    color: #ffffff;
    line-height: 1.2;
}

.gradient-text {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-top: 10px;
    margin-bottom: 28px;
}

.info-card {
    background: linear-gradient(135deg, #18233d, #141c30);
    border: 1px solid #293653;
    border-radius: 16px;
    padding: 20px;
    min-height: 110px;
}

.card-label {
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 10px;
}

.card-value {
    color: white;
    font-size: 25px;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
}

.welcome-box {
    background: linear-gradient(135deg, #171f38, #151b31);
    border: 1px solid #303c5a;
    border-radius: 22px;
    padding: 35px;
    margin-top: 25px;
}

.welcome-icon {
    font-size: 35px;
    margin-bottom: 15px;
}

.welcome-title {
    font-family: 'Space Grotesk', sans-serif;
    color: white;
    font-size: 26px;
    font-weight: 700;
}

.welcome-description {
    color: #9ca9c1;
    font-size: 14px;
    line-height: 1.7;
}

.section-title {
    color: #e2e8f0;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 15px;
}

.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    border: none;
    background: linear-gradient(90deg, #6d28d9, #4f46e5);
    color: white;
}

div[data-testid="stChatMessage"] {
    background: #151e33;
    border: 1px solid #293653;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 15px;
}

div[data-testid="stChatInput"] textarea {
    background: #151e33;
    color: white;
    border: 1px solid #34415f;
    border-radius: 14px;
}

div[data-testid="stFileUploader"] {
    background: #151e33;
    border: 1px dashed #53658b;
    border-radius: 14px;
    padding: 12px;
}

hr {
    border-color: #293653;
}

.source-box {
    background: #111a2e;
    border: 1px solid #293653;
    border-radius: 10px;
    padding: 12px 15px;
    margin-top: 8px;
    color: #b7c4da;
    font-size: 13px;
}

.small-muted {
    color: #7f8da8;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- MODELS ----------------

@st.cache_resource
def load_models():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    return embeddings, llm


# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are NOVA, an intelligent AI book assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context, say:
"I couldn't find this information in the uploaded books."

Give clear, structured and helpful answers.
Do not make up information."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}"""
    )
])


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_books" not in st.session_state:
    st.session_state.processed_books = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ---------------- SIDEBAR ----------------

with st.sidebar:

    render_html("""
        <div class="brand">✦ NOVA<span>.</span></div>
        <div class="sidebar-caption">
            YOUR PERSONAL AI BOOK ASSISTANT
        </div>
    """)

    st.markdown("### 📚 Knowledge Library")
    st.caption("Upload your books to start a conversation.")

    uploaded_files = st.file_uploader(
        "Upload PDF books",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "✦ Process Books",
        use_container_width=True
    )

    if process_button:

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")

        else:

            all_documents = []
            progress = st.progress(0)
            status = st.empty()

            try:

                for index, uploaded_file in enumerate(uploaded_files):

                    status.info(
                        f"Reading {uploaded_file.name}..."
                    )

                    temp_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        ) as temp_file:

                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name

                        loader = PyPDFLoader(temp_path)
                        documents = loader.load()

                        for doc in documents:
                            doc.metadata["source"] = uploaded_file.name

                        all_documents.extend(documents)

                    finally:

                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)

                    progress.progress(
                        (index + 1) / len(uploaded_files)
                    )

                if not all_documents:

                    st.error("No readable content found in the PDFs.")

                else:

                    status.info("Splitting documents into chunks...")

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_documents(all_documents)

                    if not chunks:
                        st.error("No text chunks were generated.")

                    else:

                        status.info("Loading local embedding model...")

                        embeddings, _ = load_models()

                        status.info(
                            f"Generating embeddings for {len(chunks)} chunks..."
                        )

                        # Remove previous database
                        if os.path.exists(DB_PATH):
                            shutil.rmtree(DB_PATH)

                        # Create ChromaDB
                        vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=DB_PATH,
                            collection_name=COLLECTION_NAME
                        )

                        st.session_state.vectorstore = vectorstore

                        st.session_state.processed_books = [
                            file.name for file in uploaded_files
                        ]

                        st.session_state.total_chunks = len(chunks)

                        st.session_state.messages = []

                        progress.progress(1.0)
                        status.success("Your library is ready!")

                        st.success(
                            f"Successfully processed {len(uploaded_files)} book(s)."
                        )

            except Exception as e:
                st.error(f"Processing failed: {e}")


    st.divider()

    st.markdown("### 📖 Your Library")

    if st.session_state.processed_books:

        for book in st.session_state.processed_books:
            st.markdown(f"📕 **{book}**")

        st.caption(
            f"{len(st.session_state.processed_books)} books available"
        )

    else:

        st.markdown(
            '<p class="small-muted">No books added yet.</p>',
            unsafe_allow_html=True
        )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<p class="small-muted">Hugging Face + ChromaDB + Groq</p>',
        unsafe_allow_html=True
    )


# ---------------- MAIN UI ----------------

render_html("""
    <div class="hero-title">
        Your knowledge.<br>
        <span class="gradient-text">One conversation away.</span>
    </div>

    <div class="hero-subtitle">
        Ask questions, explore ideas and discover insights from your books.
    </div>
""")


# ---------------- STAT CARDS ----------------

col1, col2, col3 = st.columns(3)

with col1:

    render_html(f"""
        <div class="info-card">
            <div class="card-label">BOOKS UPLOADED</div>
            <div class="card-value">
                {len(st.session_state.processed_books)}
            </div>
        </div>
    """)

with col2:

    render_html(f"""
        <div class="info-card">
            <div class="card-label">KNOWLEDGE CHUNKS</div>
            <div class="card-value">
                {st.session_state.total_chunks}
            </div>
        </div>
    """)

with col3:

    status_text = (
        "Active" if st.session_state.vectorstore else "Not Ready"
    )

    render_html(f"""
        <div class="info-card">
            <div class="card-label">AI ASSISTANT STATUS</div>
            <div class="card-value">{status_text}</div>
        </div>
    """)


# ---------------- WELCOME SCREEN ----------------

if not st.session_state.vectorstore and not st.session_state.messages:

    render_html("""
        <div class="welcome-box">

            <div class="welcome-icon">✦</div>

            <div class="welcome-title">
                Welcome to your reading space.
            </div>

            <div class="welcome-description">
                Turn your PDF books into an interactive knowledge base.
                Upload your books from the sidebar, process them, and start
                asking questions. NOVA will find relevant information
                directly from your documents.

                <br><br>

                <b>Start by uploading your first PDF book.</b>
            </div>

        </div>
    """)

    render_html("""
        <div class="section-title">
            What you can ask NOVA
        </div>
    """)

    c1, c2 = st.columns(2)

    with c1:

        render_html("""
            <div class="info-card">
                <b>✦ Summarize a chapter</b>
                <p class="small-muted">
                    Get a simple explanation of important chapters.
                </p>
            </div>
        """)

    with c2:

        render_html("""
            <div class="info-card">
                <b>⌕ Find specific information</b>
                <p class="small-muted">
                    Search your books using natural language.
                </p>
            </div>
        """)


# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:

                    render_html(f"""
                        <div class="source-box">
                            📄 {source}
                        </div>
                    """)


# ---------------- CHAT INPUT ----------------

if st.session_state.vectorstore:

    query = st.chat_input(
        "Ask anything about your uploaded books..."
    )

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("NOVA is thinking..."):

                try:

                    _, llm = load_models()

                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 4,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )

                    docs = retriever.invoke(query)

                    context = "\n\n".join(
                        doc.page_content for doc in docs
                    )

                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })

                    response = llm.invoke(final_prompt)

                    answer = response.content

                    sources = []

                    for doc in docs:

                        filename = doc.metadata.get(
                            "source", "Unknown"
                        )

                        page = doc.metadata.get("page")

                        if page is not None:
                            source_text = f"{filename} — Page {page + 1}"
                        else:
                            source_text = filename

                        if source_text not in sources:
                            sources.append(source_text)

                    st.markdown(answer)

                    if sources:

                        with st.expander("📚 View Sources"):

                            for source in sources:

                                render_html(f"""
                                    <div class="source-box">
                                        📄 {source}
                                    </div>
                                """)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    st.error(f"Something went wrong: {e}")

else:

    st.info(
        "Upload and process your PDF books from the sidebar to activate NOVA."
    )
