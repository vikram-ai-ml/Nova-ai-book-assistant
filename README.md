# NOVA – AI Book Assistant

NOVA is an AI-powered book assistant built using Retrieval-Augmented Generation (RAG). It allows users to upload PDF books and ask questions in natural language, generating context-aware answers based on the uploaded documents.

## Features

- Upload and process multiple PDF books.
- Extract and split documents into text chunks.
- Generate embeddings using Hugging Face.
- Store and retrieve document embeddings using ChromaDB.
- Use MMR-based retrieval to find relevant information.
- Generate answers using Groq LLM.
- Display source references and page numbers.
- Interactive user interface built with Streamlit.

## Tech Stack

- **Language:** Python
- **Framework:** Streamlit
- **LLM Framework:** LangChain
- **Embeddings:** Hugging Face
- **Vector Database:** ChromaDB
- **LLM Integration:** Groq API
- **RAG Technique:** Maximum Marginal Relevance (MMR)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vikram-ai-ml/nova-ai-book-assistant.git
   ```

2. Navigate to the project directory:

   ```bash
   cd nova-ai-book-assistant
   ```

3. Create a virtual environment:

   ```bash
   uv venv
   ```

4. Activate the environment:

   ```bash
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

6. Create a `.env` file and add your API keys:

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

7. Run the application:

   ```bash
   streamlit run app.py
   ```

## How It Works

1. Upload PDF books.
2. Extract and split document content into chunks.
3. Generate embeddings and store them in ChromaDB.
4. Retrieve relevant chunks based on the user's question.
5. Pass the retrieved context to the Groq LLM.
6. Display the generated answer with source references.

## Author

**Vikram Kumar**

GitHub: https://github.com/vikram-ai-ml
