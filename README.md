---

# LangChain RAG Chatbot with Streamlit and FastAPI

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system using LangChain, FastAPI, and Streamlit to provide a chatbot interface capable of handling queries with context-aware responses. It supports uploading, managing, and querying documents. The application leverages stateful conversations, embeddings, and a vector store for efficient information retrieval.

---

## Key Features

- **User-Friendly Interface**: Built using Streamlit for an intuitive and interactive frontend.
- **Real-Time Document Management**:
  - Upload, view, and delete documents in real-time.
  - Indexed documents are stored for retrieval during chat interactions.
- **Context-Aware Responses**:
  - Stateful conversations using session management.
  - Powered by embeddings and similarity search for accurate results.
- **Seamless API Integration**: A robust FastAPI backend handles all document and query operations.
- **Modular Design**: 
  - Decoupled frontend (Streamlit) and backend (FastAPI) for scalability.
  - Integration with LangChain for vectorization and semantic search.

--- 
 
## Key Components and Techniques

### Embeddings
- **Model Used**: `text-embedding-ada-002` from OpenAI.
- **Description**: Transforms text into high-dimensional vectors, enabling semantic similarity comparisons.

### Chunking
- **Method**: Rule-based chunking.
- **Description**: Divides documents into smaller segments based on paragraph or sentence boundaries.

### Vector Store
- **Database**: ChromaDB.
- **Description**: Stores embeddings efficiently and supports similarity searches.

### Retriever
- **Type**: Vector-based retriever.
- **Description**: Fetches contextually relevant chunks using cosine similarity.

### RAG Process
- Combines **retrieval** (semantic search) and **generation** (LLM-based response) for accurate, enriched answers.

### Document Handling
- Supports PDF, DOCX, and HTML uploads.
- Converts documents to text for embedding generation.

---

## Project Structure

### Frontend: Streamlit Interface
- Select a language model.
- Upload documents for indexing.
- Manage documents (list and delete).
- Interact with the chatbot.

### Backend: FastAPI
- **Endpoints**:
  - **Chat**: Processes user queries.
  - **Upload Documents**: Handles document uploads.
  - **List Documents**: Retrieves a list of documents.
  - **Delete Document**: Removes documents.

---

## Integration and Data Flow

1. **User Interaction**: Actions are initiated through the Streamlit interface.
2. **Streamlit to FastAPI Communication**: Triggers API calls for document management or query handling.
3. **Backend Processing**:
   - Embeds user queries.
   - Retrieves relevant document contexts.
   - Generates responses using the selected model.
4. **Response Handling**: Displays responses or notifications on the frontend.
5. **State Management**: Maintains session-specific information and chat history.

---

## Installation and Setup

### Prerequisites
- **Python** 3.10+
- **Pip** for package management
- Virtual environment (recommended)

### Installation
1. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the FastAPI backend:
   ```bash
   uvicorn app.main:app --reload
   ```

3. Run the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```

### Access
- **FastAPI Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Streamlit Interface**: [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

## API Endpoints

### 1. Chat (POST `/chat`)
- **Description**: Processes user queries.
- **Request**:
  ```json
  { "query": "What is LangChain?" }
  ```
- **Response**:
  ```json
  { "response": "LangChain is a framework for building applications..." }
  ```

### 2. Upload Document (POST `/upload-doc`)
- **Description**: Uploads and indexes a document.
- **Response**:
  ```json
  { "message": "Document uploaded successfully." }
  ```

### 3. List Documents (GET `/list-docs`)
- **Response**:
  ```json
  {
    "documents": [
      { "id": 1, "name": "example.pdf", "uploaded": "2024-12-01T10:00:00" }
    ]
  }
  ```

### 4. Delete Document (POST `/delete-doc`)
- **Request**:
  ```json
  { "id": 1 }
  ```
- **Response**:
  ```json
  { "message": "Document deleted successfully." }
  ```

---

## Tokens and Pricing

- **Input Tokens**: System prompt, context, user query, and conversation history.
- **Output Tokens**: Tokens generated in responses.
- **Token Pricing**:
  | Model             | Input Cost | Output Cost | Max Tokens |
  |--------------------|------------|-------------|------------|
  | GPT-4 (8k)        | $0.03/1k   | $0.06/1k    | 8,192      |
  | GPT-4 (32k)       | $0.06/1k   | $0.12/1k    | 32,768     |
  | GPT-3.5-turbo     | $0.0015/1k | $0.002/1k   | 4,096      |
  | GPT-3.5-turbo-16k | $0.003/1k  | $0.004/1k   | 16,384     |

---

## Conclusion

This project demonstrates the integration of a Streamlit frontend with a FastAPI backend to create a robust RAG chatbot. It provides efficient document management, accurate query handling, and stateful conversation capabilities, making it an ideal solution for context-aware information retrieval.

--- 