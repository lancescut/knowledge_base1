# Technical Knowledge Base RAG System

This project is a sophisticated, AI-powered Retrieval-Augmented Generation (RAG) system designed to serve as a technical knowledge base. It can ingest PDF documents, process them, and answer complex questions based on the content of those documents.

The system is built with a modern Python stack, including FastAPI for the web framework, ChromaDB for vector storage, and Celery for background task processing. The AI capabilities are powered by Qwen models for state-of-the-art text embedding and generation.

## Features

- **FastAPI Backend**: A high-performance, asynchronous API.
- **Document Ingestion**: Upload PDF documents through a REST API endpoint.
- **Intelligent Chunking**: Documents are intelligently segmented for optimal retrieval.
- **Qwen Embeddings**: State-of-the-art text embeddings for semantic understanding.
- **ChromaDB Vector Store**: Efficient storage and retrieval of document vectors.
- **Qwen Generation**: Powerful language model for generating accurate and context-aware answers.
- **RAG Pipeline**: A complete pipeline orchestrating retrieval and generation to answer queries.
- **Dockerized**: Fully containerized for easy setup and deployment.

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Environment Variables:**
    The application can be configured via environment variables. While the current development setup uses local models that don't require API keys, you would need to provide them for a production setup or if you connect to a managed Qwen service.

    Create a `.env` file in the project root:
    ```bash
    touch .env
    ```

    Add the following variable to the `.env` file if you have an API key. If not, the application will still run with the local models.
    ```
    # .env
    QWEN_API_KEY="your-qwen-api-key-if-needed"
    ```

## Running the Application

To build and start all the services (FastAPI app, ChromaDB, Celery worker, Redis), run the following command from the project root:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`. You can view the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

## How to Use

Once the application is running, you can interact with it using any HTTP client, such as `curl` or the Swagger UI.

### 1. Upload a Document

First, you need to upload a PDF document to populate the knowledge base. The system will process the PDF, chunk its content, create embeddings, and store them in the vector database.

Use the `/api/v1/documents/upload/immediate` endpoint to upload a file. This endpoint processes the document synchronously, which is suitable for smaller files and immediate feedback.

**Example using `curl`:**

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/documents/upload/immediate' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/document.pdf;type=application/pdf'
```

Replace `/path/to/your/document.pdf` with the actual path to your PDF file.

A successful upload will return a JSON response confirming the number of chunks added to the database.

### 2. Chat with the Knowledge Base

After you've uploaded a document, you can ask questions about its content using the `/api/v1/chat` endpoint. The system will take your question, find the most relevant information from the uploaded documents, and generate an answer.

**Example using `curl`:**

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is the main topic of the document?"
}'
```

Replace `"What is the main topic of the document?"` with your question.

The API will respond with a generated answer based on the knowledge it has from the documents. The response will look something like this:

```json
{
  "response": "The main topic of the document is...",
  "retrieved_context_count": 5
}
``` 