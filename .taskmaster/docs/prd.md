Product Requirements Document: Technical Knowledge Base RAG System
1.0 Introduction and System Overview
1.1 Project Vision and Goals
The project vision is to engineer a highly accurate, responsive, and user-friendly knowledge base system for interacting with complex technical PDF documents. This system will function as an intelligent assistant for engineers, researchers, and technical staff, enabling them to query dense documentation using natural language and receive precise, context-aware answers.

The primary goal is to develop a production-grade Retrieval-Augmented Generation (RAG) application that leverages state-of-the-art open-source components to deliver a seamless and trustworthy question-answering experience.

Key objectives to achieve this goal include:

Implement a robust data ingestion pipeline capable of handling the structural complexity of technical PDFs.

Utilize the specified Qwen/Qwen3-Embedding-8B embedding model for generating high-quality text representations.   

Design and implement an advanced, context-aware chunking strategy tailored for technical content, moving beyond simplistic fixed-size methods.

Build a scalable backend with a clearly defined API for retrieval and generation logic.

Develop a frontend web application with a UI/UX modeled after the Google Gemini chat interface, incorporating lessons learned from its known usability issues.   

Adhere to the principle of leveraging existing, high-quality open-source projects to accelerate development and avoid reinventing foundational components.   

1.2 High-Level System Architecture
The system is designed as a set of decoupled microservices to ensure scalability, maintainability, and independent development. This architectural choice allows for the distinct scaling of components with different resource profiles; for instance, GPU-intensive embedding workers can be scaled independently of the I/O-bound ingestion service or the user-facing web application. This modularity is inspired by production-grade RAG frameworks like Cognita, which emphasize an organized, API-driven structure.   

The core components and data flow are as follows:

Ingestion Service: A dedicated API endpoint (/upload) accepts PDF files. It leverages a sophisticated document parsing library to extract not only raw text but also structural elements like tables, titles, and lists.

Processing Pipeline (Asynchronous): Upon file upload, a job is dispatched to an asynchronous task queue, such as Celery. This background worker performs the computationally expensive steps: applying the advanced chunking strategy, enriching chunks with metadata, generating embeddings with the Qwen3 model, and upserting the resulting vectors and metadata into the vector database. This ensures the ingestion API remains responsive.   

Vector Database: A specialized database service (Milvus for production, Chroma for local development) stores and indexes the document chunks for efficient high-dimensional similarity search.

RAG API Backend: This is the central logic service. It exposes a /chat endpoint that orchestrates the RAG process: it receives a user query, generates a query embedding, retrieves relevant chunks from the vector store, performs a reranking step to refine results, constructs a detailed prompt for the generation LLM, and streams the final answer back to the client.

Frontend Web Application: A React/Next.js single-page application provides the Gemini-like chat interface. It manages user sessions, renders the conversation, and communicates with the RAG API Backend via the defined endpoints.

1.3 Technology Stack Summary
The following table summarizes the selected technologies for each system component, chosen to meet the project's technical requirements and leverage mature open-source solutions.

Component

Technology

Rationale

PDF Parsing

unstructured.io

Excels at parsing complex layouts, tables, and logical units in PDFs, which is essential for structure-aware chunking.   

RAG Framework

LlamaIndex (core) with LangChain (agents)

LlamaIndex is purpose-built for RAG pipelines. Integrating with LangChain provides future extensibility for agentic workflows.   

Embedding Model

Qwen/Qwen3-Embedding-8B

Project requirement. A top-performing model on the MTEB multilingual benchmark with a large context window, flexible output dimensions, and a permissive Apache 2.0 license.   

Vector Store (Prod)

Milvus

Offers a scalable, distributed architecture, advanced metadata filtering, and multiple index types, making it suitable for production.   

Vector Store (Dev)

Chroma

Provides an in-memory, easy-to-use solution ideal for rapid local development and prototyping.   

Generation LLM

Qwen/Qwen2-7B-Instruct

Shares architecture with the embedding model family, ensuring compatibility. It is a strong instruction-following model with an Apache 2.0 license.   

API Backend

FastAPI (Python)

A high-performance Python web framework that integrates seamlessly with the machine learning ecosystem.

Async Task Queue

Celery with Redis

The industry standard for managing long-running background tasks like document processing and embedding, as seen in similar open-source RAG APIs.   

Frontend UI

React/Next.js with lobe-chat

lobe-chat is a feature-rich, open-source UI framework that supports Gemini and other models, providing a strong foundation for our required UI.   

Deployment

Docker, Kubernetes

Standard technologies for containerizing and orchestrating scalable microservice-based applications in production.

2.0 Data Ingestion and Preprocessing Pipeline
2.1 PDF Parsing and Content Extraction
The system must accurately parse technical PDFs, which are structurally complex documents containing text, tables, code blocks, figures, headers, footers, and multi-column layouts. Simple text extraction tools often fail on such documents, producing a disorganized "wall of text" that destroys semantic context.

To address this, the system will use the unstructured.io library. This tool is specifically designed for "smart chunking," as it first partitions a document into its constituent logical elements (e.g., Title, NarrativeText, Table) before any chunking occurs. This capability to identify the document's intrinsic structure is fundamental to the advanced chunking strategy outlined below and aligns with the "deep document understanding" philosophy of sophisticated RAG systems like RAGFlow.   

2.2 Advanced Chunking Strategy for Technical Documents
The specified Weixin article on chunking was inaccessible. Therefore, a superior, hybrid chunking strategy will be implemented, drawing from established best practices for technical document processing. This multi-pass approach ensures that chunks are both structurally aware and semantically coherent.   

Pass 1: Structural Segmentation (by_title): The document will first be segmented into large, logical sections based on its outline (e.g., "1. Introduction," "2.1 System Architecture"). This corresponds to the by_title strategy from the unstructured library. This initial pass guarantees that a single chunk will not contain text from two disparate high-level topics. This prevents the dilution of the chunk's vector representation and significantly improves the precision of retrieval, as the embedding for a chunk will be more focused.   

Pass 2: Element-Type Handling: Within each structural section from the first pass, the elements identified by unstructured will be processed according to their type:

Tables & Code Blocks: These elements represent self-contained, atomic units of information. They will not be split. Instead, each table or code block will be preserved as a single chunk. To provide necessary context for the embedding model, a descriptive title (e.g., "Table 3: Performance Benchmarks," "Code Block: Python example for API call") will be prepended to the content before embedding.

Narrative Text: Standard text elements like paragraphs and lists will be aggregated and passed to the final splitting stage.

Pass 3: Recursive Character Splitting: The narrative text from the second pass will be chunked using a RecursiveCharacterTextSplitter. This method is more intelligent than fixed-size splitting because it attempts to split text along a prioritized list of separators, preserving sentence and paragraph boundaries where possible.   

Separators: A standard hierarchy of separators will be used: ["\n\n", "\n", ". ", ", ", " "].   

Chunk Size: An initial size of 512 tokens will be used. This size is a well-established starting point that balances the need for chunks to be small enough for precise retrieval against the need for them to be large enough to contain sufficient context for the LLM.   

Chunk Overlap: An overlap of 50 tokens between consecutive chunks will be configured. This overlap is a crucial technique to maintain contextual continuity, ensuring that ideas or sentences that are split across two chunks can still be fully reconstructed during retrieval.   

This multi-pass, structure-aware strategy is more computationally intensive than basic methods but is essential for achieving the high-quality retrieval required for a production-grade system handling technical documents. The asynchronous processing pipeline is designed specifically to accommodate this complexity without impacting user-facing performance.

2.3 Metadata Enrichment
Each chunk generated by the pipeline will be stored in the vector database with a rich set of metadata. This metadata is critical for enabling advanced filtering, providing auditable source information in the UI, and facilitating debugging and evaluation.

The metadata for each chunk will follow this schema:

JSON

{
  "source_pdf": "document_name.pdf",
  "page_number": 12,
  "chunk_id": "unique_hash_of_content",
  "chunk_type": "text" | "table" | "code_block" | "figure_caption",
  "section_title": "2.1 System Architecture",
  "original_text": "The full, un-embedded text of the chunk...",
  "token_count": 489
}
This schema supports several key functionalities:

Source Attribution: The source_pdf and page_number fields will be displayed in the UI alongside the LLM's response, allowing users to verify the information. This is a cornerstone of building trust in RAG systems.   

Advanced Filtering: Vector databases like Milvus can perform pre-retrieval filtering on this metadata. This would allow for complex queries such as, "Find tables related to performance benchmarks in the 'Qwen2 Technical Report' PDF."   

Debugging and Evaluation: The chunk_type and token_count fields allow for analysis of the retrieval process, such as determining which types of content are most effective for answering user queries.

3.0 Core RAG Backend Infrastructure
3.1 Embedding Model Analysis: Qwen/Qwen3-Embedding-8B
The core of the retrieval system is the embedding model, Qwen/Qwen3-Embedding-8B. This is an 8-billion-parameter model from the Qwen3 family, specifically designed for text embedding and ranking tasks. Built upon the dense foundational models of the Qwen3 series, it achieves state-of-the-art performance, ranking #1 on the MTEB multilingual leaderboard. It inherits the exceptional multilingual capabilities of the Qwen3 family, supporting over 100 languages, including programming languages.   

Parameter

Value

Significance

Model Size

8B parameters

A large model capable of capturing complex semantic relationships in technical text.   

Embedding Dimension

Up to 4096 (User-defined from 32-4096)

The high dimensionality allows for a nuanced vector representation. The flexibility to define smaller dimensions enables optimization for different use cases and resource constraints.   

Max Context Length

32,768 tokens

While the model can handle very large inputs, the chunking strategy will use smaller segments for more precise retrieval.   

Architecture

Built on Qwen3 Decoder Architecture

The model is built on the dense version of the Qwen3 foundation model, which uses a decoder-style architecture, and is fine-tuned for embedding tasks.   

Multilingual

Yes, supports over 100 languages

Excellent for technical documents in multiple languages, including code-switched text and programming languages.   

License

Apache 2.0

This permissive license allows for commercial use, which is a critical requirement for any enterprise-level deployment.   

Resource (FP16)

~40GB GPU VRAM

Running the model requires significant GPU resources, such as an NVIDIA A100 or RTX A6000, necessitating quantization for local development and careful resource planning for production.   

Implementation Guidance:

The model will be loaded and used via the sentence-transformers library, which provides a simple .encode() method.   

The model is "instruction-aware," meaning its performance can be enhanced by providing task-specific instructions. When encoding user queries, a prompt should be used (e.g., model.encode(queries, prompt_name="query")). For documents, no prompt is necessary. This asymmetry is a deliberate design choice. Customizing instructions for specific tasks can improve performance by 1-5% and is highly recommended.   

3.2 Vector Store Selection: Milvus vs. Chroma
The system requires a vector database that is easy to use for local development but is also scalable and robust enough for a production environment. No single database is optimal for both scenarios, leading to a dual-database strategy.

Feature

Milvus

Chroma

Recommendation

Architecture

Distributed, separates storage & compute

Single-node, in-memory or file-based

Milvus for Production

Scalability

Billions of vectors, horizontal scaling

~1 million vectors, limited scalability

Milvus for Production

Advanced Features

14+ index types, RBAC, hybrid search, disk index

HNSW index, basic metadata filtering

Milvus for Production

Ease of Use

More complex setup (Docker Compose, K8s)

Extremely simple, Python-native API

Chroma for Development

Deployment

Production-focused (Kubernetes, Zilliz Cloud)

Prototyping-focused, ideal for local deployment

Chroma for Development


导出到 Google 表格
The analysis clearly shows that Milvus is designed for large-scale, enterprise use cases with its distributed architecture and advanced features. In contrast, Chroma is optimized for developer experience and rapid, small-scale deployment, making it perfect for local development and testing.   

Therefore, the project will be configured to use ChromaDB for local development environments and Milvus for staging and production. This practical approach allows developers to work efficiently on their local machines while ensuring the production system is built on a foundation that can scale.

3.3 Retrieval, Reranking, and Generation Strategy
A high-quality RAG system depends on more than just a good embedding model. The overall quality of the final answer is a product of a multi-stage pipeline: retrieval, reranking, and generation.

Retrieval: The initial step involves querying the vector database with the embedded user query to retrieve the top k=20 most semantically similar document chunks. This initial retrieval casts a wide net to gather all potentially relevant information.

Reranking: The 20 chunks retrieved in the first step are then passed to a second, lightweight reranker model. A reranker is specifically trained to assess the relevance of a document to a query, performing a more fine-grained analysis than the general semantic similarity provided by the embedding model. To ensure optimal compatibility and performance, the    

Qwen/Qwen3-Reranker-8B model from the same family is recommended. It re-orders the 20 chunks based on this relevance score. This two-stage process significantly improves precision by filtering out documents that are semantically related but not directly relevant to the user's question. The top    

n=5 reranked chunks will be selected for the final context.

Generation: The user's query and the top 5 reranked chunks are formatted into a comprehensive prompt. This prompt is then sent to a generation LLM to synthesize the final answer.

Selected Model: Qwen/Qwen2-7B-Instruct.   

Justification: Using a generation model from the same family as the embedding model is advantageous due to shared architectural traits and training data. Qwen2-7B-Instruct is a powerful, instruction-tuned model with a large context window and a permissive Apache 2.0 license, making it an excellent choice for this system.   

This pipeline, with its distinct retrieval and reranking stages, adds a small amount of latency but dramatically increases the quality of the context provided to the LLM. This is a crucial trade-off, as better context directly leads to more accurate answers and fewer hallucinations, a non-negotiable requirement for a production system.

4.0 RAG Framework and Implementation
4.1 Framework Selection: LlamaIndex and LangChain
The choice of framework is a strategic decision that balances immediate needs with future extensibility.

LlamaIndex is a data framework purpose-built for creating RAG applications. It excels at the core tasks of data ingestion, indexing, and retrieval, offering highly optimized and abstracted pipelines for these functions. Its focus is narrow and deep, making it the ideal tool for building the primary functionality of this project.   

LangChain is a more general-purpose framework for creating applications by "chaining" LLM calls with various tools and data sources. Its primary strengths are its flexibility, its vast ecosystem of integrations, and its powerful support for complex, agentic workflows where an LLM makes decisions about which tools to use.   

The decision is to use LlamaIndex as the primary framework for the data ingestion and retrieval pipeline. Its native abstractions like NodeParser, VectorStoreIndex, and QueryEngine are a direct and efficient fit for our requirements. However, the application will be architected such that the LlamaIndex query engine can be seamlessly wrapped as a LangChain Tool.   

This hybrid approach offers the best of both worlds. The system benefits from LlamaIndex's specialized RAG optimizations for its core task, while retaining the ability to leverage LangChain's powerful agentic capabilities for future enhancements (e.g., adding a tool for web search, or an agent that decides whether to query the PDF knowledge base or another data source) without requiring a major architectural refactor. This is a forward-looking design choice that ensures the system is both powerful today and adaptable for tomorrow.

4.2 Custom Component Integration
Both LlamaIndex and LangChain are designed to be extensible and support the integration of custom components, which is essential for meeting our specific requirements.   

Custom Embedding Model: A custom wrapper class will be implemented to integrate the Qwen/Qwen3-Embedding-8B model. This class will use the sentence-transformers library to load the model and will expose an embed method that conforms to the interface expected by the LlamaIndex and LangChain frameworks.

Custom PDF Parser/Chunker: The advanced, multi-pass chunking strategy defined in Section 2.2 will be encapsulated within a custom NodeParser class in LlamaIndex. This allows our specialized document processing logic to be plugged directly into the framework's standard indexing pipeline.

4.3 Reference Project: RAGFlow (github.com/infiniflow/ragflow)
While the system will be built using LlamaIndex and LangChain, the open-source project RAGFlow  will serve as a valuable conceptual and architectural reference.   

Key features and philosophies from RAGFlow to emulate include:

Deep Document Understanding: RAGFlow's core principle is to parse documents based on their semantic structure before chunking. This directly aligns with our chosen strategy of using    

unstructured.io to identify titles, tables, and text blocks.

Template-Based Chunking: RAGFlow introduces the concept of using different chunking templates for different types of documents. This is a powerful idea that our system can adopt in the future to apply different logic for research papers versus technical manuals, for example.   

Visualization: A key feature of RAGFlow is its UI for visualizing how a document is parsed and chunked. While not a version 1 requirement for this project, it is an extremely valuable feature for debugging and configuration that should be considered for a future release.   

Complete End-to-End System: RAGFlow is a complete, deployable system with a backend, frontend, and clear Docker Compose setup instructions, making it an excellent learning resource for our team on best practices for building and deploying a full RAG application.   

5.0 User Interface and Experience (UI/UX)
5.1 Deconstruction of the Gemini Chat UI
The user interface will be modeled on the clean, minimalist, and conversation-focused design of Google Gemini. However, the design will also learn from and rectify the documented usability flaws of the Gemini interface.   

Key components and design patterns to be implemented include:

Main Chat View: A central, scrollable pane displaying the conversation history. User prompts and model responses will be clearly differentiated using avatars, background colors, and alignment.

Chat Input Form: A fixed input area at the bottom of the screen. The text field will expand vertically to accommodate multi-line input, and messages can be sent via a "Send" button or by pressing the Enter key.   

Left Sidebar: A collapsible side panel listing all past chat sessions. Each session title will be editable by the user.

Response Regeneration: Each model-generated response will be accompanied by controls allowing the user to regenerate the answer.

Markdown and Code Rendering: The UI must properly render Markdown formatting, including lists, tables, and syntax-highlighted code blocks with a copy button.   

Streaming Responses: To enhance perceived performance and user experience, text from the model will be streamed to the UI token-by-token.   

Source Display (Enhancement over Gemini): A critical feature for enterprise RAG is verifiability. When a response is generated, the UI will display clickable links to the source chunks used (e.g., "Source: doc_name.pdf, Page 12"), allowing users to trace the information back to the original document.

5.2 Functional UI Requirements
FR-1: Chat Session Management: The user must be able to create a new chat, view a list of all past chats in the sidebar, rename any chat session, and delete chat sessions.

FR-2: Multi-turn Conversation: The UI must maintain the state and history of the current conversation and send the relevant history with each new request to the backend API.

FR-3: Searchable History (Improvement): A search bar must be included in the sidebar to allow users to filter their chat history by title or content. This directly addresses a major usability complaint from users of the actual Gemini interface, who find it nearly impossible to locate past conversations in a long list. For a technical knowledge base, this is a critical, non-negotiable feature.   

FR-4: Rich Content Display: The UI must correctly render Markdown, LaTeX for mathematical formulas, and syntax-highlighted code blocks.

FR-5: API Key Configuration: A settings modal will be provided for the user to securely input and save their API key to the browser's local storage.

FR-6: Responsive Design: The UI must be fully functional and aesthetically pleasing on both desktop and mobile screen sizes.

5.3 Recommended Open-Source UI Project: Lobe Chat (github.com/lobehub/lobe-chat)
To accelerate development, the project will leverage an existing open-source chat UI framework. After reviewing several candidates, Lobe Chat is the recommended choice.   

Justification:

Modern, Gemini-like Design: Its default aesthetic is clean, modern, and can be easily themed to match the target look and feel.

Multi-Provider Support: It already supports Gemini, OpenAI, Ollama, and other providers, which demonstrates an underlying architectural flexibility that will make it straightforward to connect to our custom RAG backend API.   

Advanced Features: Lobe Chat includes many of our required features out-of-the-box, such as Markdown rendering, session management, and even advanced capabilities like file uploads and knowledge base management that could be leveraged in a version 2 of our product.   

Active Development: It is a popular, well-maintained project with a large community.

The implementation plan is to fork the Lobe Chat repository, remove the logic for other LLM providers, and adapt its data flow to communicate exclusively with our RAG API endpoints. The primary development effort will be focused on integration and theming, rather than building core chat functionality from the ground up.

6.0 API, Deployment, and Non-Functional Requirements
6.1 API Endpoint Specifications
The backend will expose a RESTful API with clear endpoints for the frontend to consume.

Endpoint

Method

Body (Request)

Response (Success)

Description

/api/v1/documents

POST

multipart/form-data with PDF file

{"job_id": "...", "status": "processing"}

Uploads a new PDF for asynchronous ingestion and processing.

/api/v1/documents/{job_id}

GET

(None)

{"job_id": "...", "status": "complete" | "failed"}

Checks the processing status of a specific ingestion job.

/api/v1/chat

POST

{"messages": [...], "session_id": "..."}

Streaming JSON objects with text tokens

The main endpoint for sending a conversation turn and receiving a streamed response.

/api/v1/chat/history

GET

(None)

[{"session_id": "...", "title": "..."},...]

Retrieves the list of all chat sessions for the current user.

/api/v1/chat/history/{session_id}

GET

(None)

{"messages": [...]}

Retrieves the full message history for a specified session.


导出到 Google 表格
6.2 Deployment and Operations
A clear separation between the development and production environments is critical for project velocity and stability.

Containerization: All microservices (API, processing workers, UI) will be containerized using Docker to ensure consistency across environments.

Local Development: A docker-compose.yml file will be provided to orchestrate all services for a simple, one-command local setup. This environment will use ChromaDB and a quantized version of the embedding model (e.g., GGUF format ) to allow it to run on standard developer hardware without requiring expensive GPUs.   

Production Deployment: The system will be deployed on a Kubernetes cluster. The deployment manifests will be configured to use Milvus and a higher-precision version of the embedding and generation models (e.g., AWQ, FP16, or BF16), running on dedicated GPU node pools.

CI/CD: A CI/CD pipeline using a tool like GitHub Actions will be established to automate testing, the building of Docker images, and deployment to a staging environment upon commits to the main branch.

6.3 Non-Functional Requirements (NFRs)
Performance:

Time to First Token (Chat): The system should begin streaming a response in under 1.5 seconds.

Total Response Time (Chat): Dependent on the length of the generated output, but the streaming must be immediate.

PDF Ingestion Time: A target of under 30 seconds per 100 pages of a typical technical document.

Scalability: The RAG API and the embedding worker services must be designed to be horizontally scalable to handle increases in concurrent users and bulk document uploads.

Security:

All API endpoints must be protected by an authentication mechanism.

Sensitive credentials, such as database passwords and external API keys, must be managed through a secure secrets management system (e.g., HashiCorp Vault, AWS Secrets Manager) and must not be hardcoded in the source code.

All open-source dependencies will be regularly scanned for known vulnerabilities.

Licensing Compliance: The final product must include a "NOTICE" file that properly attributes all open-source components and their respective licenses. This is a requirement of the Apache 2.0 license used by key components like Qwen/Qwen3-Embedding-8B and Qwen2-7B-Instruct. The Qwen3 embedding models are released under the permissive Apache 2.0 license, ensuring they are suitable for commercial use.   

