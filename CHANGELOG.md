# Changelog

## 0.1.0 – Initial commit (2025-06-22)

### Added
- Dockerized FastAPI backend with Celery, Redis, ChromaDB.
- Core RAG services: embedding, generation, chunking, PDF processing.
- API routes: `/documents`, `/chat`, `/health` with versioned prefix `/api/v1`.
- Taskmaster integration and initial task breakdown.
- Updated `requirements.txt` with pinned, compatible dependencies.
- Robust `Dockerfile` installing necessary system libs (`libgl1-mesa-glx`, `tesseract-ocr`, etc.).
- `docker-compose.yml` orchestrating fastapi, chromadb, redis, celery worker, flower.

### Fixed
- Resolved OpenCV `libGL.so.1` dependency issues inside container.
- Removed deprecated Chroma auth providers; switched to unauthenticated client.
- Locked `pdfminer.six==20221105` to restore `PSSyntaxError` symbol.

### Upcoming
- Web frontend (task #19) with PDF upload, vector preview, chat UI, theming, Docker deploy. 