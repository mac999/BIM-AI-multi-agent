# BIM AI Multi-Agent Backend

This project provides a lightweight multi-agent backend for BIM-related question answering in Korean. Queries are classified and routed to one of several agents:

- **General LLM Agent** – answers general questions with GPT-4.
- **RAG Agent** – retrieves Korean BIM documents using a FAISS vector database and answers using GPT-4.
- **Web Search Agent** – performs a web search via Serper/Tavily and summarizes results with GPT-4.
- **IFC Parser Agent** – parses uploaded IFC files with `ifcopenshell` and answers questions about the model.

FastAPI exposes simple endpoints for asking questions and uploading IFC files.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables for your OpenAI key and optional search API key(s):
   ```bash
   export OPENAI_API_KEY=your-key
   export SERPER_API_KEY=your-serper-key  # optional
   export TAVILY_API_KEY=your-tavily-key  # optional
   ```
3. Start the API server:
   ```bash
   uvicorn api.server:app --reload
   ```

## Example Requests

Ask a question:
```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question": "BIM은 무엇인가요?"}'
```

Ask about an IFC file:
```bash
curl -X POST "http://localhost:8000/ask_ifc" -F "question=모델의 엔티티 수는?" -F "file=@sample.ifc"
```
