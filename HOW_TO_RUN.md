# How to Run Document AI Assistant (Step-by-Step Guide)

This guide walks you through setting up and running the **Document AI Assistant (Local RAG)** application on Windows, macOS, and Linux.

---

## 📋 System Prerequisites

Ensure you have the following installed before starting:

1. **Python 3.11** (Recommended for PyTorch, PyMuPDF, and ChromaDB binary compatibility)
   - Verify: `py -3.11 --version` or `python --version`
2. **Ollama** (Local LLM server)
   - Download from: [https://ollama.com/download](https://ollama.com/download)
   - Verify: `ollama --version`
3. **Git**
   - Verify: `git --version`

---

## 🚀 Quick Start (Windows 1-Click Option)

If you already have Python 3.11 and Ollama installed:

1. Double-click `start.bat` (or run `.\start.ps1` in PowerShell).
2. It will automatically activate `.venv`, check dependencies, and start the web server at `http://127.0.0.1:8000`.

---

## 🛠️ Step-by-Step Manual Setup

### Step 1: Start Ollama & Download Models

1. Open a terminal and start the Ollama service (if not already running as a background service):
   ```powershell
   ollama serve
   ```

2. Download your preferred local models:
   ```powershell
   # Recommended default model (~2.0 GB)
   ollama pull llama3.2:3b

   # Ultra-fast lightweight model for slower CPUs (~522 MB)
   ollama pull qwen3:0.6b

   # Fast reasoning model (~1.1 GB)
   ollama pull deepseek-r1:1.5b
   ```

3. Confirm installed models:
   ```powershell
   ollama list
   ```

---

### Step 2: Set Up Python Virtual Environment

Navigate to the project root directory:

```powershell
cd d:\DharmenthiranProjects\DevelopmentProjects\AI\RAG\document-ai-assistant
```

Create and activate a Python 3.11 virtual environment:

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then re-run the activation script).*

**Windows (Command Prompt / CMD):**
```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Dependencies

Upgrade `pip` and install all required libraries:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

### Step 4: Configure Environment Variables

Create your `.env` configuration from `.env.example`:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

Key configuration options inside `.env`:
```ini
APP_NAME="Document AI Assistant"
HOST=127.0.0.1
PORT=8000
DEBUG=false

# Ollama LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT_SECONDS=300

# Embeddings & Vector Store Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=data/chroma_db
COLLECTION_NAME=document_rag_collection

# RAG & Chunking Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=4

# Upload Constraints
MAX_UPLOAD_MB=15
UPLOADS_DIR=data/uploads
```

---

### Step 5: Start the Web Application

With your virtual environment activated, run:

```powershell
python run.py
```

Or using Uvicorn directly:
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

### Step 6: Access the Application

Open your browser and navigate to:
- **Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check & Dependency Diagnostics**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 💡 How to Use the Interface

1. **Check Status**:
   - The status badge in the top right will indicate: `System Ready (LLM Online)`.
2. **Select LLM Model**:
   - Use the **Model** dropdown in the chat control bar to select any of your locally installed Ollama models (e.g. `llama3.2:3b`, `qwen3:0.6b`, `deepseek-r1:1.5b`).
3. **Upload PDF**:
   - Drag and drop any PDF file (up to 15MB) into the upload area on the left.
   - The app extracts text, generates recursive chunks, computes embeddings, and stores them in ChromaDB.
4. **Ask Questions**:
   - Type your question in the chat bar and hit **Enter** (or click the Send button).
   - Use **Shift + Enter** for multiline questions.
5. **Inspect Verified Citations**:
   - Click **Verified Sources** beneath the assistant answer to view exact citations, page numbers, similarity scores, and text snippets.
6. **Filter Scope**:
   - Use the **Scope** dropdown to limit answers to a specific document or search across all uploaded files.
7. **Delete Documents**:
   - Click the trash icon next to any document in the list to purge its physical file and delete its vector embeddings from ChromaDB.

---

## 🧪 Running Automated Tests

To run the complete automated test suite (24 unit and integration tests):

```powershell
pytest -v
```

All tests mock expensive models where appropriate and execute in ~2 seconds without requiring paid services.

---

## 🔬 Running Real End-to-End Pipeline Verification

To verify the entire real pipeline (real PDF creation, real PyMuPDF extraction, real SentenceTransformers embedding, ChromaDB vector indexing, and real Ollama answer generation):

```powershell
python verify_e2e.py
```

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| **`Ollama service is unreachable`** | Ollama is not running. | Run `ollama serve` in a separate terminal. |
| **`Ollama request timed out after 120s/300s`** | Large model running on CPU taking too long. | Switch to **`qwen3:0.6b`** or **`tinyllama:latest`** in the Model dropdown for instant CPU responses. |
| **`PermissionError: [WinError 32]` on cleanup** | Windows memory-mapped files locked. | Handled automatically in `pytest.ini` with `ignore_cleanup_errors=True`. |
| **`Script execution disabled on this system`** | PowerShell execution policy. | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| **Port 8000 already in use** | Another instance is running. | Change `PORT=8001` in your `.env` or terminate the existing process. |

---

## 🛑 Stopping the Server

Press `Ctrl + C` in the terminal where `python run.py` is running to cleanly shut down the application.
