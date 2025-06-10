# RAG Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents, ask questions about their content, and receive answers powered by Google Gemini (via LangChain) and HuggingFace embeddings.

## Features
- Upload and process multiple PDF files
- Chunk and embed document text using HuggingFace models
- Store and search document chunks using Chroma vector store
- Answer questions about uploaded documents using Gemini LLM
- LangSmith tracing support for debugging and monitoring

## Setup
1. **Clone the repository**
2. **Create a virtual environment**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Configure environment variables**
   - Create a `.env` file in the project root with the following (replace with your real keys):
     ```ini
     GOOGLE_API_KEY=your-google-gemini-api-key
     LANGSMITH_TRACING=True
     LANGSMITH_ENDPOINT=https://api.smith.langchain.com
     LANGSMITH_API_KEY=your-langsmith-api-key
     LANGSMITH_PROJECT=your-langsmith-project
     ```

## Usage
Run the app with:
```sh
streamlit run chatbot.py
```
Open the provided local URL in your browser.

## Challenges & Solutions

### 1. **FAISS Installation Issues on macOS**
- **Challenge:** `faiss-cpu` failed to install due to missing C++ headers and lack of pre-built wheels for macOS.
- **Solution:** Switched to [Chroma](https://www.trychroma.com/) as the vector store, which is pure Python and works cross-platform.

### 2. **LangChain Deprecation Warnings**
- **Challenge:** Deprecated imports for FAISS and VertexAIEmbeddings in LangChain 0.2+.
- **Solution:** Updated imports to use `langchain_community.vectorstores` and `langchain_huggingface` as recommended by LangChain docs.

### 3. **HuggingFaceEmbeddings Deprecation**
- **Challenge:** `HuggingFaceEmbeddings` was deprecated in LangChain core.
- **Solution:** Installed and imported from `langchain-huggingface` package.

### 4. **API Key Management and Security**
- **Challenge:** API keys and secrets were hardcoded in the Python script, risking accidental exposure.
- **Solution:** Moved all secrets and configuration to a `.env` file and loaded them using `python-dotenv` and `os.getenv()`.

### 5. **Google Gemini API Key Errors**
- **Challenge:** Invalid or missing Gemini API key caused authentication errors.
- **Solution:** Ensured the correct key is set in `.env` as `GOOGLE_API_KEY` and removed runtime prompts for API keys.

### 6. **NumPy and PyTorch Compatibility**
- **Challenge:** Some packages were incompatible with NumPy 2.x.
- **Solution:** Ensured `numpy<2` is installed if needed, and used compatible versions for all dependencies.

### 7. **SSL/LibreSSL Warnings**
- **Challenge:** urllib3 v2 only supports OpenSSL 1.1.1+, but macOS Python uses LibreSSL.
- **Solution:** This is a warning and does not block functionality, but can be resolved by using a Python build linked to OpenSSL if needed.

## Notes
- For best results, use Python 3.9+ and keep all dependencies up to date.
- If you encounter issues with model downloads or network errors, check your internet connection and proxy settings.
- For more on RAG and LangChain, see the [LangChain QA with RAG guide](https://python.langchain.com/docs/how_to/qa_with_rag).

---

**Author:** Ebube Imoh
