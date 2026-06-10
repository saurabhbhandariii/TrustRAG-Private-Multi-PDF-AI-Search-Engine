# TrustRAG: Private Multi-PDF AI Search Engine
A powerful, privacy-aware Retrieval-Augmented Generation (RAG) application built with **Streamlit**, featuring hybrid retrieval, robust security guardrails, artifact generation, and automated response evaluation.

## Key Features

- **Hybrid Retrieval Architecture**: Combines Dense Vector Search (FAISS) and Sparse Keyword Search (BM25) via Reciprocal Rank Fusion (RRF) for highly accurate document retrieval.
- **Multi-Layer Security (LLM Guard)**: Complete input and output sanitization to prevent prompt injections, toxicity, and unauthorized data leakage.
- **Automatic PII Redaction**: Automatically masks sensitive information (Emails, Phones, Credit Cards, API Keys, SSNs) from both uploaded documents and generated responses.
- **Dynamic Artifact Generation**: Automatically parses data to generate rich artifacts including Excel spreadsheets, Word documents, PDF reports, and visual charts (Pie, Bar, Line, Histogram).
- **Automated Response Evaluation**: Integrates **DeepEval** to automatically score generated answers on Relevancy, Faithfulness, and Hallucination, with a deterministic fallback mode for offline/local environments.
- **High-Performance LLMs**: Powered by the Groq API utilizing state-of-the-art models (`llama-3.3-70b-versatile`, `llama3-70b-8192`, `mixtral-8b-32768`).
- **Compliance & Audit Trails**: Maintains hashed violation logging in `violations.jsonl` for enterprise compliance and auditing.

## Project Structure

```
multi_pdf_rag-main/
├── app.py                           # Main Streamlit application and UI
├── guard_utils.py                   # Privacy & Security Guard integrations
├── privacy_config.py                # Configuration for privacy levels and PII patterns
├── guards/                          # Policies and additional security guard layers
├── utils/                           # Output detection and privacy-aware artifact formatters
├── requirements.txt                 # Project dependencies
├── PRIVACY.md                       # Comprehensive privacy architecture documentation
├── .env                             # Environment configuration
└── violations.jsonl                 # Auto-generated security violation audit trail
```

## 🚀 Quick Setup

### 1. Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com/keys)

### 2. Clone and Install

Clone the repository and set up a virtual environment:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory and configure your keys:

```bash
GROQ_API_KEY=your_groq_api_key_here
DEEPEVAL_THRESHOLD=0.5
```

### 4. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will launch automatically at **http://localhost:8501**.

## 📊 Artifacts & Exports

The system intelligently detects when a user is asking for structured data or visualization and can automatically generate and allow downloads for:

- **Data Tables** (displayed dynamically in the UI)
- **Charts** (Pie, Bar, Line, Histograms)
- **Excel Documents** (`.xlsx`)
- **Word Documents** (`.docx`)
- **PDF Reports** (`.pdf`)

## DeepEval Metrics

Every response can be automatically evaluated if `Evaluate answers` is checked. The RAG system calculates:

- **Answer Relevancy**: Does the answer directly address the question?
- **Faithfulness**: Is the answer entirely supported by the retrieved document context?
- **Hallucination Penalty**: Flags information that isn't present in the source documents.

## Support & Troubleshooting

- **No module named 'streamlit'**: Ensure your virtual environment is activated before running the app.
- **Prompt Security Blocked**: If your input is rejected, ensure you are not requesting PII, toxic content, or attempting a prompt injection.
- **No relevant content found**: Ensure your uploaded PDF contains actual selectable text, not just scanned images.

---

**Maintained by**: Security Team  
**Version**: 2.0 (Privacy-First)
