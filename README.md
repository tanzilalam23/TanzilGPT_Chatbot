# TanzilGPT — AI-Powered CV Assistant

TanzilGPT is a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to ask questions about a professional profile, including work experience, skills, education, and projects.

Instead of generating answers only from an LLM, the system first retrieves relevant information from a structured knowledge base and uses that context to generate a grounded response.

## Features

* 🤖 AI-powered CV and portfolio assistant
* 🔎 Semantic search using FAISS and Sentence Transformers
* 📚 Retrieves information from CV and project documentation
* 🧠 Context-aware follow-up questions
* 🌍 English and German language support
* 💬 Interactive Streamlit chat interface
* 🔗 GitHub and LinkedIn integration
* 🛡️ Guardrails to reduce unsupported or irrelevant responses
* 📌 Source-grounded answers based on available profile information

## How It Works

```text
User Question
     ↓
Question Classification
     ↓
Relevant Information Retrieval
     ↓
FAISS + Semantic Search
     ↓
Context / Evidence
     ↓
Groq LLM
     ↓
Answer
```

The system combines **semantic retrieval, lexical matching, CV-specific routing, and conversation context** to improve answer relevance and reduce incorrect information.

## Technology Stack

* **Frontend:** Streamlit
* **LLM:** Groq — `openai/gpt-oss-20b`
* **Embeddings:** Sentence Transformers
* **Vector Search:** FAISS
* **Language:** Python
* **Configuration:** YAML
* **Document Processing:** Python, GitPython

## Project Structure

```text
TanzilGPT_Chatbot/
│
├── app.py              # Streamlit chatbot application
├── prep_index.py       # Builds the RAG knowledge base
├── config.yaml         # Application configuration
├── requirements.txt    # Python dependencies
│
├── content/            # CV and project information
├── index/              # Generated FAISS knowledge base
├── assets/             # Images and documentation assets
│
└── .streamlit/
    └── secrets.toml   # API credentials
```

The application will then be available at:

```text
https://tanzilgptchatbot-xnar8e26wtwwhkuz3nhy7r.streamlit.app/
```

## Knowledge Base

`prep_index.py` processes the profile content, creates embeddings, and generates a FAISS index containing the searchable knowledge used by the chatbot.

Whenever the underlying profile or project information changes, rebuild the index:

```bash
python prep_index.py
```

## Purpose

TanzilGPT is designed as a practical example of how **RAG, vector search, LLMs, and Streamlit** can be combined to build an interactive AI assistant for professional profiles.
