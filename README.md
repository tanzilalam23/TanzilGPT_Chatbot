# TanzilGPT — AI-Powered CV Assistant

TanzilGPT is a Streamlit chatbot that answers questions about Mohammad Tanzil Alam’s professional profile. It uses retrieval-augmented generation (RAG): it searches profile information first, then uses the relevant passages to help generate a grounded answer.

**Live app:** [Open TanzilGPT](https://tanzilgptchatbot-xnar8e26wtwwhkuz3nhy7r.streamlit.app/)

## Features

- Ask about experience, skills, education, and projects.
- Semantic retrieval with Sentence Transformers and FAISS.
- Answers generated with Groq (`openai/gpt-oss-20b`).
- Streamlit chat interface, with English and German support.
- Guardrails to reduce irrelevant or unsupported responses.

## How it works

```text
Question → Retrieve relevant profile passages → Provide context to Groq → Answer
```

## Run locally

Requires Python and access to the private data repository used by the app.

```bash
git clone <repository-url>
cd <repository-folder>
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Configure the required values in `.streamlit/secrets.toml` locally, or in the app’s Secrets settings when using Streamlit Community Cloud. Do not commit tokens or API keys. The app needs the Groq key (named `pikaboo` in this project) and GitHub access settings for the private index: `GITHUB_DATA_REPO`, `GITHUB_DATA_BRANCH` (optional; defaults to `main`), and `GITHUB_TOKEN`.

The private index files are fetched at runtime by `data_loader.py`; they are not included in this code repository. Access to the private data repository is required for local use.

## Project files

```text
app.py              Streamlit application
 data_loader.py     Downloads the private FAISS index and chunks
prep_index.py        Builds an index from profile content (if available)
config.yaml          Non-secret application configuration
requirements.txt     Python dependencies
```

The index-building script requires the source profile content. If that content is private or not included in your checkout, you cannot rebuild the index from this repository alone.

## Privacy and access

The code may be public while the index repository remains private. Keep API keys and access tokens out of Git. Anyone using a hosted app can receive answers based on the information the app makes available, so do not put information in the knowledge base that should not be disclosed to its users.

## License

Add a license before allowing others to reuse or redistribute this project.
