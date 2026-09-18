import os
import json
import time
import html
import re

import yaml
import faiss
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mohammad Tanzil Alam — CV Assistant",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CONFIG
# ============================================================

@st.cache_data
def load_config():
    with open(
        "config.yaml",
        "r",
        encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)


try:
    CFG = load_config()
except FileNotFoundError:
    st.error("config.yaml not found.")
    st.stop()
except Exception:
    st.error("Could not load configuration.")
    st.stop()


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def load_model():
    api_key = st.secrets.get("pikaboo")

    if not api_key:
        raise RuntimeError(
            "Groq API key is missing from Streamlit secrets."
        )

    timeout = int(
        CFG.get("llm", {}).get(
            "timeout",
            30
        )
    )

    return Groq(
        api_key=api_key,
        timeout=timeout
    )


try:
    llm = load_model()
except Exception:
    st.error(
        "The AI service is currently unavailable. "
        "Please try again later."
    )
    st.stop()


# ============================================================
# RETRIEVAL COMPONENTS
# ============================================================

@st.cache_resource
def load_retrieval_components():

    index_path = "index/faiss.index"
    chunks_path = "index/chunks.jsonl"

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "index/faiss.index not found."
        )

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            "index/chunks.jsonl not found."
        )

    embedder = SentenceTransformer(
        CFG["embeddings"]["model_name"]
    )

    index = faiss.read_index(
        index_path
    )

    chunks = []

    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

                if isinstance(item, dict):
                    chunks.append(item)

            except json.JSONDecodeError:
                continue

    if not chunks:
        raise RuntimeError(
            "No valid chunks found in chunks.jsonl."
        )

    return embedder, index, chunks


try:
    embedder, index, CHUNKS = load_retrieval_components()
except Exception:
    st.error(
        "The knowledge base could not be loaded. "
        "Please make sure the index files exist."
    )
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

# Session-specific and temporary.
# No global/shared conversation history is used.

if "messages" not in st.session_state:
    st.session_state.messages = []

if "language" not in st.session_state:
    st.session_state.language = "🇬🇧 EN"


# ============================================================
# CSS
# ============================================================

CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body {
    color-scheme: only light !important;
}

*, *::before, *::after {
    font-family: 'Inter', sans-serif !important;
}


/* ============================================================
   Hide Streamlit chrome
   ============================================================ */

[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapseButton"],
header[data-testid="stHeader"],
#MainMenu,
footer {
    display: none !important;
}


/* ============================================================
   Background
   ============================================================ */

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: #1c1508 !important;
}

.main .block-container {
    padding: 2rem 2.5rem 6rem !important;
    max-width: 920px !important;
}


/* ============================================================
   Decorative orbs
   ============================================================ */

.orb-tr {
    position: fixed;
    top: -120px;
    right: -100px;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(212,175,88,0.11) 0%,
        transparent 68%
    );
    pointer-events: none;
    z-index: 0;
}

.orb-bl {
    position: fixed;
    bottom: -80px;
    left: -80px;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(180,130,40,0.07) 0%,
        transparent 68%
    );
    pointer-events: none;
    z-index: 0;
}


/* ============================================================
   Sidebar
   ============================================================ */

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: #130e03 !important;
}

[data-testid="stSidebar"] {
    border-right: 0.5px solid rgba(212,175,88,0.14) !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.4rem !important;
}


/* ============================================================
   Sidebar language
   ============================================================ */

[data-testid="stSidebar"] .stRadio > div {
    gap: 6px !important;
    flex-direction: row !important;
}

[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border: 0.5px solid rgba(212,175,88,0.25) !important;
    border-radius: 100px !important;
    padding: 5px 16px !important;
    font-size: 13px !important;
    color: #a89060 !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(212,175,88,0.14) !important;
    border-color: rgba(212,175,88,0.5) !important;
    color: #d4af58 !important;
}

[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}


/* ============================================================
   Avatar
   ============================================================ */

.avatar-circle {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    background: rgba(212,175,88,0.08);
    border: 2px solid rgba(212,175,88,0.45);
    box-shadow: 0 0 20px rgba(212,175,88,0.18);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 600;
    color: #d4af58;
    margin: 0 auto 12px auto;
}


/* ============================================================
   Sidebar text
   ============================================================ */

.profile-name {
    font-size: 16px;
    font-weight: 600;
    color: #f0e8d0;
    line-height: 1.3;
    text-align: center;
}

.profile-loc {
    font-size: 13px;
    color: #6a5c3a;
    margin-top: 4px;
    text-align: center;
}

.status-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    font-size: 13px;
    color: #7a9060;
    margin-top: 6px;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #7a9060;
    flex-shrink: 0;
    animation: softpulse 2.5s infinite;
}

@keyframes softpulse {
    0%,100% { opacity:0.35 }
    50% { opacity:1 }
}

.gold-divider {
    border: none;
    border-top: 0.5px solid rgba(212,175,88,0.1);
    margin: 16px 0;
}

.sidebar-section-label {
    font-size: 10px;
    color: #4a3f22;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
}


/* ============================================================
   Sidebar link buttons
   ============================================================ */

[data-testid="stSidebar"] .stLinkButton a {
    width: 100% !important;
    justify-content: flex-start !important;
    background: transparent !important;
    border: 0.5px solid rgba(212,175,88,0.18) !important;
    border-radius: 8px !important;
    color: #c8b070 !important;
    text-decoration: none !important;
    font-size: 14px !important;
}

[data-testid="stSidebar"] .stLinkButton a:hover {
    border-color: rgba(212,175,88,0.5) !important;
    color: #d4af58 !important;
    background: rgba(212,175,88,0.06) !important;
}


/* ============================================================
   Header
   ============================================================ */

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: #a89060;
    background: rgba(212,175,88,0.06);
    border: 0.5px solid rgba(212,175,88,0.25);
    border-radius: 100px;
    padding: 5px 14px;
    margin-bottom: 14px;
}

.badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #d4af58;
    animation: softpulse 2s infinite;
}

.main-title {
    font-size: 28px;
    font-weight: 600;
    color: #d4af58;
    letter-spacing: -0.3px;
    line-height: 1.2;
    margin: 0 0 2rem 0;
    white-space: nowrap;
}


/* ============================================================
   Quick questions
   ============================================================ */

.section-label {
    font-size: 10px;
    color: #4a3f22;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0 0 10px 0;
}

.stButton > button {
    background: rgba(212,175,88,0.05) !important;
    border: 0.5px solid rgba(212,175,88,0.2) !important;
    border-radius: 10px !important;
    color: #c8b070 !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    padding: 0.75rem 1rem !important;
    line-height: 1.45 !important;
    white-space: normal !important;
    height: auto !important;
    text-align: left !important;
    width: 100% !important;
}

.stButton > button:hover {
    border-color: rgba(212,175,88,0.5) !important;
    color: #d4af58 !important;
    box-shadow: 0 0 12px rgba(212,175,88,0.12) !important;
    background: rgba(212,175,88,0.08) !important;
}


/* ============================================================
   Chat
   ============================================================ */

.msg-wrap {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 16px;
}

.msg-wrap.user {
    flex-direction: row-reverse;
}

.msg-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    flex-shrink: 0;
    background: rgba(212,175,88,0.10);
    border: 0.5px solid rgba(212,175,88,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 500;
    color: #d4af58;
}

.msg-bubble {
    max-width: 80%;
    background: rgba(255,255,255,0.04);
    border: 0.5px solid rgba(212,175,88,0.12);
    border-radius: 14px 14px 14px 4px;
    padding: 13px 17px;
    font-size: 15px;
    color: #c8b88a;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

.msg-bubble.user {
    background: rgba(212,175,88,0.09);
    border: 0.5px solid rgba(212,175,88,0.42);
    border-radius: 14px 14px 4px 14px;
    color: #f0e0a8;
    box-shadow:
        0 0 18px rgba(212,175,88,0.14),
        inset 0 0 8px rgba(212,175,88,0.04);
}


/* ============================================================
   Markdown
   ============================================================ */

.msg-bubble p {
    margin-top: 0;
    margin-bottom: 8px;
}

.msg-bubble p:last-child {
    margin-bottom: 0;
}

.msg-bubble ul,
.msg-bubble ol {
    margin-top: 6px;
    margin-bottom: 6px;
    padding-left: 20px;
}

.msg-bubble li {
    margin-bottom: 4px;
}

.msg-bubble strong {
    color: #e0ca8c;
}


/* ============================================================
   Input
   ============================================================ */

[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    background: #1c1508 !important;
}

[data-testid="stBottom"] {
    border-top: 0.5px solid rgba(212,175,88,0.1) !important;
    padding: 10px 2.5rem !important;
}

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
div[class*="stChatInput"] {
    background: rgba(28,21,8,0.98) !important;
    border: 1px solid rgba(212,175,88,0.38) !important;
    border-radius: 12px !important;
    box-shadow:
        0 0 20px rgba(212,175,88,0.13),
        0 0 6px rgba(212,175,88,0.07) !important;
}

[data-testid="stChatInput"]:focus-within,
div[class*="stChatInput"]:focus-within {
    border-color: rgba(212,175,88,0.7) !important;
}

[data-testid="stChatInput"] textarea,
div[class*="stChatInput"] textarea {
    background: transparent !important;
    color: #f0e8d0 !important;
    font-size: 15px !important;
    caret-color: #d4af58 !important;
}

[data-testid="stChatInput"] textarea::placeholder,
div[class*="stChatInput"] textarea::placeholder {
    color: #5a4e30 !important;
}

[data-testid="stChatInputSubmitButton"] > button {
    background: rgba(212,175,88,0.2) !important;
    border: 0.5px solid rgba(212,175,88,0.45) !important;
    border-radius: 8px !important;
    color: #d4af58 !important;
}

.stSpinner > div {
    border-top-color: #d4af58 !important;
}


/* ============================================================
   Scrollbar / footer
   ============================================================ */

::-webkit-scrollbar {
    width: 5px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(212,175,88,0.2);
    border-radius: 10px;
}

.footer-bar {
    text-align: center;
    font-size: 11px;
    color: #3a3018;
    letter-spacing: 0.4px;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 0.5px solid rgba(212,175,88,0.07);
}


/* ============================================================
   Mobile topbar
   ============================================================ */

.mobile-topbar {
    display: none;
    background: #130e03;
    border-bottom: 0.5px solid rgba(212,175,88,0.14);
    padding: 14px 16px;
    margin: -2rem -1rem 1.5rem -1rem;
    align-items: center;
    justify-content: space-between;
}

.mobile-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: rgba(212,175,88,0.08);
    border: 1.5px solid rgba(212,175,88,0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 600;
    color: #d4af58;
}

.mobile-name {
    font-size: 14px;
    font-weight: 600;
    color: #f0e8d0;
}

.mobile-loc {
    font-size: 11px;
    color: #6a5c3a;
}

.mobile-links {
    display: flex;
    gap: 10px;
}

.mobile-link {
    width: 30px;
    height: 30px;
    border: 0.5px solid rgba(212,175,88,0.25);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none !important;
    color: #d4af58 !important;
    font-size: 12px;
}

.mobile-lang-toggle {
    display: flex;
    gap: 6px;
}

.mobile-lang-btn {
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 100px;
    border: 0.5px solid rgba(212,175,88,0.25);
    color: #a89060 !important;
    text-decoration: none !important;
    background: transparent;
}

.mobile-lang-btn.active {
    background: rgba(212,175,88,0.14) !important;
    border-color: rgba(212,175,88,0.5) !important;
    color: #d4af58 !important;
}


/* Mobile visibility */
@media (max-width: 768px) {

    .mobile-topbar {
        display: flex !important;
    }

    .main .block-container {
        padding: 1rem 1rem 6rem !important;
    }

    .main-title {
        font-size: 20px !important;
        white-space: normal !important;
    }

    .avatar-circle {
        width: 70px !important;
        height: 70px !important;
        font-size: 18px !important;
    }
}

</style>
"""


# ============================================================
# QUESTION TYPE
# ============================================================

def detect_question_type(
    question: str
) -> str:

    q = question.lower()

    if any(
        phrase in q
        for phrase in [
            "where did mohammad work",
            "where has mohammad worked",
            "where did he work",
            "where has he worked",
            "work history",
            "employment history",
            "career history",
            "work experience",
            "professional experience",
            "employers",
            "employer history",
            "previous employer",
            "previous employers",
            "former employer",
            "former employers",
            "most recent employer",
            "latest employer",
            "current employer",
            "who did he work for",
            "arbeitgeber",
            "arbeitgebern",
            "beruflicher werdegang",
            "berufserfahrung"
        ]
    ):
        return "employment"

    if any(
        phrase in q
        for phrase in [
            "programming language",
            "programming languages",
            "what languages does",
            "which languages does",
            "programmiersprache",
            "programmiersprachen"
        ]
    ):
        return "languages"

    if any(
        phrase in q
        for phrase in [
            "roche",
            "oncology guideline",
            "clinical guideline",
            "guideline project",
            "athena",
            "aws glue",
            "harvester",
            "s3",
            "parquet"
        ]
    ):
        return "roche"

    if any(
        phrase in q
        for phrase in [
            "technology",
            "technologies",
            "tech stack",
            "tools used",
            "tools did",
            "technologie",
            "technologien",
            "which tools",
            "what tools"
        ]
    ):
        return "technologies"

    if any(
        phrase in q
        for phrase in [
            "what was the result",
            "what were the results",
            "result of the project",
            "results of the project",
            "project result",
            "project outcome",
            "project outcomes",
            "outcome of the project",
            "what did the project achieve",
            "what did he achieve",
            "what was achieved",
            "ergebnis",
            "ergebnisse"
        ]
    ):
        return "result"

    if any(
        phrase in q
        for phrase in [
            "key projects",
            "important projects",
            "main projects",
            "what projects",
            "which projects",
            "project involved",
            "projects involved",
            "project",
            "projects",
            "projekt",
            "projekte"
        ]
    ):
        return "projects"

    if any(
        phrase in q
        for phrase in [
            "skills",
            "technical skills",
            "technical stack",
            "skillset",
            "fähigkeiten",
            "faehigkeiten"
        ]
    ):
        return "skills"

    if any(
        phrase in q
        for phrase in [
            "summarize mohammad",
            "summarise mohammad",
            "tell me about mohammad",
            "about mohammad",
            "recruiter",
            "profile",
            "professional summary",
            "short summary",
            "concise recruiter",
            "who is mohammad",
            "who is tanzil",
            "who is he",
            "who's mohammad",
            "who's tanzil",
            "who's he",
            "tell me about him",
            "tell me about tanzil",
            "about tanzil",
            "wer ist mohammad",
            "wer ist tanzil",
            "wer ist er"
        ]
    ):
        return "profile"

    return "general"


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(
    text: str
) -> str:
    return str(
        text or ""
    ).strip()


def normalize_text(
    text: str
) -> str:
    return " ".join(
        clean_text(
            text
        ).lower().split()
    )


def is_primary_cv(
    chunk: dict
) -> bool:

    source_type = str(
        chunk.get(
            "source_type",
            ""
        )
    ).lower()

    source = str(
        chunk.get(
            "source",
            ""
        )
    ).lower()

    return (
        source_type == "primary_cv"
        or "01_cv_public" in source
    )


def chunk_search_text(
    chunk: dict
) -> str:

    heading_path = chunk.get(
        "heading_path",
        []
    )

    if not isinstance(
        heading_path,
        list
    ):
        heading_path = []

    return " ".join(
        [
            str(chunk.get("source", "")),
            str(chunk.get("section", "")),
            str(chunk.get("category", "")),
            " ".join(
                str(x)
                for x in heading_path
            ),
            str(chunk.get("text", ""))
        ]
    )


# ============================================================
# GUARDRAILS
# ============================================================

def check_guardrails(
    text: str
) -> bool:

    blocked = CFG.get(
        "guardrails",
        {}
    ).get(
        "blocked_phrases",
        []
    )

    lower = text.lower()

    return not any(
        phrase.lower() in lower
        for phrase in blocked
    )


# ============================================================
# CV ROUTING
# ============================================================

def get_primary_cv_chunks(
    question_type: str,
    limit: int = 8
) -> list:
    """
    Prioritize the authoritative CV depending on the question.
    """

    primary = [
        chunk
        for chunk in CHUNKS
        if is_primary_cv(chunk)
    ]

    if not primary:
        return []

    candidates = []

    for chunk in primary:

        section = str(
            chunk.get(
                "section",
                ""
            )
        ).lower()

        category = str(
            chunk.get(
                "category",
                ""
            )
        ).lower()

        text = str(
            chunk.get(
                "text",
                ""
            )
        ).lower()

        score = 0

        if question_type == "employment":

            if category == "experience":
                score += 20

            if any(
                term in section
                for term in [
                    "work experience",
                    "associate consultant",
                    "technical mentor",
                    "data engineer",
                    "software engineer"
                ]
            ):
                score += 10

            if any(
                employer in text
                for employer in [
                    "arcondis",
                    "roche diagnostics",
                    "fortress6"
                ]
            ):
                score += 8

        elif question_type == "languages":

            if category == "skills":
                score += 25

            if "programming languages" in section:
                score += 30

            if "programming languages" in text:
                score += 25

            if any(
                term in section
                for term in [
                    "fortress6",
                    "associate consultant",
                    "technical mentor",
                    "roche"
                ]
            ):
                score -= 20

        elif question_type == "skills":

            if category == "skills":
                score += 25

        elif question_type == "projects":

            if category == "projects":
                score += 20

            if category == "experience":
                score += 6

        elif question_type == "roche":

            if "roche" in section:
                score += 20

            if "roche" in text:
                score += 15

            if any(
                term in text
                for term in [
                    "oncology",
                    "guideline",
                    "data drift"
                ]
            ):
                score += 12

        elif question_type == "result":

            if category == "experience":
                score += 8

            if any(
                term in text
                for term in [
                    "result",
                    "validated",
                    "accuracy",
                    "reduced",
                    "saved",
                    "successfully"
                ]
            ):
                score += 10

        elif question_type == "profile":

            if category in {
                "experience",
                "education",
                "skills",
                "general"
            }:
                score += 8

        else:

            score += 1

        candidates.append(
            (
                score,
                chunk
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        chunk
        for _, chunk in candidates[:limit]
    ]


# ============================================================
# TOKENIZATION / LEXICAL SCORE
# ============================================================

def tokenize(
    text: str
) -> set:

    tokens = re.findall(
        r"[a-zA-ZÀ-ÿ0-9][a-zA-ZÀ-ÿ0-9+#._/-]*",
        str(text).lower()
    )

    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by",
        "can", "did", "do", "does", "for", "from", "he",
        "her", "his", "how", "i", "in", "is", "it", "me",
        "of", "on", "or", "tell", "that", "the", "their",
        "them", "there", "these", "this", "to", "was",
        "what", "when", "where", "which", "who", "why",
        "with", "you", "about",
        "der", "die", "das", "den", "dem", "des", "ein",
        "eine", "einer", "einem", "einen", "und", "oder",
        "ist", "sind", "war", "wie", "was", "wer", "wo",
        "bei", "mit", "von", "zu", "über", "für", "fuer",
        "er", "sie", "seine", "seiner"
    }

    return {
        token
        for token in tokens
        if len(token) >= 2
        and token not in stopwords
    }


def lexical_score(
    query_terms: set,
    text: str
) -> float:

    chunk_terms = tokenize(
        text
    )

    if not query_terms or not chunk_terms:
        return 0.0

    overlap = (
        query_terms
        & chunk_terms
    )

    if not overlap:
        return 0.0

    return min(
        1.0,
        len(overlap)
        / max(
            len(query_terms),
            1
        )
    )


# ============================================================
# EMBEDDING
# ============================================================

def embed_query(
    text: str
) -> np.ndarray:

    try:

        vector = embedder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return np.asarray(
            vector,
            dtype=np.float32
        )

    except Exception:

        return np.array(
            [],
            dtype=np.float32
        )


# ============================================================
# CONVERSATION
# ============================================================

def get_prior_messages() -> list:

    messages = st.session_state.get(
        "messages",
        []
    )

    if not messages:
        return []

    return messages[:-1][-8:]


def is_follow_up_question(
    question: str
) -> bool:

    q = question.lower()

    markers = [
        "there",
        "that",
        "this",
        "it",
        "they",
        "them",
        "he",
        "his",
        "her",
        "that project",
        "this project",
        "that company",
        "this company",
        "what about",
        "why did he",
        "why was it",
        "what was the result",
        "what did he use"
    ]

    return (
        len(
            question.split()
        ) <= 12
        or any(
            marker in q
            for marker in markers
        )
    )


def get_previous_contexts() -> list:

    for message in reversed(
        get_prior_messages()
    ):

        if message.get(
            "role"
        ) != "assistant":
            continue

        contexts = message.get(
            "contexts",
            []
        )

        if (
            isinstance(contexts, list)
            and contexts
        ):

            return [
                chunk
                for chunk in contexts
                if isinstance(
                    chunk,
                    dict
                )
            ]

    return []


def get_previous_user_questions() -> list:

    result = []

    for message in get_prior_messages():

        if message.get(
            "role"
        ) != "user":
            continue

        content = clean_text(
            message.get(
                "content",
                ""
            )
        )

        if content:
            result.append(
                content
            )

    return result[-3:]


def build_contextual_query(
    question: str
) -> str:

    question = clean_text(
        question
    )

    previous = (
        get_previous_user_questions()
    )

    if not previous:
        return question

    if not is_follow_up_question(
        question
    ):
        return question

    previous_text = "\n".join(
        f"Previous user question: {item}"
        for item in previous
    )

    return (
        f"Current question: {question}\n"
        f"{previous_text}"
    )


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def semantic_retrieve(
    query: str,
    limit: int = 10
) -> list:

    if not query:
        return []

    multiplier = int(
        CFG.get(
            "retriever",
            {}
        ).get(
            "search_multiplier",
            3
        )
    )

    search_k = min(
        max(
            limit * multiplier,
            limit * 2
        ),
        len(CHUNKS)
    )

    vector = embed_query(
        query
    )

    if vector.size == 0:
        return []

    distances, indices = index.search(
        vector,
        search_k
    )

    results = []

    for position, idx in enumerate(
        indices[0]
    ):

        if idx < 0 or idx >= len(CHUNKS):
            continue

        results.append(
            {
                "chunk": CHUNKS[idx],
                "semantic": float(
                    distances[0][position]
                )
            }
        )

    return results


# ============================================================
# MAIN RETRIEVAL CONTROLLER
# ============================================================

def retrieve(
    question: str,
    question_type: str,
    limit: int = 6
) -> list:
    """
    Combine:

    - targeted primary-CV retrieval
    - semantic FAISS retrieval
    - previous-turn evidence
    - lexical matching
    """

    contextual_query = build_contextual_query(
        question
    )

    candidates = []

    # --------------------------------------------------------
    # Previous evidence
    # --------------------------------------------------------

    previous_contexts = get_previous_contexts()

    if (
        is_follow_up_question(question)
        and previous_contexts
    ):

        for chunk in previous_contexts:

            candidates.append(
                {
                    "chunk": chunk,
                    "score": 1.25
                }
            )

    # --------------------------------------------------------
    # Targeted primary CV
    # --------------------------------------------------------

    targeted = get_primary_cv_chunks(
        question_type,
        limit=8
    )

    for chunk in targeted:

        score = 0.95

        if question_type in {
            "employment",
            "languages",
            "skills",
            "profile"
        }:

            score += 0.20

        candidates.append(
            {
                "chunk": chunk,
                "score": score
            }
        )

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    semantic_candidates = semantic_retrieve(
        contextual_query,
        limit=12
    )

    query_terms = tokenize(
        contextual_query
    )

    for item in semantic_candidates:

        chunk = item["chunk"]

        searchable = chunk_search_text(
            chunk
        )

        lexical = lexical_score(
            query_terms,
            searchable
        )

        score = (
            item["semantic"] * 0.70
            + lexical * 0.20
        )

        if is_primary_cv(chunk):
            score += 0.10

        # Keep Roche project continuity.
        if question_type == "roche":

            section = str(
                chunk.get(
                    "section",
                    ""
                )
            ).lower()

            if "roche" in section:
                score += 0.15

        candidates.append(
            {
                "chunk": chunk,
                "score": score
            }
        )

    # --------------------------------------------------------
    # Merge duplicates
    # --------------------------------------------------------

    merged = {}

    for item in candidates:

        chunk = item["chunk"]

        text = clean_text(
            chunk.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        key = normalize_text(
            text
        )

        if key not in merged:

            merged[key] = {
                "chunk": chunk,
                "score": item["score"]
            }

        else:

            merged[key]["score"] = max(
                merged[key]["score"],
                item["score"]
            )

    ranked = sorted(
        merged.values(),
        key=lambda item: item["score"],
        reverse=True
    )

    selected = []
    seen = set()

    for item in ranked:

        chunk = item["chunk"]

        normalized = normalize_text(
            chunk.get(
                "text",
                ""
            )
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        selected.append(
            chunk
        )

        if len(selected) >= limit:
            break

    return selected


# ============================================================
# SYSTEM PROMPT
# ============================================================

def get_system_prompt(
    is_german: bool
) -> str:

    if is_german:

        language_rule = (
            "Respond only in German."
        )

        refusal = (
            "Ich habe diese Information nicht in meiner Wissensbasis."
        )

    else:

        language_rule = (
            "Respond only in English."
        )

        refusal = (
            "I don't have that information in my knowledge base."
        )

    return f"""
You are the professional CV assistant for Mohammad Tanzil Alam.

{language_rule}

Your job is to answer recruiter-style questions about
Mohammad's professional background.

FACTUAL RULES:

1. Every factual claim must be supported by the supplied
   knowledge-base evidence.

2. Conversation history is only used to resolve references:
   "there", "that project", "that company", "he", "his", "it".

3. A previous assistant answer is NOT evidence.

4. Never invent or guess missing facts.

5. Never invent employers, dates, job titles, technologies,
   project outcomes, salaries, personal preferences,
   or achievements.

6. Prefer the primary CV when sources overlap.

7. GitHub documentation can support project-specific technical
   details, but should not override the primary CV for
   employment history, dates, job titles, or core skills.

8. If multiple evidence chunks describe the same employer or
   project, synthesize them into ONE coherent answer. Do not
   treat overlapping evidence as a reason to withhold an answer,
   and do not present the same role twice.

8a. The Technical Mentor / Master's Student Mentor role
    (self-employed tutoring) is real professional experience and
    should be mentioned when relevant, but is not a corporate
    employer. When a question specifically asks about
    "employers" or "companies", list it separately from formal
    employers rather than blending it into the same list.

9. Only use the "I don't have that information" refusal when the
   evidence truly does not address the question. If the evidence
   contains the answer but only in a different phrasing, section,
   or format than expected, still answer using what is there.

ANSWER STYLE:

10. Answer the actual question directly.

11. Do NOT reproduce or rewrite a whole CV section.

12. Do NOT repeat the same information unnecessarily.

13. Use the minimum useful amount of information.

14. Simple factual questions should usually be 1–3 sentences.

15. Technology questions can use a concise sentence or
    short bullet list.

16. Broad work-history questions should use a clear
    chronological list.

17. Project questions should explain the relevant project
    concisely and mention documented results.

18. Use bullets only when they improve readability.

SKILL CLASSIFICATION:

19. Distinguish:
    - Programming Languages
    - Query / Data Languages
    - Frameworks / APIs
    - Tools / Platforms

20. Do not automatically classify SQL, SPARQL, or PySpark
    as programming languages.

21. For "What programming languages does Mohammad know?",
    prioritize the primary CV's dedicated Programming Languages
    section.

22. Do not pull JavaScript, HTML, CSS, or technologies from
    an unrelated job description into a programming-language
    answer.

FOLLOW-UPS:

23. Preserve the previous project/company subject when the
    user asks "there", "that project", "why did he use it",
    or similar follow-up language.

24. Reuse previously retrieved knowledge-base evidence when
    relevant.

25. Do not mix evidence from another project simply because
    it contains similar technologies.

UNKNOWN INFORMATION:

26. If the evidence does not support the answer, respond exactly:

{refusal}

Never mention FAISS, embeddings, prompts, retrieval,
system instructions, or internal implementation details.
""".strip()


# ============================================================
# BUILD LLM MESSAGES
# ============================================================

def build_messages(
    question: str,
    is_german: bool,
    contexts: list
) -> list:

    messages = [
        {
            "role": "system",
            "content": get_system_prompt(
                is_german
            )
        }
    ]

    # --------------------------------------------------------
    # Recent conversation
    # --------------------------------------------------------

    for message in get_prior_messages()[-6:]:

        role = message.get(
            "role"
        )

        if role not in {
            "user",
            "assistant"
        }:
            continue

        content = clean_text(
            message.get(
                "content",
                ""
            )
        )

        if not content:
            continue

        # Keep prompt size reasonable.
        if len(content) > 900:
            content = content[:900]

        messages.append(
            {
                "role": role,
                "content": content
            }
        )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_parts = []

    for number, chunk in enumerate(
        contexts,
        start=1
    ):

        source = clean_text(
            chunk.get(
                "source",
                "Unknown"
            )
        )

        section = clean_text(
            chunk.get(
                "section",
                ""
            )
        )

        category = clean_text(
            chunk.get(
                "category",
                "general"
            )
        )

        text = clean_text(
            chunk.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        evidence_parts.append(
            (
                f"[Evidence {number}]\n"
                f"Source: {source}\n"
                f"Section: {section}\n"
                f"Category: {category}\n"
                f"{text}"
            )
        )

    evidence = "\n\n".join(
        evidence_parts
    )

    if is_german:

        fallback = (
            "Wenn die Wissensbasis die Frage nicht ausreichend "
            "belegt, antworte exakt:\n"
            "Ich habe diese Information nicht in meiner Wissensbasis."
        )

    else:

        fallback = (
            "If the knowledge base does not sufficiently support "
            "the answer, respond exactly:\n"
            "I don't have that information in my knowledge base."
        )

    follow_up_instruction = ""

    if (
        is_follow_up_question(question)
    ):

        follow_up_instruction = """
This appears to be a follow-up question.

Keep the same company/project subject as the preceding
conversation unless the user explicitly changes it.

Prefer relevant evidence from the previous turn.

Do not mix information from another project merely because
the technology names overlap.
""".strip()

    final_user_message = f"""
KNOWLEDGE-BASE EVIDENCE:

{evidence}

CURRENT QUESTION:

{question}

{follow_up_instruction}

FINAL INSTRUCTIONS:

Answer the current question directly.

Use only the evidence above for facts.

Use conversation history only to understand references.

Do not copy or rewrite the CV.

Do not add unsupported information.

{fallback}
""".strip()

    messages.append(
        {
            "role": "user",
            "content": final_user_message
        }
    )

    return messages


# ============================================================
# API ERROR HANDLING
# ============================================================

def friendly_generation_error(
    error: Exception,
    is_german: bool
) -> str:

    text = (
        str(error).lower()
        if error
        else ""
    )

    if (
        "429" in text
        or "rate limit" in text
        or "too many requests" in text
    ):

        return (
            "Ich habe gerade kurzzeitig das Anfragelimit erreicht. "
            "Bitte versuche es in einem Moment erneut."
            if is_german
            else
            "I've temporarily reached the request limit. "
            "Please try again in a moment."
        )

    if (
        "timeout" in text
        or "timed out" in text
        or "deadline" in text
        or "504" in text
    ):

        return (
            "Die Anfrage hat zu lange gedauert. "
            "Bitte versuche es erneut."
            if is_german
            else
            "The request took too long to complete. "
            "Please try again."
        )

    if (
        "401" in text
        or "authentication" in text
        or "api key" in text
    ):

        return (
            "Der KI-Dienst ist momentan nicht verfügbar."
            if is_german
            else
            "The AI service is currently unavailable."
        )

    if (
        "model not found" in text
        or "decommissioned" in text
        or "deprecated" in text
    ):

        return (
            "Das konfigurierte KI-Modell ist momentan nicht verfügbar."
            if is_german
            else
            "The configured AI model is currently unavailable."
        )

    return (
        "Es gab gerade ein vorübergehendes Problem. "
        "Bitte versuche es erneut."
        if is_german
        else
        "There was a temporary problem. "
        "Please try again."
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def resolve_k(
    question_type: str,
    explicit_k: int = None
) -> int:
    """
    Broad questions (full work history, profile summary) need
    every employer chunk in context at once, or the model only
    sees a partial history and either omits a role or guesses
    at the gap. Narrower question types use the default budget.
    """

    if explicit_k is not None:
        return explicit_k

    retriever_cfg = CFG.get(
        "retriever",
        {}
    )

    default_k = int(
        retriever_cfg.get(
            "k",
            6
        )
    )

    if question_type in {
        "employment",
        "profile"
    }:

        return int(
            retriever_cfg.get(
                "k_employment",
                default_k
            )
        )

    return default_k


def generate_answer(
    question: str,
    is_german: bool,
    k: int = None
):

    question = clean_text(
        question
    )

    if not question:

        return (
            "Bitte stelle eine Frage."
            if is_german
            else
            "Please ask a question."
        ), []

    if not check_guardrails(
        question
    ):

        return (
            "Ich beantworte nur Fragen zum "
            "beruflichen Werdegang von "
            "Mohammad Tanzil Alam."
            if is_german
            else
            "I can only answer questions about "
            "Mohammad Tanzil Alam's professional "
            "background."
        ), []

    question_type = detect_question_type(
        question
    )

    k = resolve_k(
        question_type,
        explicit_k=k
    )

    contexts = retrieve(
        question,
        question_type,
        limit=k
    )

    if not contexts:

        return (
            "Ich habe diese Information nicht in meiner Wissensbasis."
            if is_german
            else
            "I don't have that information in my knowledge base."
        ), []

    messages = build_messages(
        question=question,
        is_german=is_german,
        contexts=contexts
    )

    configured_model = CFG.get(
        "llm",
        {}
    ).get(
        "model",
        "openai/gpt-oss-20b"
    )

    # Compatibility fallback.
    if configured_model == "llama-3.1-8b-instant":
        model_name = "openai/gpt-oss-20b"
    else:
        model_name = configured_model

    temperature = float(
        CFG.get(
            "llm",
            {}
        ).get(
            "temperature",
            0.1
        )
    )

    max_tokens = int(
        CFG.get(
            "llm",
            {}
        ).get(
            "max_tokens",
            500
        )
    )

    max_retries = int(
        CFG.get(
            "llm",
            {}
        ).get(
            "max_retries",
            2
        )
    )

    last_error = None

    for attempt in range(
        max_retries + 1
    ):

        try:

            response = llm.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if not response.choices:

                raise RuntimeError(
                    "The model returned no choices."
                )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

            if not answer:

                raise RuntimeError(
                    "The model returned an empty response."
                )

            return (
                answer.strip(),
                contexts
            )

        except Exception as error:

            last_error = error

            error_text = str(
                error
            ).lower()

            permanent = (
                "401" in error_text
                or "authentication" in error_text
                or "api key" in error_text
                or "model not found" in error_text
                or "decommissioned" in error_text
                or "deprecated" in error_text
            )

            if permanent:
                break

            if attempt < max_retries:

                time.sleep(
                    min(
                        1.5 * (attempt + 1),
                        4
                    )
                )

    return (
        friendly_generation_error(
            last_error,
            is_german
        ),
        []
    )


# ============================================================
# CHAT RENDERING
# ============================================================

def render_chat_bubble(
    role: str,
    text: str
):

    is_user = (
        role == "user"
    )

    if is_user:

        safe_text = html.escape(
            clean_text(text)
        ).replace(
            "\n",
            "<br>"
        )

        
        user_bubble_html = (
            '<div class="msg-wrap user">'
            '<div class="msg-avatar">You</div>'
            f'<div class="msg-bubble user">{safe_text}</div>'
            "</div>"
        )

        st.markdown(
            user_bubble_html,
            unsafe_allow_html=True
        )

        return

    # Assistant answer: native Markdown rendering for the body,
    # with a flat (single-line, unindented) HTML wrapper around
    # it for the same reason as above.
    assistant_open_html = (
        '<div class="msg-wrap">'
        '<div class="msg-avatar">AI</div>'
        '<div class="msg-bubble">'
    )

    st.markdown(
        assistant_open_html,
        unsafe_allow_html=True
    )

    st.markdown(
        str(text),
        unsafe_allow_html=False
    )

    assistant_close_html = (
        "</div>"
        "</div>"
    )

    st.markdown(
        assistant_close_html,
        unsafe_allow_html=True
    )


# ============================================================
# SESSION MESSAGES
# ============================================================

def add_user_message(
    text: str
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": clean_text(text)
        }
    )


def add_assistant_message(
    text: str,
    contexts: list
):

    # Contexts are the actual KB evidence used for this answer.
    # They remain within this Streamlit session only.

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": clean_text(text),
            "contexts": contexts
        }
    )


# ============================================================
# PROCESS QUESTION
# ============================================================

def process_question(
    question: str,
    is_german: bool
):

    question = clean_text(
        question
    )

    if not question:
        return

    add_user_message(
        question
    )

    with st.spinner(
        "Thinking..."
        if not is_german
        else
        "Verarbeite..."
    ):

        answer, contexts = generate_answer(
            question,
            is_german
        )

    add_assistant_message(
        answer,
        contexts
    )

    st.rerun()


# ============================================================
# MAIN
# ============================================================

def main():

    st.markdown(
        CSS,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Decorative background
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="orb-tr"></div>
        <div class="orb-bl"></div>
        """,
        unsafe_allow_html=True
    )

    persona = CFG.get(
        "persona",
        {}
    )

    name = persona.get(
        "name",
        "Mohammad Tanzil Alam"
    )

    location = persona.get(
        "location",
        "Wuppertal, Germany"
    )

    status = persona.get(
        "status",
        "Available for opportunities"
    )

    github = persona.get(
        "github",
        ""
    )

    linkedin = persona.get(
        "linkedin",
        ""
    )

    tagline = persona.get(
        "tagline",
        "AI-powered CV Assistant"
    )

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        lang = st.radio(
            "",
            [
                "🇬🇧 EN",
                "🇩🇪 DE"
            ],
            horizontal=True,
            key="language",
            label_visibility="collapsed"
        )

        is_german = (
            lang == "🇩🇪 DE"
        )

        st.markdown(
            '<hr class="gold-divider">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="avatar-circle">MTA</div>',
            unsafe_allow_html=True
        )

        status_text = (
            "Offen für Angebote"
            if is_german
            else
            status
        )

        st.markdown(
            f"""
            <div class="profile-name">
                {html.escape(name)}
            </div>

            <div class="profile-loc">
                {html.escape(location)}
            </div>

            <div class="status-badge">
                <div class="status-dot"></div>
                {html.escape(status_text)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<hr class="gold-divider">',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="sidebar-section-label">
                {"Verbinden" if is_german else "Connect"}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Native Streamlit links: avoids raw HTML/SVG fragments.
        if github:

            st.link_button(
                "GitHub",
                github,
                use_container_width=True
            )

        if linkedin:

            st.link_button(
                "LinkedIn",
                linkedin,
                use_container_width=True
            )

    # ========================================================
    # LANGUAGE
    # ========================================================

    is_german = (
        st.session_state.get(
            "language",
            "🇬🇧 EN"
        ) == "🇩🇪 DE"
    )

    query_language = st.query_params.get(
        "lang",
        "en"
    )

    if query_language == "de":
        is_german = True

    # ========================================================
    # MOBILE BAR
    # ========================================================

    english_class = (
        "mobile-lang-btn active"
        if not is_german
        else
        "mobile-lang-btn"
    )

    german_class = (
        "mobile-lang-btn active"
        if is_german
        else
        "mobile-lang-btn"
    )

    mobile_github = (
        f"""
        <a class="mobile-link"
           href="{html.escape(github)}"
           target="_blank"
           rel="noopener noreferrer">
            GH
        </a>
        """
        if github
        else
        ""
    )

    mobile_linkedin = (
        f"""
        <a class="mobile-link"
           href="{html.escape(linkedin)}"
           target="_blank"
           rel="noopener noreferrer">
            in
        </a>
        """
        if linkedin
        else
        ""
    )

    st.markdown(
        f"""
        <div class="mobile-topbar">

            <div style="display:flex;
                        align-items:center;
                        gap:10px;">

                <div class="mobile-avatar">
                    MTA
                </div>

                <div>

                    <div class="mobile-name">
                        {html.escape(name)}
                    </div>

                    <div class="mobile-loc">
                        {html.escape(location)}
                    </div>

                </div>

            </div>

            <div style="display:flex;
                        flex-direction:column;
                        align-items:flex-end;
                        gap:8px;">

                <div class="mobile-links">
                    {mobile_github}
                    {mobile_linkedin}
                </div>

                <div class="mobile-lang-toggle">

                    <a href="?lang=en"
                       class="{english_class}">
                        🇬🇧 EN
                    </a>

                    <a href="?lang=de"
                       class="{german_class}">
                        🇩🇪 DE
                    </a>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # HEADER
    # ========================================================

    if is_german:

        tagline_text = (
            "KI-gestützter Lebenslauf-Assistent"
        )

        title = (
            "Mohammad in unter 2 Minuten kennenlernen"
        )

    else:

        tagline_text = tagline

        title = (
            "Evaluate Mohammad in under 2 minutes"
        )

    st.markdown(
        f"""
        <div class="header-badge">
            <div class="badge-dot"></div>
            {html.escape(tagline_text)}
        </div>

        <h1 class="main-title">
            {html.escape(title)}
        </h1>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # QUICK QUESTIONS
    # ========================================================

    st.markdown(
        f"""
        <div class="section-label">
            {"Schnellfragen" if is_german else "Quick questions"}
        </div>
        """,
        unsafe_allow_html=True
    )

    examples_en = [
        "Where did Mohammad work?",
        "What programming languages does he know?",
        "Summarize his key projects.",
        "What's his work experience?"
    ]

    examples_de = [
        "Wo hat Mohammad gearbeitet?",
        "Welche Programmiersprachen beherrscht er?",
        "Fasse seine wichtigsten Projekte zusammen.",
        "Was ist sein beruflicher Werdegang?"
    ]

    examples = (
        examples_de
        if is_german
        else
        examples_en
    )

    chip_question = None

    row1 = st.columns(2)
    row2 = st.columns(2)

    grid = [
        row1[0],
        row1[1],
        row2[0],
        row2[1]
    ]

    for i, example in enumerate(
        examples
    ):

        if grid[i].button(
            example,
            key=f"chip_{i}"
        ):

            chip_question = example

    if chip_question:

        process_question(
            chip_question,
            is_german
        )

    st.markdown(
        "<div style='height:1.2rem'></div>",
        unsafe_allow_html=True
    )

    # ========================================================
    # CHAT
    # ========================================================

    welcome_message = persona.get(
        "welcome_message",
        "Hi! I'm Mohammad's AI assistant. "
        "Ask me anything about his background, "
        "projects, or experience."
    )

    welcome = (
        "Hallo! Ich bin Mohammads KI-Assistent. "
        "Frag mich alles über seinen Werdegang, "
        "Projekte oder Erfahrungen."
        if is_german
        else
        welcome_message
    )

    if not st.session_state.messages:

        render_chat_bubble(
            "assistant",
            welcome
        )

    else:

        for message in st.session_state.messages:

            render_chat_bubble(
                message.get(
                    "role",
                    "assistant"
                ),
                message.get(
                    "content",
                    ""
                )
            )

    # ========================================================
    # FOOTER
    # ========================================================

    footer = (
        "Antworten basieren auf verifizierten Profildaten"
        if is_german
        else
        "Responses grounded in verified profile data"
    )

    st.markdown(
        f"""
        <div class="footer-bar">
            {html.escape(footer)}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # CHAT INPUT
    # ========================================================

    placeholder = (
        "Stell eine Frage über Mohammads Hintergrund..."
        if is_german
        else
        "Ask anything about Mohammad's background..."
    )

    question = st.chat_input(
        placeholder
    )

    if question and question.strip():

        process_question(
            question.strip(),
            is_german
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()