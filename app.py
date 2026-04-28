import os, json, yaml, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

st.set_page_config(
    page_title="Mohammad Tanzil Alam — CV Assistant",
    page_icon="🧠",
    layout="wide"
)

@st.cache_data
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

try:
    CFG = load_config()
except FileNotFoundError:
    st.error("config.yaml not found.")
    st.stop()

@st.cache_resource
def load_model():
    return Groq(api_key=st.secrets["pikaboo"])

@st.cache_resource
def load_retrieval_components():
    try:
        embedder = SentenceTransformer(CFG["embeddings"]["model_name"])
        index_path = "index/faiss.index"
        chunks_path = "index/chunks.jsonl"
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            st.error("Index files not found. Run prep_index.py first.")
            st.stop()
        index = faiss.read_index(index_path)
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunks.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return embedder, index, chunks
    except Exception as e:
        st.error(f"Failed to load retrieval components: {e}")
        st.stop()

llm = load_model()
embedder, index, CHUNKS = load_retrieval_components()

def _embed(texts):
    try:
        vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs).astype("float32")
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return np.array([]).astype("float32")

def retrieve(query, k):
    try:
        qv = _embed([query])
        if qv.size == 0:
            return []
        search_k = min(k * 3, len(CHUNKS))
        D, I = index.search(qv, search_k)
        results, seen = [], set()
        for idx in I[0]:
            if idx == -1 or idx >= len(CHUNKS):
                continue
            chunk = CHUNKS[idx]
            txt = chunk.get("text", "")
            if txt in seen or len(txt.strip()) < 10:
                continue
            seen.add(txt)
            results.append(chunk)
            if len(results) >= k:
                break
        return results
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return []

def get_system_prompt(is_german):
    lang = (
        "Antworte IMMER auf Deutsch, egal in welcher Sprache die Frage gestellt wird."
        if is_german else "Always respond in English."
    )
    return f"""You are an AI assistant representing {CFG['persona']['name']}.
{lang}

CRITICAL RULES:
1. Answer ONLY using the provided CONTEXT from CV & project information
2. If the answer is not in the CONTEXT, respond: "I don't have that information in my knowledge base."
3. Be {CFG['persona']['style']}
4. Never fabricate information
5. Always positive and professional — highlight strengths only
6.  When answering about technical skills, organize them clearly:
   - Programming Languages: (Python, Java, C++, JavaScript, etc.)
   - Frameworks & Libraries: (FastAPI, PyTorch, React, scikit-learn, etc.) 
   - Tools & Platforms: (AWS, Docker, Git, etc.)
   - Don't duplicate items across categories
7. Only list skills explicitly mentioned in context
8. Present information in a confident, professional manner that showcases expertise

Keep responses focused, accurate, and well-organized."""

def check_guardrails(text):
    for phrase in CFG.get("guardrails", {}).get("blocked_phrases", []):
        if phrase.lower() in text.lower():
            return False
    return True

def preprocess_question(question):
    q = question.lower()
    if any(w in q for w in ["programming language", "languages do you", "proficient", "programmiersprachen"]):
        return (question + "\n\nNote: Organize by categories (Programming Languages, Frameworks & Libraries, "
                "Tools & Platforms). Only include skills explicitly in context.")
    elif any(w in q for w in ["skill", "fahigkeiten"]) and any(w in q for w in ["technical", "key", "technische"]):
        return (question + "\n\nNote: Organize by categories, highlight expertise. Only mention skills in context.")
    return question

def generate_answer(question, is_german, k=5):
    if not check_guardrails(question):
        return (
            "Ich beantworte nur Fragen zum beruflichen Werdegang von Mohammad Tanzil Alam."
            if is_german else
            "I can only answer questions about Mohammad Tanzil Alam's professional background."
        ), []
    processed = preprocess_question(question)
    contexts = retrieve(processed, k)
    if not contexts:
        return (
            "Ich habe keine relevanten Informationen dazu."
            if is_german else
            "I don't have relevant information to answer that question."
        ), []
    try:
        context_block = ""
        for i, chunk in enumerate(contexts, 1):
            context_block += f"[{i}] ({chunk.get('source','?')} | {chunk.get('section','')}) {chunk.get('text','')}\n\n"
        user_message = f"Question: {processed}\n\nCONTEXT (use ONLY this):\n{context_block}\nAnswer concisely based solely on the context above."
        response = llm.chat.completions.create(
            model=CFG["llm"]["model"],
            messages=[
                {"role": "system", "content": get_system_prompt(is_german)},
                {"role": "user", "content": user_message}
            ],
            temperature=CFG["llm"]["temperature"],
            max_tokens=CFG["llm"]["max_tokens"]
        )
        return response.choices[0].message.content.strip(), contexts
    except Exception as e:
        st.error(f"Generation error: {e}")
        return "Sorry, an error occurred.", []


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body { color-scheme: only light !important; }
*, *::before, *::after { font-family: 'Inter', sans-serif !important; }

[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child > div:first-child button { display: none !important; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main { background: #1c1508 !important; }

.main .block-container { padding: 2rem 2.5rem 6rem !important; max-width: 920px !important; }

.orb-tr {
    position: fixed; top: -120px; right: -100px;
    width: 420px; height: 420px; border-radius: 50%;
    background: radial-gradient(circle, rgba(212,175,88,0.11) 0%, transparent 68%);
    pointer-events: none; z-index: 0;
}
.orb-bl {
    position: fixed; bottom: -80px; left: -80px;
    width: 280px; height: 280px; border-radius: 50%;
    background: radial-gradient(circle, rgba(180,130,40,0.07) 0%, transparent 68%);
    pointer-events: none; z-index: 0;
}

[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: #130e03 !important; }
[data-testid="stSidebar"] { border-right: 0.5px solid rgba(212,175,88,0.14) !important; }
[data-testid="stSidebar"] .block-container { padding: 2rem 1.4rem !important; }

[data-testid="stSidebar"] .stRadio > div { gap: 6px !important; flex-direction: row !important; }
[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border: 0.5px solid rgba(212,175,88,0.25) !important;
    border-radius: 100px !important; padding: 5px 16px !important;
    font-size: 13px !important; color: #a89060 !important;
    cursor: pointer !important; transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(212,175,88,0.14) !important;
    border-color: rgba(212,175,88,0.5) !important; color: #d4af58 !important;
}
[data-testid="stSidebar"] .stRadio > label { display: none !important; }

.avatar-circle {
    width: 90px; height: 90px; border-radius: 50%;
    background: rgba(212,175,88,0.08);
    border: 2px solid rgba(212,175,88,0.45);
    box-shadow: 0 0 20px rgba(212,175,88,0.18);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; font-weight: 600; color: #d4af58;
    margin: 0 auto 12px auto;
    overflow: hidden;
}
.avatar-circle img { width: 100%; height: 100%; object-fit: cover; border-radius: 50%; }

.profile-name { font-size: 16px; font-weight: 600; color: #f0e8d0; line-height: 1.3; text-align: center; }
.profile-loc { font-size: 13px; color: #6a5c3a; margin-top: 4px; text-align: center; }
.status-badge { display: flex; align-items: center; justify-content: center; gap: 7px; font-size: 13px; color: #7a9060; margin-top: 6px; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #7a9060; flex-shrink: 0; animation: softpulse 2.5s infinite; }
@keyframes softpulse { 0%,100%{opacity:0.35} 50%{opacity:1} }

.gold-divider { border: none; border-top: 0.5px solid rgba(212,175,88,0.1); margin: 16px 0; }
.sidebar-section-label { font-size: 10px; color: #4a3f22; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px; }
.sidebar-link { display: flex; align-items: center; gap: 10px; padding: 7px 0; font-size: 15px; color: #c8b070; text-decoration: none !important; transition: color 0.15s; }
.sidebar-link:hover { color: #d4af58; }
.sidebar-link-icon { width: 24px; height: 24px; border: 0.5px solid rgba(212,175,88,0.25); border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }

.header-badge { display: inline-flex; align-items: center; gap: 8px; font-size: 11px; color: #a89060; background: rgba(212,175,88,0.06); border: 0.5px solid rgba(212,175,88,0.25); border-radius: 100px; padding: 5px 14px; margin-bottom: 14px; }
.badge-dot { width: 6px; height: 6px; border-radius: 50%; background: #d4af58; animation: softpulse 2s infinite; }

.main-title { font-size: 28px; font-weight: 600; color: #d4af58; letter-spacing: -0.3px; line-height: 1.2; margin: 0 0 2rem 0; white-space: nowrap; }

.section-label { font-size: 10px; color: #4a3f22; text-transform: uppercase; letter-spacing: 2px; margin: 0 0 10px 0; }

.stButton > button {
    background: rgba(212,175,88,0.05) !important; border: 0.5px solid rgba(212,175,88,0.2) !important;
    border-radius: 10px !important; color: #c8b070 !important; font-size: 14px !important;
    font-weight: 400 !important; padding: 0.75rem 1rem !important; line-height: 1.45 !important;
    white-space: normal !important; height: auto !important; text-align: left !important;
    transition: all 0.15s !important; width: 100% !important;
}
.stButton > button:hover { border-color: rgba(212,175,88,0.5) !important; color: #d4af58 !important; box-shadow: 0 0 12px rgba(212,175,88,0.12) !important; background: rgba(212,175,88,0.08) !important; }
.stButton > button:active { background: rgba(212,175,88,0.15) !important; border-color: rgba(212,175,88,0.6) !important; }

.msg-wrap { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 16px; }
.msg-wrap.user { flex-direction: row-reverse; }
.msg-avatar { width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0; background: rgba(212,175,88,0.10); border: 0.5px solid rgba(212,175,88,0.3); display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 500; color: #d4af58; }
.msg-bubble { max-width: 80%; background: rgba(255,255,255,0.04); border: 0.5px solid rgba(212,175,88,0.12); border-radius: 14px 14px 14px 4px; padding: 13px 17px; font-size: 15px; color: #c8b88a; line-height: 1.7; }
.msg-bubble.user { background: rgba(212,175,88,0.09); border: 0.5px solid rgba(212,175,88,0.42); border-radius: 14px 14px 4px 14px; color: #f0e0a8; box-shadow: 0 0 18px rgba(212,175,88,0.14), inset 0 0 8px rgba(212,175,88,0.04); }

[data-testid="stBottom"], [data-testid="stBottom"] > div { background: #1c1508 !important; }
[data-testid="stBottom"] { border-top: 0.5px solid rgba(212,175,88,0.1) !important; padding: 10px 2.5rem !important; }
[data-testid="stChatInput"] { background: rgba(212,175,88,0.05) !important; border: 1px solid rgba(212,175,88,0.38) !important; border-radius: 12px !important; box-shadow: 0 0 20px rgba(212,175,88,0.13), 0 0 6px rgba(212,175,88,0.07) !important; }
[data-testid="stChatInput"]:focus-within { border-color: rgba(212,175,88,0.7) !important; box-shadow: 0 0 28px rgba(212,175,88,0.22), 0 0 10px rgba(212,175,88,0.12) !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: #f0e8d0 !important; font-size: 15px !important; caret-color: #d4af58 !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #5a4e30 !important; }
[data-testid="stChatInputSubmitButton"] > button { background: rgba(212,175,88,0.2) !important; border: 0.5px solid rgba(212,175,88,0.45) !important; border-radius: 8px !important; color: #d4af58 !important; }
[data-testid="stChatInputSubmitButton"] > button:hover { background: rgba(212,175,88,0.35) !important; box-shadow: 0 0 14px rgba(212,175,88,0.28) !important; }

.stSpinner > div { border-top-color: #d4af58 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(212,175,88,0.2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(212,175,88,0.4); }
.footer-bar { text-align: center; font-size: 11px; color: #3a3018; letter-spacing: 0.4px; margin-top: 1rem; padding-top: 1rem; border-top: 0.5px solid rgba(212,175,88,0.07); }

@media (max-width: 768px) {
    .main .block-container { padding: 1rem 1rem 6rem !important; }
    .main-title { font-size: 20px !important; white-space: normal !important; }
    .avatar-circle { width: 70px !important; height: 70px !important; font-size: 18px !important; }
}
</style>
"""

GITHUB_SVG = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#d4af58" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>'
LINKEDIN_SVG = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#d4af58" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/><rect x="2" y="9" width="4" height="12"/><circle cx="4" cy="4" r="2"/></svg>'

def render_chat_bubble(role, text):
    is_user = role == "user"
    cls = "msg-bubble user" if is_user else "msg-bubble"
    label = "You" if is_user else "AI"
    align = "user" if is_user else ""
    st.markdown(f'<div class="msg-wrap {align}"><div class="msg-avatar">{label}</div><div class="{cls}">{text}</div></div>', unsafe_allow_html=True)

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="orb-tr"></div><div class="orb-bl"></div>', unsafe_allow_html=True)

    with st.sidebar:
        lang = st.radio("", ["🇬🇧 EN", "🇩🇪 DE"], horizontal=True, key="language", label_visibility="collapsed")
        IS_GERMAN = lang == "🇩🇪 DE"
        st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
        st.markdown('<div class="avatar-circle">MTA</div>', unsafe_allow_html=True)
        status_txt = "Offen fur Angebote" if IS_GERMAN else CFG['persona'].get('status', 'Available for opportunities')
        st.markdown(f'<div class="profile-name">{CFG["persona"]["name"]}</div><div class="profile-loc">{CFG["persona"]["location"]}</div><div class="status-badge"><div class="status-dot"></div>{status_txt}</div>', unsafe_allow_html=True)
        st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-section-label">{"Verbinden" if IS_GERMAN else "Connect"}</div>', unsafe_allow_html=True)
        if CFG["persona"].get("github"):
            st.markdown(f'<a class="sidebar-link" href="{CFG["persona"]["github"]}" target="_blank"><div class="sidebar-link-icon">{GITHUB_SVG}</div>GitHub</a>', unsafe_allow_html=True)
        if CFG["persona"].get("linkedin"):
            st.markdown(f'<a class="sidebar-link" href="{CFG["persona"]["linkedin"]}" target="_blank"><div class="sidebar-link-icon">{LINKEDIN_SVG}</div>LinkedIn</a>', unsafe_allow_html=True)

    IS_GERMAN = st.session_state.get("language", "🇬🇧 EN") == "🇩🇪 DE"
    tagline = "KI-gestutzter Lebenslauf-Assistent" if IS_GERMAN else CFG['persona']['tagline']
    title   = "Mohammad in unter 2 Minuten kennenlernen" if IS_GERMAN else "Evaluate Mohammad in under 2 minutes"

    st.markdown(f'<div class="header-badge"><div class="badge-dot"></div>{tagline}</div><h1 class="main-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">{"Schnellfragen" if IS_GERMAN else "Quick questions"}</div>', unsafe_allow_html=True)

    EXAMPLES_EN = ["What programming languages does he know?","Summarize his key projects","Which frameworks has he used?","What's his work experience?"]
    EXAMPLES_DE = ["Welche Programmiersprachen beherrscht er?","Fasse seine wichtigsten Projekte zusammen","Welche Frameworks hat er verwendet?","Was ist sein beruflicher Werdegang?"]
    EXAMPLES = EXAMPLES_DE if IS_GERMAN else EXAMPLES_EN

    chip_question = None
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        if cols[i].button(ex, key=f"chip_{i}"):
            chip_question = ex

    if chip_question:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": chip_question})
        with st.spinner("Thinking..." if not IS_GERMAN else "Verarbeite..."):
            answer, _ = generate_answer(chip_question, IS_GERMAN)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    welcome = "Hallo! Ich bin Mohammads KI-Assistent. Frag mich alles über seinen Werdegang, Projekte oder Erfahrungen." if IS_GERMAN else CFG['persona']['welcome_message']

    if not st.session_state.messages:
        render_chat_bubble("assistant", welcome)
    else:
        for msg in st.session_state.messages:
            render_chat_bubble(msg["role"], msg["content"])

    st.markdown(f'<div class="footer-bar">{"Antworten basieren auf verifizierten Profildaten" if IS_GERMAN else "Responses grounded in verified profile data"}</div>', unsafe_allow_html=True)

    placeholder = "Stell eine Frage über Mohammads Hintergrund..." if IS_GERMAN else "Ask anything about Mohammad's background..."
    question = st.chat_input(placeholder)

    if question and question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..." if not IS_GERMAN else "Verarbeite..."):
            answer, _ = generate_answer(question, IS_GERMAN)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == "__main__":
    main()
