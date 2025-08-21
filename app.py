import os, json, yaml, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import streamlit as st

# ---------- Page config (must be first Streamlit call) ----------
st.set_page_config(
    page_title="Ask CV Assistant",
    page_icon="👤",
    layout="wide"
)

# ---------- Load config ----------
@st.cache_data
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

try:
    CFG = load_config()
except FileNotFoundError:
    st.error("❌ config.yaml not found! Please ensure it exists in the root directory.")
    st.stop()

# ---------- Ensure models folder exists ----------
os.makedirs("models", exist_ok=True)

# ---------- Download and load model ----------
@st.cache_resource
def load_model():
    local_model_path = os.path.join("models", os.path.basename(CFG["llm"]["gguf_file"]))
    
    if not os.path.exists(local_model_path):
        st.info("🔄 Model not found locally. Downloading from Hugging Face...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=CFG["llm"]["gguf_repo"],
                filename=CFG["llm"]["gguf_file"],
                local_dir="models",
                local_dir_use_symlinks=False
            )
            local_model_path = downloaded_path if os.path.exists(downloaded_path) else local_model_path
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()
    
    try:
        return Llama(
            model_path=local_model_path,
            n_ctx=CFG["llm"].get("n_ctx", 4096),
            n_threads=CFG["llm"].get("n_threads", 4),
            n_gpu_layers=0,
            verbose=False
        )
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# ---------- Load embeddings & FAISS index ----------
@st.cache_resource
def load_retrieval_components():
    try:
        embedder = SentenceTransformer(CFG["embeddings"]["model_name"])
        index_path = "index/faiss.index"
        chunks_path = "index/chunks.jsonl"
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            st.error("❌ Index files not found! Please run prep_index.py first.")
            st.info("Run: python prep_index.py to create the knowledge base.")
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
        st.error(f"❌ Failed to load retrieval components: {e}")
        st.stop()

# ---------- Initialize components ----------
llm = load_model()
embedder, index, CHUNKS = load_retrieval_components()

def _embed(texts):
    try:
        vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs).astype("float32")
    except Exception as e:
        st.error(f"❌ Embedding error: {e}")
        return np.array([]).astype("float32")

def retrieve(query, k):
    try:
        qv = _embed([query])
        if qv.size == 0:
            return []
        
        search_k = min(k * 3, len(CHUNKS))
        D, I = index.search(qv, search_k)
        
        results = []
        seen_texts = set()
        for idx in I[0]:
            if idx == -1 or idx >= len(CHUNKS):
                continue
            
            chunk = CHUNKS[idx]
            chunk_text = chunk.get("text", "")
            
            if chunk_text in seen_texts or len(chunk_text.strip()) < 10:
                continue
            
            seen_texts.add(chunk_text)
            results.append(chunk)
            
            if len(results) >= k:
                break
        
        return results
    except Exception as e:
        st.error(f"❌ Retrieval error: {e}")
        return []

# ---------- System prompt ----------
SYSTEM_PROMPT = f"""You are an AI assistant representing {CFG['persona']['name']}.

CRITICAL RULES:
1. Answer questions ONLY using the provided CONTEXT from CV & project information
2. If the answer is not in the CONTEXT, respond: "I don't have that information in my knowledge base."
3. Always cite sources using [n] format
4. Be {CFG['persona']['style']}
5. Do NOT make up or infer information not explicitly stated in the context
6. ALWAYS provide POSITIVE, professional responses - focus on skills, experience and achievements present in the context
7. NEVER mention what someone is "not proficient in" or "lacks" - only highlight what they ARE skilled at
8. When answering about technical skills, organize them clearly:
   - Programming Languages: (Python, Java, C++, JavaScript, etc.)
   - Frameworks & Libraries: (FastAPI, PyTorch, React, scikit-learn, etc.) 
   - Tools & Platforms: (AWS, Docker, Git, etc.)
   - Don't duplicate items across categories
9. Present information in a confident, professional manner that showcases expertise
10. If asked about skills/languages, only list those explicitly mentioned in the context

Keep responses focused, positive, accurate, and well-organized."""

def check_guardrails(text):
    text_lower = text.lower()
    blocked = CFG.get("guardrails", {}).get("blocked_phrases", [])
    for phrase in blocked:
        if phrase.lower() in text_lower:
            return False
    return True

def preprocess_question(question):
    question_lower = question.lower()
    if "programming language" in question_lower or "languages do you" in question_lower or "proficient" in question_lower:
        return f"{question}\n\nNote: Please organize the response by clear categories (Programming Languages, Frameworks & Libraries, Tools & Platforms) and only include skills explicitly mentioned in the context. Focus on showcasing expertise and achievements."
    elif "skill" in question_lower and ("technical" in question_lower or "key" in question_lower):
        return f"{question}\n\nNote: Please organize the response by categories and highlight the person's expertise and accomplishments. Only mention skills that are explicitly stated in the context."
    return question

def format_ai_response(text):
    """Format AI response text with cyberpunk terminal styling"""
    if not text:
        return text
    
    # Replace bullet points and structure the content
    text = text.replace('*', '')  # Remove asterisks
    text = text.replace('+ ', '> ')  # Replace + with terminal-style prompt
    text = text.replace('- ', '> ')  # Replace - with terminal-style prompt
    
    # Format sections with cyberpunk styling
    formatted_lines = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Main headers (like "Professional Experience:", "Technical Skills:")
        if line.endswith(':') and not line.startswith('>'):
            formatted_lines.append(f"""
<div style="margin: 1.5rem 0 0.8rem 0; padding: 1rem; background: linear-gradient(90deg, #00ffcc, #0077ff); border-radius: 8px; color: #0f0f0f; border-left: 4px solid #00aaff;">
    <h4 style="margin: 0; font-weight: 600; font-size: 1.2rem; font-family: 'JetBrains Mono', monospace;">
        >> {line}
    </h4>
</div>
""")
        # Sub-headers (job titles, project names)
        elif line.startswith('*') and ':' in line:
            clean_line = line.replace('*', '').strip()
            formatted_lines.append(f"""
<div style="margin: 1rem 0 0.5rem 0; padding: 0.8rem; background: #0f172a; border: 1px solid #00aaff; border-radius: 6px;">
    <h5 style="margin: 0; color: #00ffcc; font-weight: 600; font-size: 1rem; font-family: 'JetBrains Mono', monospace;">
        🔹 {clean_line}
    </h5>
</div>
""")
        # Terminal-style bullet points
        elif line.startswith('>'):
            clean_line = line.replace('>', '').strip()
            formatted_lines.append(f"""
<div style="margin: 0.4rem 0; padding: 0.6rem 1rem; background: rgba(0, 255, 204, 0.1); border-left: 3px solid #00ffcc; border-radius: 4px;">
    <span style="color: #e0e0e0; font-size: 0.95rem; font-family: 'JetBrains Mono', monospace;">
        <span style="color: #00ffcc;">></span> {clean_line}
    </span>
</div>
""")
        # Regular paragraphs
        else:
            if line:
                formatted_lines.append(f"""
<p style="margin: 0.8rem 0; color: #e0e0e0; line-height: 1.6; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem;">
    {line}
</p>
""")
    
    return ''.join(formatted_lines)

def generate_answer(question, k=5):
    if not check_guardrails(question):
        return "I can only answer questions about Mohammad Tanzil Alam's professional background and experience.", []
    
    processed_question = preprocess_question(question)
    contexts = retrieve(processed_question, k)
    
    if not contexts:
        return "I don't have relevant information to answer that question.", []
    
    try:
        context_window = llm.n_ctx()
        max_output_tokens = CFG["llm"].get("max_tokens", 512)
        system_tokens = len(SYSTEM_PROMPT) // 4
        question_tokens = len(question) // 4
        max_context_chars = (context_window - max_output_tokens - system_tokens - question_tokens - 100) * 4
        
        context_block = ""
        for i, chunk in enumerate(contexts, 1):
            source = chunk.get('source', 'Unknown')
            section = chunk.get('section', '')
            text = chunk.get('text', '')
            
            chunk_text = f"[{i}] ({source} | {section}) {text}\n\n"
            if len(context_block + chunk_text) > max_context_chars:
                break
            context_block += chunk_text
        
        user_message = f"""Question: {processed_question}

CONTEXT (use ONLY this information to answer):
{context_block}

Provide a concise answer with citations [n] based solely on the context above."""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        response = llm.create_chat_completion(
            messages=messages,
            temperature=CFG["llm"]["temperature"],
            top_p=CFG["llm"]["top_p"],
            max_tokens=max_output_tokens,
            stop=["\n\nUser:", "\n\nHuman:"]
        )
        
        answer = response["choices"][0]["message"]["content"].strip()
        return answer, contexts
    
    except Exception as e:
        st.error(f"❌ Generation error: {e}")
        return "Sorry, I encountered an error while processing your question.", []

# ---------- Streamlit UI ----------
def main():
    # Cyberpunk/Neon CSS Styling
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

/* Global */
.stApp {
    background: radial-gradient(circle at 20% 20%, #0f2027, #203a43, #2c5364);
    font-family: 'JetBrains Mono', monospace;
    color: #e0e0e0;
}

/* Block container */
.main .block-container {
    padding: 2rem;
    max-width: 1100px;
}

/* Header with cyberpunk glow */
.header-container {
    background: rgba(15, 15, 30, 0.85);
    border: 1px solid rgba(0, 255, 180, 0.3);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 0 25px rgba(0, 255, 180, 0.3);
}

.header-container h1 {
    font-size: 2.3rem;
    font-weight: 600;
    background: linear-gradient(90deg, #00ffcc, #00aaff, #ff00aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: hueShift 6s infinite linear;
}

@keyframes hueShift {
    0% { filter: hue-rotate(0deg); }
    100% { filter: hue-rotate(360deg); }
}

/* Avatar styling */
.profile-image {
    border-radius: 50%;
    border: 3px solid #00ffcc;
    box-shadow: 0 0 20px rgba(0, 255, 180, 0.6);
}

/* Cyber buttons */
.stButton > button {
    background: linear-gradient(90deg, #00ffcc, #0077ff) !important;
    color: #0f0f0f !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.4rem !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    transition: 0.3s !important;
    box-shadow: 0 0 12px rgba(0,255,180,0.5) !important;
    border: none !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 25px rgba(0,255,180,0.9) !important;
}

/* Input areas styled like a code editor */
.stTextArea textarea {
    background: #111827 !important;
    color: #00ffcc !important;
    border: 1px solid #00aaff !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
}

.stTextArea textarea:focus {
    border-color: #00ffcc !important;
    box-shadow: 0 0 15px rgba(0, 255, 180, 0.5) !important;
}

/* Answer container (terminal style) */
.answer-container {
    background: #0d1117;
    border-left: 4px solid #00ffcc;
    padding: 1.5rem;
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: #e0e0e0;
    box-shadow: 0 0 15px rgba(0, 255, 180, 0.2);
    position: relative;
}

.answer-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00ffcc, #0077ff, #ff00aa);
    background-size: 200% 100%;
    animation: neonFlow 3s ease infinite;
}

@keyframes neonFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Source container with hover glow */
.source-container {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 1rem;
    transition: 0.3s;
    margin-bottom: 1rem;
}

.source-container:hover {
    border-color: #00ffcc;
    box-shadow: 0 0 15px rgba(0,255,180,0.5);
}

/* Sidebar */
.css-1d391kg {
    background: rgba(15, 15, 30, 0.9) !important;
    border-right: 2px solid #00aaff !important;
}

/* Quick questions styling */
.quick-questions {
    background: rgba(15, 15, 30, 0.7);
    border: 1px solid #00aaff;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
}

/* Metrics styling */
.css-1xarl3l {
    background: rgba(17, 24, 39, 0.8) !important;
    border: 1px solid #00aaff !important;
    border-radius: 8px !important;
}

/* Scrollbar neon */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(15, 15, 30, 0.5);
    border-radius: 6px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00ffcc, #0077ff);
    border-radius: 6px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #00aaff, #ff00aa);
}

/* Warning and info boxes */
.stAlert {
    background: rgba(17, 24, 39, 0.9) !important;
    border: 1px solid #00aaff !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
}

/* Loading spinner */
.stSpinner > div {
    border-color: #00ffcc !important;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .header-container {
        padding: 1.5rem;
    }
    
    .answer-container {
        padding: 1.2rem;
    }
}
</style>
""", unsafe_allow_html=True)

    # Cyberpunk Header
    st.markdown(f"""
<div class="header-container">
    <div style="display: flex; align-items: center; gap: 2rem;">
        <div>
        </div>
        <div>
            <h1 style="margin: 0; font-family: 'JetBrains Mono', monospace;">
                👤 {CFG['persona']['name']}
            </h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; color: #00ffcc; font-weight: 600;">
                {CFG['persona'].get('tagline', 'AI-powered CV Assistant')}
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; color: #888; font-family: 'JetBrains Mono', monospace;">
                🚀 Terminal-style AI assistant for professional background queries
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
<div style="text-align: center; padding: 1rem; background: rgba(0, 255, 204, 0.1); border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: #00ffcc; margin-bottom: 1rem; font-family: 'JetBrains Mono', monospace;">⚡ System Info</h2>
</div>
""", unsafe_allow_html=True)
        
        st.info(f"🤖 This terminal assists with queries about {CFG['persona']['name']} using RAG technology.")
        
        st.markdown("""
<div style="background: rgba(0, 255, 204, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0; border: 1px solid #00aaff;">
    <h3 style="color: #00ffcc; margin-bottom: 0.5rem; font-family: 'JetBrains Mono', monospace;">📊 Database Stats</h3>
</div>
""", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Chunks", f"{len(CHUNKS)}")
        with col2:
            st.metric("🧠 Model", CFG["llm"]["gguf_repo"].split("/")[-1].split("-")[0])
        
        st.markdown("""
<div style="background: rgba(0, 255, 204, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0; border: 1px solid #00aaff;">
    <h3 style="color: #00ffcc; margin-bottom: 1rem; font-family: 'JetBrains Mono', monospace;">🌐 Network Links</h3>
</div>
""", unsafe_allow_html=True)
        
        if CFG["persona"].get("github"):
            st.markdown(f"""
<a href="{CFG['persona']['github']}" target="_blank" style="
    display: inline-block; 
    padding: 0.5rem 1rem; 
    margin: 0.2rem; 
    background: linear-gradient(90deg, #00ffcc, #0077ff); 
    color: #0f0f0f; 
    text-decoration: none; 
    border-radius: 8px; 
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    transition: all 0.3s ease;
">🐙 GitHub</a>
""", unsafe_allow_html=True)
        
        if CFG["persona"].get("linkedin"):
            st.markdown(f"""
<a href="{CFG['persona']['linkedin']}" target="_blank" style="
    display: inline-block; 
    padding: 0.5rem 1rem; 
    margin: 0.2rem; 
    background: linear-gradient(90deg, #00ffcc, #0077ff); 
    color: #0f0f0f; 
    text-decoration: none; 
    border-radius: 8px; 
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    transition: all 0.3s ease;
">💼 LinkedIn</a>
""", unsafe_allow_html=True)

    # Quick Questions Section
    st.markdown("""
<div class="quick-questions">
    <h3 style="margin-bottom: 1rem; color: #00ffcc; font-weight: 600; font-family: 'JetBrains Mono', monospace;">
        💡 Quick Terminal Commands
    </h3>
</div>
""", unsafe_allow_html=True)
    
    EXAMPLES = [
        "🔧 What programming languages are you proficient in?",
        "💼 Summarize key projects and experiences", 
        "🌐 Which frameworks and tools have you used?"
    ]
    
    example_buttons = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if example_buttons[i].button(ex, key=f"example_{i}"):
            st.session_state["question"] = ex.split(" ", 1)[1]  # Remove emoji for cleaner input

    # Input Section
    st.markdown("""
<div style="margin: 2rem 0 1rem 0;">
    <h3 style="color: #00ffcc; font-weight: 600; font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 10px rgba(0, 255, 180, 0.5);">
        🤔 Terminal Query Interface
    </h3>
    <p style="color: #888; font-family: 'JetBrains Mono', monospace;">
        > Enter your query about Mohammad Tanzil Alam's professional profile
    </p>
</div>
""", unsafe_allow_html=True)
    
    question = st.text_area(
        "",
        value=st.session_state.get("question", ""),
        placeholder="$ query --about 'technical skills' --format 'detailed'\n$ query --about 'recent projects' --citations\n$ query --about 'experience with machine learning'",
        height=120,
        key="question_input"
    )

    # Ask Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ask_button = st.button("🚀 Execute Query", key="ask_button", type="primary")

    if ask_button:
        if question.strip() == "":
            st.warning("⚠️ Please enter a query to execute!")
        else:
            with st.spinner("🤖 Processing query through neural network..."):
                answer, sources = generate_answer(question)
            
            # Format and display answer
            formatted_answer = format_ai_response(answer)
            
            st.markdown(f"""
<div class="answer-container">
    <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.5rem; border-bottom: 2px solid #00aaff; padding-bottom: 0.8rem;">
        <div style="background: linear-gradient(45deg, #00ffcc, #0077ff); padding: 0.5rem; border-radius: 8px;">
            <span style="font-size: 1.5rem;">🤖</span>
        </div>
        <div>
            <h3 style="margin: 0; color: #00ffcc; font-weight: 600; font-family: 'JetBrains Mono', monospace;">Terminal Output</h3>
            <p style="margin: 0; color: #888; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">Query executed successfully</p>
        </div>
    </div>
    <div style="font-size: 1rem; line-height: 1.7;">
        {formatted_answer}
    </div>
</div>
""", unsafe_allow_html=True)
            
            # Sources display
            if sources:
                st.markdown("""
<div style="margin: 2rem 0 1rem 0;">
    <h4 style="color: #00ffcc; font-weight: 600; font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 10px rgba(0, 255, 180, 0.5);">
        📚 Source References & Citations
    </h4>
</div>
""", unsafe_allow_html=True)
                
                for i, chunk in enumerate(sources, 1):
                    src = chunk.get("source", "Unknown")
                    sec = chunk.get("section", "")
                    txt = chunk.get("text", "")
                    
                    # Truncate long text
                    display_text = txt[:300] + "..." if len(txt) > 300 else txt
                    
                    # Color coding
                    source_color = "#00ffcc" if "CV:" in src else "#0077ff"
                    source_icon = "📄" if "CV:" in src else "🔗"
                    
                    st.markdown(f"""
<div class="source-container">
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
        <div style="display: flex; align-items: center; gap: 0.8rem;">
            <span style="background: {source_color}; color: #0f0f0f; padding: 0.3rem 0.8rem; border-radius: 15px; font-weight: 600; font-size: 0.85rem; min-width: 40px; text-align: center; font-family: 'JetBrains Mono', monospace;">
                [{i}]
            </span>
            <div>
                <div style="font-weight: 600; color: {source_color}; font-size: 0.95rem; font-family: 'JetBrains Mono', monospace;">
                    {source_icon} {src.replace(':', ' • ')}
                </div>
                {f'<div style="color: #888; font-size: 0.8rem; margin-top: 0.2rem; font-family: JetBrains Mono, monospace;">{sec}</div>' if sec else ''}
            </div>
        </div>
        <div style="background: rgba(0, 255, 204, 0.2); padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; color: #00ffcc; font-weight: 500; font-family: 'JetBrains Mono', monospace;">
            {len(txt.split())} words
        </div>
    </div>
    <div style="background: #0f172a; padding: 0.8rem; border-radius: 8px; border-left: 3px solid {source_color};">
        <div style="color: #e0e0e0; line-height: 1.5; font-size: 0.9rem; font-family: 'JetBrains Mono', monospace;">
            <span style="color: #00aaff;">"</span>{display_text}<span style="color: #00aaff;">"</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Terminal footer
    st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(0, 255, 204, 0.1); border-radius: 10px; border: 1px solid #00aaff;">
    <p style="color: #00ffcc; font-size: 0.9rem; margin: 0; font-family: 'JetBrains Mono', monospace;">
        💡 All responses generated from indexed knowledge base with neural citations
    </p>
    <p style="color: #888; font-size: 0.8rem; margin: 0.5rem 0 0 0; font-family: 'JetBrains Mono', monospace;">
        System status: <span style="color: #00ffcc;">ONLINE</span> | Neural network: <span style="color: #00ffcc;">ACTIVE</span> | Citations: <span style="color: #00ffcc;">ENABLED</span>
    </p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()