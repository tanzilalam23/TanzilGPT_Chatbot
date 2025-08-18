import os, json, yaml, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import streamlit as st

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
    
    # Download model if missing
    if not os.path.exists(local_model_path):
        st.info("🔄 Model not found locally. Downloading from Hugging Face...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=CFG["llm"]["gguf_repo"],
                filename=CFG["llm"]["gguf_file"],
                local_dir="models",
                local_dir_use_symlinks=False
            )
            # Update path if download location differs
            local_model_path = downloaded_path if os.path.exists(downloaded_path) else local_model_path
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()
    
    # Load LLM
    try:
        return Llama(
            model_path=local_model_path,
            n_ctx=CFG["llm"].get("n_ctx", 4096),
            n_threads=CFG["llm"].get("n_threads", 4),
            n_gpu_layers=0,  # Set to 0 for CPU-only
            verbose=False
        )
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# ---------- Load embeddings & FAISS index ----------
@st.cache_resource
def load_retrieval_components():
    try:
        # Load embedder
        embedder = SentenceTransformer(CFG["embeddings"]["model_name"])
        
        # Check if index exists
        index_path = "index/faiss.index"
        chunks_path = "index/chunks.jsonl"
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            st.error("❌ Index files not found! Please run prep_index.py first.")
            st.info("Run: `python prep_index.py` to create the knowledge base.")
            st.stop()
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load chunks
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunks.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
        
        return embedder, index, chunks
    
    except Exception as e:
        st.error(f"❌ Failed to load retrieval components: {e}")
        st.stop()

# Initialize components
llm = load_model()
embedder, index, CHUNKS = load_retrieval_components()

def _embed(texts):
    """Embed texts and return normalized vectors"""
    try:
        vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs).astype("float32")
    except Exception as e:
        st.error(f"❌ Embedding error: {e}")
        return np.array([]).astype("float32")

def retrieve(query, k):
    """Retrieve top-k relevant chunks with deduplication"""
    try:
        qv = _embed([query])
        if qv.size == 0:
            return []
        
        # Search with extra results for deduplication
        search_k = min(k * 3, len(CHUNKS))
        D, I = index.search(qv, search_k)
        
        results = []
        seen_texts = set()
        
        for idx in I[0]:
            if idx == -1 or idx >= len(CHUNKS):
                continue
            
            chunk = CHUNKS[idx]
            chunk_text = chunk.get("text", "")
            
            # Skip duplicates and empty chunks
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
6. When answering about programming languages, distinguish between:
   - Programming Languages (Python, Java, C++, JavaScript, etc.)
   - Frameworks/Libraries (FastAPI, PyTorch, React, etc.)
   - Tools/Platforms (AWS, Docker, Git, etc.)
   - Don't mix these categories in your response

Keep responses focused, accurate, and well-categorized."""

def check_guardrails(text):
    """Check if input contains blocked phrases"""
    text_lower = text.lower()
    blocked = CFG.get("guardrails", {}).get("blocked_phrases", [])
    
    for phrase in blocked:
        if phrase.lower() in text_lower:
            return False
    return True

def preprocess_question(question):
    """Add context to certain types of questions for better responses"""
    question_lower = question.lower()
    
    if "programming language" in question_lower or "languages do you" in question_lower:
        return f"{question}\n\nNote: Please distinguish between programming languages (like Python, Java, C++) and frameworks/tools/libraries in your response."
    
    elif "skill" in question_lower and ("technical" in question_lower or "key" in question_lower):
        return f"{question}\n\nNote: Please organize the response by categories like Programming Languages, Frameworks/Libraries, Tools/Platforms, etc."
    
    return question

def generate_answer(question, k=5):
    """Generate answer using RAG pipeline"""
    
    # Check guardrails
    if not check_guardrails(question):
        return "I can only answer questions about Mohammad Tanzil Alam's professional background and experience.", []
    
    # Preprocess question for better responses
    processed_question = preprocess_question(question)
    
    # Retrieve relevant chunks
    contexts = retrieve(processed_question, k)
    
    if not contexts:
        return "I don't have relevant information to answer that question.", []
    
    # Build context safely within token limits
    try:
        context_window = llm.n_ctx()
        max_output_tokens = CFG["llm"].get("max_tokens", 512)
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
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
        
        # Create user message
        user_message = f"""Question: {processed_question}

CONTEXT (use ONLY this information to answer):
{context_block}

Provide a concise answer with citations [n] based solely on the context above."""

        # Generate response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        response = llm.create_chat_completion(
            messages=messages,
            temperature=CFG["llm"]["temperature"],
            top_p=CFG["llm"]["top_p"],
            max_tokens=max_output_tokens,
            stop=["\n\nUser:", "\n\nHuman:"]  # Prevent model from continuing conversation
        )
        
        answer = response["choices"][0]["message"]["content"].strip()
        return answer, contexts
        
    except Exception as e:
        st.error(f"❌ Generation error: {e}")
        return "Sorry, I encountered an error while processing your question.", []

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(
        page_title=f"Ask {CFG['persona']['name']}",
        page_icon="👤",
        layout="wide"
    )
    
    st.title(f"👤 Ask {CFG['persona']['name']} — AI CV Assistant")
    st.markdown("Get answers about my professional background, projects, and skills with cited sources.")
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("### About")
        st.info(f"This AI assistant answers questions about {CFG['persona']['name']}'s CV and projects using retrieval-augmented generation (RAG).")
        
        st.markdown("### Stats")
        st.metric("Knowledge Base", f"{len(CHUNKS)} chunks")
        st.metric("Model", CFG["llm"]["gguf_repo"].split("/")[-1])
    
    # Example questions
    EXAMPLES = [
        "What programming languages are you proficient in?",
        "Tell me about your most significant project.",
        "What frameworks and technologies do you work with?",
        "Do you have experience with machine learning and AI?",
        "How can I contact you?",
    ]
    
    st.markdown("### Quick Questions")
    cols = st.columns(len(EXAMPLES))
    selected_example = None
    
    for i, example in enumerate(EXAMPLES):
        if cols[i].button(example, key=f"example_{i}"):
            selected_example = example
    
    # Main input
    user_input = st.text_input(
        "Ask about experience, projects, skills, or contact info...",
        value=selected_example if selected_example else "",
        key="user_question"
    )
    
    # Generate response
    if user_input and user_input.strip():
        with st.spinner("🔍 Searching knowledge base..."):
            answer, contexts = generate_answer(user_input.strip(), k=5)
        
        # Display response
        st.markdown("---")
        st.markdown(f"**Question:** {user_input}")
        st.markdown(f"**Answer:** {answer}")
        
        # Show sources if available
        if contexts:
            with st.expander("📚 Sources", expanded=False):
                for i, ctx in enumerate(contexts, 1):
                    source = ctx.get('source', 'Unknown')
                    section = ctx.get('section', '')
                    text = ctx.get('text', '')[:200] + "..." if len(ctx.get('text', '')) > 200 else ctx.get('text', '')
                    
                    st.markdown(f"**[{i}]** {source} — {section}")
                    st.markdown(f"_{text}_")
                    st.markdown("")

if __name__ == "__main__":
    main()