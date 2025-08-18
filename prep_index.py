import os
import re
import nbformat
import json
import faiss
import numpy as np
import yaml
from pathlib import Path
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from git import Repo, InvalidGitRepositoryError
import fitz
import hashlib
from typing import List, Tuple, Dict

# Configuration
CONTENT_DIR = "content"
REPO_DIR = "repos"
INDEX_DIR = "index"
CONFIG_FILE = "config.yaml"

def load_config():
    """Load configuration from YAML file"""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ {CONFIG_FILE} not found!")
        return None

def ensure_directories():
    """Create necessary directories"""
    for directory in [REPO_DIR, INDEX_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Ensured directory exists: {directory}")

def extract_github_urls(content_dir: str) -> List[str]:
    """Extract GitHub URLs from markdown files in content directory"""
    repo_urls = set()  # Use set to avoid duplicates
    
    if not os.path.exists(content_dir):
        print(f"⚠️ Content directory '{content_dir}' not found!")
        return []
    
    for md_file in Path(content_dir).glob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Find GitHub URLs
                urls = re.findall(r"https://github\.com/[^\s)]+", content)
                repo_urls.update(urls)
                print(f"📄 Found {len(urls)} GitHub URLs in {md_file.name}")
        except Exception as e:
            print(f"⚠️ Error reading {md_file}: {e}")
    
    return list(repo_urls)

def fetch_repository(url: str) -> Path:
    """Clone or update a GitHub repository"""
    try:
        # Create a safe directory name from URL
        repo_name = url.replace("https://github.com/", "").replace("/", "-")
        local_path = Path(REPO_DIR) / repo_name
        
        if local_path.exists():
            try:
                # Try to update existing repo
                repo = Repo(local_path)
                if repo.remotes:
                    repo.remotes.origin.pull()
                    print(f"🔄 Updated {url}")
                else:
                    print(f"⚠️ No remote found for {local_path}")
            except (InvalidGitRepositoryError, Exception) as e:
                # If update fails, remove and re-clone
                print(f"⚠️ Update failed ({e}), re-cloning...")
                import shutil
                shutil.rmtree(local_path)
                repo = Repo.clone_from(url, local_path, depth=1)
                print(f"📦 Cloned {url}")
        else:
            # Clone fresh repo
            repo = Repo.clone_from(url, local_path, depth=1)
            print(f"📦 Cloned {url}")
        
        return local_path
    
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None

def extract_text_from_file(filepath: Path) -> str:
    """Extract text from various file formats"""
    try:
        if filepath.suffix.lower() in [".md", ".py", ".txt", ".js", ".css", ".yml", ".yaml", ".json"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        
        elif filepath.suffix.lower() == ".ipynb":
            nb = nbformat.read(filepath, as_version=4)
            text_parts = []
            for cell in nb.cells:
                if cell.cell_type in ("markdown", "code") and cell.source.strip():
                    text_parts.append(f"# {cell.cell_type.upper()} CELL\n{cell.source}\n")
            return "\n".join(text_parts)
        
        elif filepath.suffix.lower() in [".html", ".htm"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text(separator="\n", strip=True)
        
        elif filepath.suffix.lower() == ".pdf":
            with fitz.open(filepath) as pdf:
                text_parts = []
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    text_parts.append(page.get_text())
                return "\n".join(text_parts)
        
        return ""
    
    except Exception as e:
        print(f"⚠️ Error extracting text from {filepath}: {e}")
        return ""

def extract_texts_from_directory(directory_path: Path, source_name: str) -> List[Tuple[str, str]]:
    """Extract texts from all files in a directory"""
    texts = []
    valid_extensions = {".md", ".py", ".ipynb", ".html", ".htm", ".pdf", ".txt", ".js", ".css", ".yml", ".yaml", ".json"}
    skip_dirs = {".git", "venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "build"}
    
    for root, dirs, files in os.walk(directory_path):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for filename in files:
            filepath = Path(root) / filename
            
            # Skip files that are too large (>5MB)
            if filepath.stat().st_size > 5 * 1024 * 1024:
                continue
            
            if filepath.suffix.lower() in valid_extensions:
                text = extract_text_from_file(filepath)
                if text.strip() and len(text.strip()) > 50:  # Only include substantial content
                    relative_path = str(filepath.relative_to(directory_path))
                    texts.append((text, f"{source_name}:{relative_path}"))
    
    return texts

def collect_all_documents(content_dir: str, repo_urls: List[str]) -> List[Dict[str, str]]:
    """Collect all documents from CV files and repositories"""
    documents = []
    
    # 1. Process CV markdown files
    if os.path.exists(content_dir):
        for md_file in Path(content_dir).glob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        documents.append({
                            "text": content,
                            "source": f"CV:{md_file.name}",
                            "section": md_file.stem
                        })
                        print(f"📄 Added CV file: {md_file.name}")
            except Exception as e:
                print(f"⚠️ Error reading {md_file}: {e}")
    
    # 2. Process GitHub repositories
    for url in repo_urls:
        repo_path = fetch_repository(url)
        if repo_path and repo_path.exists():
            repo_name = url.split("/")[-1]
            repo_texts = extract_texts_from_directory(repo_path, f"Repo:{repo_name}")
            
            for text, source in repo_texts:
                documents.append({
                    "text": text,
                    "source": source,
                    "section": repo_name
                })
            
            print(f"📦 Added {len(repo_texts)} documents from {repo_name}")
    
    print(f"✅ Total collected documents: {len(documents)}")
    return documents

def smart_chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Intelligently chunk text by sentences and paragraphs"""
    # First try to split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        words = paragraph.split()
        
        # If paragraph itself is too long, split by sentences
        if len(words) > chunk_size:
            sentences = [s.strip() for s in re.split(r'[.!?]+', paragraph) if s.strip()]
            
            for sentence in sentences:
                sentence_words = sentence.split()
                
                if current_size + len(sentence_words) > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.split()) > 10:  # Only save substantial chunks
                        chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_words + sentence_words
                    current_size = len(current_chunk)
                else:
                    current_chunk.extend(sentence_words)
                    current_size += len(sentence_words)
        else:
            # Paragraph fits in chunk size
            if current_size + len(words) > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) > 10:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words + words
                current_size = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_size += len(words)
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text.split()) > 10:
            chunks.append(chunk_text)
    
    return chunks

def create_knowledge_base(documents: List[Dict[str, str]], config: dict):
    """Create embeddings and FAISS index from documents"""
    print("🔄 Creating knowledge base...")
    
    # Chunk all documents
    all_chunks = []
    chunk_metadata = []
    
    for doc in documents:
        doc_chunks = smart_chunk_text(doc["text"])
        
        for chunk in doc_chunks:
            if len(chunk.split()) < 5:  # Skip very short chunks
                continue
            
            all_chunks.append(chunk)
            chunk_metadata.append({
                "text": chunk,
                "source": doc["source"],
                "section": doc["section"],
                "hash": hashlib.md5(chunk.encode()).hexdigest()[:8]  # For deduplication
            })
    
    print(f"✂️ Created {len(all_chunks)} text chunks")
    
    if not all_chunks:
        print("❌ No chunks created! Check your content.")
        return False
    
    # Initialize embedder
    try:
        model_name = config["embeddings"]["model_name"]
        print(f"🧠 Loading embedding model: {model_name}")
        embedder = SentenceTransformer(model_name)
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        return False
    
    # Create embeddings
    try:
        print("🔄 Creating embeddings...")
        embeddings = embedder.encode(
            all_chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print(f"✅ Created embeddings: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Failed to create embeddings: {e}")
        return False
    
    # Create and save FAISS index
    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        
        index_path = os.path.join(INDEX_DIR, "faiss.index")
        faiss.write_index(index, index_path)
        print(f"💾 Saved FAISS index: {index_path}")
    except Exception as e:
        print(f"❌ Failed to create FAISS index: {e}")
        return False
    
    # Save chunk metadata
    try:
        chunks_path = os.path.join(INDEX_DIR, "chunks.jsonl")
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk_meta in chunk_metadata:
                f.write(json.dumps(chunk_meta, ensure_ascii=False) + "\n")
        print(f"💾 Saved chunk metadata: {chunks_path}")
    except Exception as e:
        print(f"❌ Failed to save chunks: {e}")
        return False
    
    return True

def main():
    """Main indexing pipeline"""
    print("🚀 Starting knowledge base creation...")
    
    # Load configuration
    config = load_config()
    if not config:
        print("❌ Cannot proceed without configuration file")
        return False
    
    # Ensure directories exist
    ensure_directories()
    
    # Extract GitHub URLs
    repo_urls = extract_github_urls(CONTENT_DIR)
    print(f"🔗 Found {len(repo_urls)} GitHub repositories")
    
    if not repo_urls and not os.path.exists(CONTENT_DIR):
        print("❌ No content found! Please ensure you have either:")
        print("   1. Markdown files in 'content/' directory")
        print("   2. GitHub URLs in your markdown files")
        return False
    
    # Collect all documents
    documents = collect_all_documents(CONTENT_DIR, repo_urls)
    
    if not documents:
        print("❌ No documents collected! Check your content sources.")
        return False
    
    # Create knowledge base
    success = create_knowledge_base(documents, config)
    
    if success:
        print("\n🎉 Knowledge base creation complete!")
        print(f"📊 Statistics:")
        print(f"   • Documents processed: {len(documents)}")
        print(f"   • GitHub repositories: {len(repo_urls)}")
        print(f"   • Index saved to: {INDEX_DIR}/")
        print("\n✅ Your AI CV chatbot is ready! Run 'streamlit run app.py' to start.")
    else:
        print("❌ Knowledge base creation failed!")
    
    return success

if __name__ == "__main__":
    main()