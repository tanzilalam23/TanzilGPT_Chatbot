#prep_index.py
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
import shutil

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
    repo_urls = set()
    
    if not os.path.exists(content_dir):
        print(f"⚠️ Content directory '{content_dir}' not found!")
        return []
    
    for md_file in Path(content_dir).glob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Find GitHub URLs with better regex
                urls = re.findall(r"https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", content)
                repo_urls.update(urls)
                print(f"📄 Found {len(urls)} GitHub URLs in {md_file.name}")
        except Exception as e:
            print(f"⚠️ Error reading {md_file}: {e}")
    
    return list(repo_urls)

def fetch_repository(url: str) -> Path:
    """Clone or update a GitHub repository"""
    try:
        repo_name = url.replace("https://github.com/", "").replace("/", "-")
        local_path = Path(REPO_DIR) / repo_name
        
        if local_path.exists():
            try:
                repo = Repo(local_path)
                if repo.remotes:
                    repo.remotes.origin.pull()
                    print(f"🔄 Updated {url}")
                else:
                    print(f"⚠️ No remote found for {local_path}")
            except (InvalidGitRepositoryError, Exception) as e:
                print(f"⚠️ Update failed ({e}), re-cloning...")
                shutil.rmtree(local_path)
                repo = Repo.clone_from(url, local_path, depth=1)
                print(f"📦 Cloned {url}")
        else:
            repo = Repo.clone_from(url, local_path, depth=1)
            print(f"📦 Cloned {url}")
        
        return local_path
    
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None

def extract_text_from_file(filepath: Path) -> str:
    """Extract text from various file formats with improved handling"""
    try:
        suffix = filepath.suffix.lower()
        
        if suffix in [".md", ".py", ".txt", ".js", ".css", ".yml", ".yaml", ".json", ".sh", ".sql", ".r"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        
        elif suffix == ".ipynb":
            nb = nbformat.read(filepath, as_version=4)
            text_parts = []
            for cell in nb.cells:
                if cell.cell_type == "markdown" and cell.source.strip():
                    text_parts.append(f"## Markdown Cell\n{cell.source}\n")
                elif cell.cell_type == "code" and cell.source.strip():
                    text_parts.append(f"## Code Cell\n```python\n{cell.source}\n```\n")
            return "\n".join(text_parts)
        
        elif suffix in [".html", ".htm"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text(separator="\n", strip=True)
        
        elif suffix == ".pdf":
            with fitz.open(filepath) as pdf:
                text_parts = []
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    text_parts.append(page.get_text())
                return "\n".join(text_parts)
        
        return ""
    
    except Exception as e:
        print(f"Warning: Error extracting text from {filepath}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove very long lines of repeated characters
    text = re.sub(r'(.)\1{20,}', '', text)
    
    # Remove code blocks that are too long (over 500 chars)
    text = re.sub(r'```[\s\S]{500,}?```', '[Long code block omitted]', text)
    
    return text.strip()

def extract_texts_from_directory(directory_path: Path, source_name: str) -> List[Tuple[str, str]]:
    """Extract texts from all files in a directory with improved filtering"""
    texts = []
    valid_extensions = {".md", ".py", ".ipynb", ".html", ".htm", ".pdf", ".txt", ".js", ".css", ".yml", ".yaml", ".json", ".sh", ".sql", ".r"}
    skip_dirs = {".git", "venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "build", "target", ".idea", ".vscode"}
    skip_files = {"package-lock.json", "yarn.lock", "requirements.txt"}
    
    for root, dirs, files in os.walk(directory_path):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        
        for filename in files:
            if filename in skip_files:
                continue
                
            filepath = Path(root) / filename
            
            # Skip files that are too large (>2MB)
            try:
                if filepath.stat().st_size > 2 * 1024 * 1024:
                    continue
            except OSError:
                continue
            
            if filepath.suffix.lower() in valid_extensions:
                text = extract_text_from_file(filepath)
                if text.strip() and len(text.split()) > 20:  # Require at least 20 words
                    cleaned_text = clean_text(text)
                    if cleaned_text:
                        relative_path = str(filepath.relative_to(directory_path))
                        texts.append((cleaned_text, f"{source_name}:{relative_path}"))
    
    return texts

def split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    """Split markdown into sections based on headers"""
    lines = text.split('\n')
    sections = []
    current_section = ""
    current_title = "Introduction"
    
    for line in lines:
        # Check for headers
        if re.match(r'^#{1,6}\s+', line):
            # Save previous section
            if current_section.strip():
                sections.append((current_title, current_section.strip()))
            
            # Start new section
            current_title = re.sub(r'^#{1,6}\s+', '', line).strip()
            current_section = ""
        else:
            current_section += line + '\n'
    
    # Add final section
    if current_section.strip():
        sections.append((current_title, current_section.strip()))
    
    return sections

def collect_all_documents(content_dir: str, repo_urls: List[str]) -> List[Dict[str, str]]:
    """Collect all documents from CV files and repositories"""
    documents = []
    
    # 1. Process CV markdown files with better section detection
    if os.path.exists(content_dir):
        for md_file in Path(content_dir).glob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        cleaned_content = clean_text(content)
                        
                        # Try to extract sections from markdown
                        sections = split_markdown_sections(cleaned_content)
                        
                        if sections:
                            for section_title, section_content in sections:
                                if len(section_content.split()) > 10:
                                    documents.append({
                                        "text": section_content,
                                        "source": f"CV:{md_file.name}",
                                        "section": section_title
                                    })
                        else:
                            documents.append({
                                "text": cleaned_content,
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

def intelligent_chunk_text(text: str, config: dict) -> List[str]:
    """Improved text chunking with better boundary detection"""
    chunk_size = config.get("chunking", {}).get("chunk_size", 400)
    overlap = config.get("chunking", {}).get("overlap", 80)
    min_words = config.get("chunking", {}).get("min_chunk_words", 15)
    
    # First split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for paragraph in paragraphs:
        words = paragraph.split()
        para_word_count = len(words)
        
        # If paragraph is too long, split by sentences
        if para_word_count > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                sentence_words = sentence.split()
                
                if current_word_count + len(sentence_words) > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.split()) >= min_words:
                        chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else []
                    current_chunk = overlap_words + sentence_words
                    current_word_count = len(current_chunk)
                else:
                    current_chunk.extend(sentence_words)
                    current_word_count += len(sentence_words)
        else:
            # Check if adding this paragraph exceeds chunk size
            if current_word_count + para_word_count > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= min_words:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else []
                current_chunk = overlap_words + words
                current_word_count = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_word_count += para_word_count
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text.split()) >= min_words:
            chunks.append(chunk_text)
    
    return chunks

def deduplicate_chunks(chunks_with_metadata: List[Dict]) -> List[Dict]:
    """Remove duplicate or very similar chunks"""
    unique_chunks = []
    seen_hashes = set()
    
    for chunk_data in chunks_with_metadata:
        text = chunk_data["text"]
        
        # Create hash of normalized text
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        chunk_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        if chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            chunk_data["hash"] = chunk_hash[:8]
            unique_chunks.append(chunk_data)
    
    print(f"🔄 Deduplicated: {len(chunks_with_metadata)} -> {len(unique_chunks)} chunks")
    return unique_chunks

def create_knowledge_base(documents: List[Dict[str, str]], config: dict):
    """Create embeddings and FAISS index with improved processing"""
    print("🔄 Creating knowledge base...")
    
    # Chunk all documents with improved chunking
    all_chunks_metadata = []
    
    for doc in documents:
        doc_chunks = intelligent_chunk_text(doc["text"], config)
        
        for chunk in doc_chunks:
            all_chunks_metadata.append({
                "text": chunk,
                "source": doc["source"],
                "section": doc["section"]
            })
    
    print(f"✂️ Created {len(all_chunks_metadata)} text chunks")
    
    # Deduplicate chunks
    all_chunks_metadata = deduplicate_chunks(all_chunks_metadata)
    
    if not all_chunks_metadata:
        print("❌ No chunks created! Check your content.")
        return False
    
    # Extract just the text for embedding
    all_chunks = [chunk["text"] for chunk in all_chunks_metadata]
    
    # Initialize embedder
    try:
        model_name = config["embeddings"]["model_name"]
        print(f"🧠 Loading embedding model: {model_name}")
        embedder = SentenceTransformer(model_name)
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        return False
    
    # Create embeddings in batches to avoid memory issues
    try:
        print("🔄 Creating embeddings...")
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_embeddings = embedder.encode(
                batch,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        print(f"✅ Created embeddings: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Failed to create embeddings: {e}")
        return False
    
    # Create and save FAISS index
    try:
        dim = embeddings.shape[1]
        # Use IndexHNSWFlat for better performance on larger datasets
        if len(embeddings) > 1000:
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 40
        else:
            index = faiss.IndexFlatIP(dim)  # Inner product for normalized embeddings
        
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
            for chunk_meta in all_chunks_metadata:
                f.write(json.dumps(chunk_meta, ensure_ascii=False) + "\n")
        print(f"💾 Saved chunk metadata: {chunks_path}")
    except Exception as e:
        print(f"❌ Failed to save chunks: {e}")
        return False
    
    # Save statistics
    try:
        stats = {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks_metadata),
            "embedding_dim": int(embeddings.shape[1]),
            "model_name": config["embeddings"]["model_name"],
            "chunk_size": config.get("chunking", {}).get("chunk_size", 400),
            "sources": list(set([chunk["source"] for chunk in all_chunks_metadata]))
        }
        
        stats_path = os.path.join(INDEX_DIR, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"📊 Saved statistics: {stats_path}")
    except Exception as e:
        print(f"⚠️ Failed to save stats: {e}")
    
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