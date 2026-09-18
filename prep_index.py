import os
import re
import json
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import faiss
import fitz
import nbformat
import numpy as np
import yaml

from bs4 import BeautifulSoup
from git import Repo
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CONTENT_DIR = "content"
REPO_DIR = "repos"
INDEX_DIR = "index"
CONFIG_FILE = "config.yaml"

PRIMARY_CV = "01_cv_public.md"

# Repos that must never be ingested as "GitHub documentation" evidence.
# This app's own repo is the classic footgun: the CV links to it under
# Projects, so extract_github_urls() would otherwise clone it and index
# whatever (possibly stale) copy of the CV lives inside it, creating a
# second, conflicting source of "truth" for the same facts.
EXCLUDED_REPO_URLS = {
    "https://github.com/tanzilalam23/TanzilGPT",
}


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config() -> Optional[dict]:
    """Load YAML configuration."""

    try:
        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            return yaml.safe_load(f)

    except FileNotFoundError:
        print(f"❌ {CONFIG_FILE} not found.")
        return None

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return None


# ============================================================
# DIRECTORIES
# ============================================================

def ensure_directories() -> None:
    """Create required directories."""

    for directory in [
        REPO_DIR,
        INDEX_DIR
    ]:
        os.makedirs(
            directory,
            exist_ok=True
        )

        print(
            f"📁 Ensured directory exists: {directory}"
        )


# ============================================================
# GITHUB URL EXTRACTION
# ============================================================

def extract_github_urls(
    content_dir: str
) -> List[str]:
    """
    Find GitHub repository URLs inside markdown files.

    URLs are normalized and deduplicated.
    """

    repo_urls = set()

    content_path = Path(
        content_dir
    )

    if not content_path.exists():
        print(
            f"⚠️ Content directory '{content_dir}' not found."
        )
        return []

    pattern = re.compile(
        r"https://github\.com/"
        r"[A-Za-z0-9_.-]+/"
        r"[A-Za-z0-9_.-]+"
    )

    for md_file in content_path.glob(
        "*.md"
    ):

        try:

            content = md_file.read_text(
                encoding="utf-8"
            )

            urls = pattern.findall(
                content
            )

            for url in urls:

                clean_url = url.rstrip(
                    ".,);]"
                )

                repo_urls.add(
                    clean_url
                )

            print(
                f"📄 Found {len(urls)} GitHub URL(s) "
                f"in {md_file.name}"
            )

        except Exception as e:

            print(
                f"⚠️ Error reading {md_file}: {e}"
            )

    # Drop this app's own repo (and any other explicitly excluded repo).
    # See EXCLUDED_REPO_URLS for why.
    before = len(repo_urls)

    repo_urls = {
        url
        for url in repo_urls
        if url not in EXCLUDED_REPO_URLS
    }

    excluded_count = before - len(repo_urls)

    if excluded_count:

        print(
            f"🚫 Excluded {excluded_count} repo(s) from ingestion "
            f"(self-repo / explicit exclusion list): "
            f"{sorted(EXCLUDED_REPO_URLS)}"
        )

    return sorted(
        repo_urls
    )


# ============================================================
# GITHUB FETCH
# ============================================================

def fetch_repository(
    url: str
) -> Optional[Path]:
    """
    Clone or update a GitHub repository.
    """

    try:

        repo_name = (
            url.rstrip("/")
            .replace(
                "https://github.com/",
                ""
            )
            .replace(
                "/",
                "-"
            )
        )

        local_path = (
            Path(REPO_DIR)
            / repo_name
        )

        # ----------------------------------------------------
        # Existing repository
        # ----------------------------------------------------

        if local_path.exists():

            try:

                repo = Repo(
                    local_path
                )

                if repo.remotes:

                    repo.remotes.origin.pull()

                    print(
                        f"🔄 Updated {url}"
                    )

                else:

                    print(
                        f"⚠️ No Git remote found for "
                        f"{local_path}"
                    )

            except Exception as e:

                print(
                    f"⚠️ Update failed ({e}). "
                    f"Re-cloning..."
                )

                shutil.rmtree(
                    local_path,
                    ignore_errors=True
                )

                Repo.clone_from(
                    url,
                    local_path,
                    depth=1
                )

                print(
                    f"📦 Re-cloned {url}"
                )

        # ----------------------------------------------------
        # New repository
        # ----------------------------------------------------

        else:

            Repo.clone_from(
                url,
                local_path,
                depth=1
            )

            print(
                f"📦 Cloned {url}"
            )

        return local_path

    except Exception as e:

        print(
            f"❌ Failed to fetch {url}: {e}"
        )

        return None


# ============================================================
# FILE TEXT EXTRACTION
# ============================================================

def extract_text_from_file(
    filepath: Path
) -> str:
    """
    Extract useful text from supported file formats.
    """

    try:

        suffix = filepath.suffix.lower()

        # ----------------------------------------------------
        # Markdown / text / code
        # ----------------------------------------------------

        if suffix in {
            ".md",
            ".py",
            ".txt",
            ".js",
            ".css",
            ".yml",
            ".yaml",
            ".json",
            ".sh",
            ".sql",
            ".r"
        }:

            return filepath.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        # ----------------------------------------------------
        # HTML
        # ----------------------------------------------------

        if suffix in {
            ".html",
            ".htm"
        }:

            content = filepath.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            soup = BeautifulSoup(
                content,
                "html.parser"
            )

            return soup.get_text(
                separator="\n",
                strip=True
            )

        # ----------------------------------------------------
        # Jupyter notebook
        # ----------------------------------------------------

        if suffix == ".ipynb":

            notebook = nbformat.read(
                filepath,
                as_version=4
            )

            parts = []

            for cell in notebook.cells:

                if (
                    cell.cell_type == "markdown"
                    and cell.source.strip()
                ):

                    parts.append(
                        cell.source.strip()
                    )

                elif (
                    cell.cell_type == "code"
                    and cell.source.strip()
                ):

                    parts.append(
                        "Python code:\n"
                        + cell.source.strip()
                    )

            return "\n\n".join(
                parts
            )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if suffix == ".pdf":

            pages = []

            with fitz.open(
                filepath
            ) as pdf:

                for page in pdf:

                    page_text = page.get_text()

                    if page_text.strip():

                        pages.append(
                            page_text
                        )

            return "\n".join(
                pages
            )

        return ""

    except Exception as e:

        print(
            f"⚠️ Error extracting "
            f"{filepath}: {e}"
        )

        return ""


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(
    text: str
) -> str:
    """
    Normalize extracted text while preserving useful structure.
    """

    if not text:
        return ""

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Normalize tabs/spaces, but preserve line breaks.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Normalize excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Remove accidental long repeated characters.
    text = re.sub(
        r"(.)\1{20,}",
        "",
        text
    )

    # Remove very large code blocks.
    text = re.sub(
        r"```[\s\S]{800,}?```",
        "[Large code block omitted]",
        text
    )

    return text.strip()


# ============================================================
# CATEGORY
# ============================================================

def infer_category(
    heading_path: List[str]
) -> str:
    """
    Infer a broad semantic category from the heading hierarchy.

    Callers should pass only the heading(s) that actually carry
    category meaning (see category_heading in flush_section),
    not the full document path. Matching a job-title keyword like
    "data engineer" against a full joined path is unreliable: a
    heading such as "Education > MSc. Data Engineering" contains
    that substring and would otherwise be misclassified as
    "experience" purely because a degree name happens to contain
    a job-title keyword.
    """

    top_level = " ".join(
        heading_path
    ).lower()

    if any(
        term in top_level
        for term in [
            "education",
            "degree"
        ]
    ):
        return "education"

    if any(
        term in top_level
        for term in [
            "work experience",
            "experience",
            "employment",
            "associate consultant",
            "data engineer",
            "software engineer",
            "technical mentor"
        ]
    ):
        return "experience"

    if any(
        term in top_level
        for term in [
            "skill",
            "technical skill",
            "programming language",
            "cloud",
            "machine learning",
            "devops"
        ]
    ):
        return "skills"

    if any(
        term in top_level
        for term in [
            "project",
            "projects"
        ]
    ):
        return "projects"

    if any(
        term in top_level
        for term in [
            "award",
            "certification"
        ]
    ):
        return "achievements"

    if any(
        term in top_level
        for term in [
            "language",
            "languages"
        ]
    ):
        return "languages"

    return "general"


# ============================================================
# MARKDOWN SECTION PARSER
# ============================================================

def parse_markdown_sections(
    text: str,
    source_name: str,
    source_type: str = "document"
) -> List[Dict]:
    """
    Parse Markdown while preserving heading hierarchy.

    Example:

        ## Data Engineer | Roche
        ### Project 1 — Clinical Data Drift Detection

    becomes metadata containing both headings.

    This is especially important for Roche because the project
    chunks need to retain their employer/role context.
    """

    lines = text.splitlines()

    sections = []

    heading_stack: List[Tuple[int, str]] = []

    current_lines = []

    def flush_section():

        if not current_lines:
            return

        content = "\n".join(
            current_lines
        ).strip()

        if not content:
            return

        heading_path = [
            title
            for _, title in heading_stack
        ]

        if not heading_path:
            heading_path = [
                Path(source_name).stem
            ]

        # Category is decided from the section-level heading, not
        # the document's own top-level title and not the full
        # path. If the stack has a level-1 document title wrapping
        # everything (e.g. "# Master CV" > "## Experience" > "###
        # Associate Consultant"), the section-defining heading is
        # the second distinct level present ("Experience"), not
        # level 1 and not the deepest subsection — a subsection
        # title can share a keyword with an unrelated category
        # (e.g. a degree called "MSc. Data Engineering" under
        # "Education" would otherwise look like "experience").
        # If there is no title wrapper, the shallowest level
        # present is used directly.
        if heading_stack:

            levels_present = sorted(
                {
                    level
                    for level, _ in heading_stack
                }
            )

            target_level = (
                levels_present[1]
                if len(levels_present) > 1
                else levels_present[0]
            )

            category_headings = [
                title
                for level, title in heading_stack
                if level == target_level
            ]

        else:

            category_headings = heading_path

        sections.append(
            {
                "text": content,
                "source": source_name,
                "source_type": source_type,
                "heading_path": heading_path,
                "section": " > ".join(
                    heading_path
                ),
                "category": infer_category(
                    category_headings
                )
            }
        )

        current_lines.clear()

    for line in lines:

        heading_match = re.match(
            r"^(#{1,6})\s+(.+?)\s*$",
            line
        )

        if heading_match:

            flush_section()

            level = len(
                heading_match.group(1)
            )

            title = heading_match.group(2).strip()

            while (
                heading_stack
                and heading_stack[-1][0] >= level
            ):
                heading_stack.pop()

            heading_stack.append(
                (
                    level,
                    title
                )
            )

        else:

            current_lines.append(
                line
            )

    flush_section()

    return sections


# ============================================================
# DOCUMENT COLLECTION
# ============================================================

def collect_cv_documents(
    content_dir: str
) -> List[Dict]:
    """
    Collect Markdown knowledge-base documents.

    The primary CV receives highest authority.
    """

    documents = []

    content_path = Path(
        content_dir
    )

    if not content_path.exists():
        return documents

    markdown_files = sorted(
        content_path.glob(
            "*.md"
        ),
        key=lambda p: (
            0 if p.name == PRIMARY_CV else 1,
            p.name.lower()
        )
    )

    for md_file in markdown_files:

        try:

            content = md_file.read_text(
                encoding="utf-8"
            )

            content = clean_text(
                content
            )

            if not content:
                continue

            source_type = (
                "primary_cv"
                if md_file.name == PRIMARY_CV
                else "profile_document"
            )

            sections = parse_markdown_sections(
                content,
                f"CV:{md_file.name}",
                source_type
            )

            for section in sections:

                if len(
                    section["text"].split()
                ) < 8:

                    continue

                documents.append(
                    section
                )

            print(
                f"📄 Added {md_file.name} "
                f"({len(sections)} sections)"
            )

        except Exception as e:

            print(
                f"⚠️ Error reading {md_file}: {e}"
            )

    return documents


# ============================================================
# GITHUB DOCUMENT COLLECTION
# ============================================================

def collect_github_documents(
    repo_path: Path,
    repo_name: str
) -> List[Dict]:
    """
    Collect useful GitHub documentation.

    Deliberately avoids indexing every source-code file.
    This prevents repository implementation details from
    overwhelming the authoritative CV.

    Priority:
        README / project documentation
        docs / documentation
        selected notebooks

    """

    documents = []

    # --------------------------------------------------------
    # Files we explicitly want
    # --------------------------------------------------------

    preferred_exact_names = {
        "readme.md",
        "readme",
        "project.md",
        "about.md",
        "documentation.md"
    }

    preferred_extensions = {
        ".md",
        ".ipynb"
    }

    skip_dirs = {
        ".git",
        "venv",
        ".venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",
        ".idea",
        ".vscode",
        "site-packages"
    }

    discovered = []

    for root, dirs, files in os.walk(
        repo_path
    ):

        dirs[:] = [
            directory
            for directory in dirs
            if directory not in skip_dirs
            and not directory.startswith(".")
        ]

        for filename in files:

            filepath = Path(
                root
            ) / filename

            lower_name = filename.lower()

            # ------------------------------------------------
            # Exact documentation files
            # ------------------------------------------------

            # Never ingest a copy of the primary CV file itself,
            # wherever it happens to live inside a cloned repo.
            # A stale/duplicate CV committed in a project repo must
            # not become a second, conflicting "source of truth".
            if lower_name == PRIMARY_CV.lower():

                continue

            if lower_name in preferred_exact_names:

                priority = 0

            # ------------------------------------------------
            # Documentation directory
            # ------------------------------------------------

            elif (
                "docs" in filepath.parts
                or "documentation" in filepath.parts
            ) and filepath.suffix.lower() in {
                ".md",
                ".txt"
            }:

                priority = 1

            # ------------------------------------------------
            # Other Markdown
            # ------------------------------------------------

            elif filepath.suffix.lower() == ".md":

                priority = 2

            # ------------------------------------------------
            # Notebooks
            # ------------------------------------------------

            elif filepath.suffix.lower() == ".ipynb":

                priority = 3

            else:

                continue

            try:

                size = filepath.stat().st_size

                if size > 2 * 1024 * 1024:
                    continue

            except OSError:
                continue

            discovered.append(
                (
                    priority,
                    filepath
                )
            )

    discovered.sort(
        key=lambda item: (
            item[0],
            str(item[1]).lower()
        )
    )

    for _, filepath in discovered:

        text = extract_text_from_file(
            filepath
        )

        text = clean_text(
            text
        )

        if not text:
            continue

        if len(
            text.split()
        ) < 20:
            continue

        relative_path = str(
            filepath.relative_to(
                repo_path
            )
        )

        source = (
            f"Repo:{repo_name}:{relative_path}"
        )

        # ----------------------------------------------------
        # Markdown
        # ----------------------------------------------------

        if filepath.suffix.lower() == ".md":

            sections = parse_markdown_sections(
                text,
                source,
                "github_documentation"
            )

            for section in sections:

                if len(
                    section["text"].split()
                ) < 8:

                    continue

                section["repo"] = repo_name

                documents.append(
                    section
                )

        # ----------------------------------------------------
        # Notebook
        # ----------------------------------------------------

        else:

            documents.append(
                {
                    "text": text,
                    "source": source,
                    "source_type": "github_notebook",
                    "heading_path": [
                        repo_name
                    ],
                    "section": repo_name,
                    "category": "project",
                    "repo": repo_name
                }
            )

    return documents


# ============================================================
# COLLECT EVERYTHING
# ============================================================

def collect_all_documents(
    content_dir: str,
    repo_urls: List[str]
) -> List[Dict]:
    """
    Collect authoritative CV data and supporting GitHub
    documentation.
    """

    documents = []

    # --------------------------------------------------------
    # CV / profile documents
    # --------------------------------------------------------

    cv_documents = collect_cv_documents(
        content_dir
    )

    documents.extend(
        cv_documents
    )

    print(
        f"✅ Collected {len(cv_documents)} "
        f"CV/profile sections"
    )

    # --------------------------------------------------------
    # GitHub repositories
    # --------------------------------------------------------

    total_repo_docs = 0

    for url in repo_urls:

        repo_path = fetch_repository(
            url
        )

        if (
            not repo_path
            or not repo_path.exists()
        ):
            continue

        repo_name = (
            url.rstrip("/")
            .split("/")
            [-1]
        )

        repo_documents = collect_github_documents(
            repo_path,
            repo_name
        )

        documents.extend(
            repo_documents
        )

        total_repo_docs += len(
            repo_documents
        )

        print(
            f"📦 Added {len(repo_documents)} "
            f"documentation sections from {repo_name}"
        )

    print(
        f"✅ Total GitHub documentation sections: "
        f"{total_repo_docs}"
    )

    print(
        f"✅ Total collected sections: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# CHUNKING
# ============================================================

def split_long_text(
    text: str,
    max_words: int
) -> List[str]:
    """
    Split very long text into sentence-aware chunks.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    output = []

    current = []
    current_count = 0

    for sentence in sentences:

        words = sentence.split()

        if not words:
            continue

        # Normal sentence fits.
        if (
            current_count + len(words)
            <= max_words
        ):

            current.extend(
                words
            )

            current_count += len(
                words
            )

            continue

        # Save current section.
        if current:

            output.append(
                " ".join(
                    current
                )
            )

        # Extremely long sentence.
        if len(words) > max_words:

            for start in range(
                0,
                len(words),
                max_words
            ):

                output.append(
                    " ".join(
                        words[
                            start:
                            start + max_words
                        ]
                    )
                )

            current = []
            current_count = 0

        else:

            current = words
            current_count = len(
                words
            )

    if current:

        output.append(
            " ".join(
                current
            )
        )

    return output


def intelligent_chunk_text(
    text: str,
    config: dict
) -> List[str]:
    """
    Create chunks while preserving paragraph boundaries.

    Default:
        400 words
        80 words overlap
    """

    chunk_cfg = config.get(
        "chunking",
        {}
    )

    chunk_size = int(
        chunk_cfg.get(
            "chunk_size",
            400
        )
    )

    overlap = int(
        chunk_cfg.get(
            "overlap",
            80
        )
    )

    min_words = int(
        chunk_cfg.get(
            "min_chunk_words",
            15
        )
    )

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text
        )
        if paragraph.strip()
    ]

    normalized_paragraphs = []

    for paragraph in paragraphs:

        words = paragraph.split()

        if len(words) <= chunk_size:

            normalized_paragraphs.append(
                paragraph
            )

        else:

            normalized_paragraphs.extend(
                split_long_text(
                    paragraph,
                    chunk_size
                )
            )

    chunks = []

    current_words = []

    for paragraph in normalized_paragraphs:

        words = paragraph.split()

        if not words:
            continue

        # Add to current chunk.
        if (
            len(current_words)
            + len(words)
            <= chunk_size
        ):

            current_words.extend(
                words
            )

            continue

        # Save current chunk.
        if len(current_words) >= min_words:

            chunks.append(
                " ".join(
                    current_words
                )
            )

        # Carry overlap.
        overlap_words = (
            current_words[
                -overlap:
            ]
            if overlap > 0
            else []
        )

        current_words = (
            overlap_words
            + words
        )

        # Prevent runaway overlap.
        if len(current_words) > chunk_size:

            current_words = (
                current_words[
                    -chunk_size:
                ]
            )

    # Save final chunk.
    if len(current_words) >= min_words:

        chunks.append(
            " ".join(
                current_words
            )
        )

    return chunks


# ============================================================
# ADD METADATA TO CHUNKS
# ============================================================

def build_chunk_records(
    documents: List[Dict],
    config: dict
) -> List[Dict]:
    """
    Convert document sections into final RAG records.

    Every chunk gets its heading context prefixed so that a
    project chunk does not lose its employer/project identity.
    """

    records = []

    for document in documents:

        text = clean_text(
            document.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        heading_path = document.get(
            "heading_path",
            []
        )

        source = document.get(
            "source",
            "Unknown"
        )

        source_type = document.get(
            "source_type",
            "document"
        )

        category = document.get(
            "category",
            "general"
        )

        section = document.get(
            "section",
            ""
        )

        # ----------------------------------------------------
        # Context prefix
        # ----------------------------------------------------

        if heading_path:

            context_prefix = (
                "Document context: "
                + " > ".join(
                    heading_path
                )
                + "\n\n"
            )

        else:

            context_prefix = ""

        # Work-experience sections must never be split across
        # chunks. A single employer's dates, role, and bullet
        # points getting divided at an arbitrary word boundary
        # (or overlapping with the next employer) is exactly what
        # causes the model to blend or misattribute roles when it
        # only receives a partial section as "evidence". Since a
        # heading in the primary CV corresponds to one employer,
        # keeping category == "experience" whole guarantees the
        # model always sees a complete, unambiguous role.
        if category == "experience":

            chunks = [text]

        else:

            chunks = intelligent_chunk_text(
                text,
                config
            )

        for chunk in chunks:

            final_text = (
                context_prefix
                + chunk
            )

            if len(
                final_text.split()
            ) < 15:

                continue

            records.append(
                {
                    "text": final_text,
                    "source": source,
                    "source_type": source_type,
                    "section": section,
                    "category": category,
                    "heading_path": heading_path
                }
            )

    return records


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_chunks(
    chunks: List[Dict]
) -> List[Dict]:
    """
    Remove exact duplicate chunks.
    """

    unique = []
    seen = set()

    for chunk in chunks:

        normalized = re.sub(
            r"\s+",
            " ",
            clean_text(
                chunk["text"]
            ).lower()
        )

        digest = hashlib.md5(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()

        if digest in seen:
            continue

        seen.add(
            digest
        )

        chunk["hash"] = digest[:12]

        unique.append(
            chunk
        )

    print(
        f"🔄 Deduplicated: "
        f"{len(chunks)} → {len(unique)} chunks"
    )

    return unique


# ============================================================
# EMBEDDING
# ============================================================

def create_embeddings(
    chunks: List[Dict],
    config: dict
) -> Optional[np.ndarray]:
    """
    Create normalized embeddings.
    """

    model_name = config[
        "embeddings"
    ][
        "model_name"
    ]

    print(
        f"🧠 Loading embedding model: "
        f"{model_name}"
    )

    try:

        embedder = SentenceTransformer(
            model_name
        )

    except Exception as e:

        print(
            f"❌ Failed to load embedding model: {e}"
        )

        return None

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"🔄 Creating embeddings for "
        f"{len(texts)} chunks..."
    )

    try:

        batch_size = 64

        embeddings = []

        for start in range(
            0,
            len(texts),
            batch_size
        ):

            batch = texts[
                start:
                start + batch_size
            ]

            batch_embeddings = embedder.encode(
                batch,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            embeddings.append(
                batch_embeddings
            )

        result = np.vstack(
            embeddings
        ).astype(
            "float32"
        )

        print(
            f"✅ Created embeddings: "
            f"{result.shape}"
        )

        return result

    except Exception as e:

        print(
            f"❌ Failed to create embeddings: {e}"
        )

        return None


# ============================================================
# CREATE FAISS
# ============================================================

def create_faiss_index(
    embeddings: np.ndarray
) -> Optional[faiss.Index]:
    """
    Create an exact cosine-similarity index.

    Because embeddings are normalized, inner product equals
    cosine similarity.
    """

    try:

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            embeddings
        )

        print(
            f"✅ FAISS index created with "
            f"{index.ntotal} vectors"
        )

        return index

    except Exception as e:

        print(
            f"❌ Failed to create FAISS index: {e}"
        )

        return None


# ============================================================
# SAVE KNOWLEDGE BASE
# ============================================================

def save_knowledge_base(
    chunks: List[Dict],
    embeddings: np.ndarray,
    index: faiss.Index,
    config: dict
) -> bool:
    """
    Save FAISS index, chunk metadata and statistics.
    """

    try:

        os.makedirs(
            INDEX_DIR,
            exist_ok=True
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        index_path = (
            Path(INDEX_DIR)
            / "faiss.index"
        )

        faiss.write_index(
            index,
            str(index_path)
        )

        print(
            f"💾 Saved FAISS index: "
            f"{index_path}"
        )

        # ----------------------------------------------------
        # Chunk metadata
        # ----------------------------------------------------

        chunks_path = (
            Path(INDEX_DIR)
            / "chunks.jsonl"
        )

        with open(
            chunks_path,
            "w",
            encoding="utf-8"
        ) as f:

            for chunk in chunks:

                f.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        print(
            f"💾 Saved chunk metadata: "
            f"{chunks_path}"
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        source_counts = {}

        category_counts = {}

        source_type_counts = {}

        for chunk in chunks:

            source = chunk.get(
                "source",
                "Unknown"
            )

            category = chunk.get(
                "category",
                "general"
            )

            source_type = chunk.get(
                "source_type",
                "document"
            )

            source_counts[source] = (
                source_counts.get(
                    source,
                    0
                )
                + 1
            )

            category_counts[category] = (
                category_counts.get(
                    category,
                    0
                )
                + 1
            )

            source_type_counts[source_type] = (
                source_type_counts.get(
                    source_type,
                    0
                )
                + 1
            )

        stats = {
            "total_chunks": len(chunks),
            "embedding_dimension": int(
                embeddings.shape[1]
            ),
            "embedding_model": config[
                "embeddings"
            ][
                "model_name"
            ],
            "chunk_size": config.get(
                "chunking",
                {}
            ).get(
                "chunk_size",
                400
            ),
            "overlap": config.get(
                "chunking",
                {}
            ).get(
                "overlap",
                80
            ),
            "primary_cv": PRIMARY_CV,
            "source_counts": source_counts,
            "category_counts": category_counts,
            "source_type_counts": source_type_counts
        }

        stats_path = (
            Path(INDEX_DIR)
            / "stats.json"
        )

        with open(
            stats_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                stats,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"📊 Saved statistics: "
            f"{stats_path}"
        )

        return True

    except Exception as e:

        print(
            f"❌ Failed to save knowledge base: {e}"
        )

        return False


# ============================================================
# VALIDATE KNOWLEDGE BASE
# ============================================================

def validate_knowledge_base(
    chunks: List[Dict]
) -> None:
    """
    Print important validation information before completion.
    """

    primary_cv_chunks = [
        chunk
        for chunk in chunks
        if chunk.get(
            "source_type"
        ) == "primary_cv"
    ]

    roche_chunks = [
        chunk
        for chunk in chunks
        if "roche" in (
            chunk.get(
                "section",
                ""
            ).lower()
            + " "
            + chunk.get(
                "text",
                ""
            ).lower()
        )
    ]

    print(
        "\n🔎 Validation"
    )

    print(
        f"   Primary CV chunks: "
        f"{len(primary_cv_chunks)}"
    )

    print(
        f"   Roche-related chunks: "
        f"{len(roche_chunks)}"
    )

    # Print Roche headings so we can immediately verify that
    # both projects survived indexing.
    roche_sections = sorted(
        {
            chunk.get(
                "section",
                ""
            )
            for chunk in roche_chunks
            if chunk.get(
                "section",
                ""
            )
        }
    )

    if roche_sections:

        print(
            "   Roche sections:"
        )

        for section in roche_sections:

            print(
                f"      - {section}"
            )


# ============================================================
# CREATE KNOWLEDGE BASE
# ============================================================

def create_knowledge_base(
    documents: List[Dict],
    config: dict
) -> bool:
    """
    Build the complete RAG knowledge base.
    """

    print(
        "\n🔄 Building knowledge base..."
    )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    chunks = build_chunk_records(
        documents,
        config
    )

    print(
        f"✂️ Created "
        f"{len(chunks)} raw chunks"
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    chunks = deduplicate_chunks(
        chunks
    )

    if not chunks:

        print(
            "❌ No chunks created."
        )

        return False

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_knowledge_base(
        chunks
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        chunks,
        config
    )

    if embeddings is None:
        return False

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    index = create_faiss_index(
        embeddings
    )

    if index is None:
        return False

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    return save_knowledge_base(
        chunks,
        embeddings,
        index,
        config
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "🚀 Starting knowledge base creation..."
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = load_config()

    if not config:
        return False

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    ensure_directories()

    # --------------------------------------------------------
    # GitHub URLs
    # --------------------------------------------------------

    repo_urls = extract_github_urls(
        CONTENT_DIR
    )

    print(
        f"🔗 Found "
        f"{len(repo_urls)} GitHub repositories"
    )

    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------

    documents = collect_all_documents(
        CONTENT_DIR,
        repo_urls
    )

    if not documents:

        print(
            "❌ No documents collected."
        )

        return False

    # --------------------------------------------------------
    # Build index
    # --------------------------------------------------------

    success = create_knowledge_base(
        documents,
        config
    )

    if not success:

        print(
            "\n❌ Knowledge base creation failed."
        )

        return False

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print(
        "\n🎉 Knowledge base creation complete!"
    )

    print(
        f"📊 Document sections processed: "
        f"{len(documents)}"
    )

    print(
        f"📊 GitHub repositories: "
        f"{len(repo_urls)}"
    )

    print(
        f"📁 Index saved to: "
        f"{INDEX_DIR}/"
    )

    print(
        "\n✅ Your AI CV chatbot knowledge base is ready."
    )

    return True


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()