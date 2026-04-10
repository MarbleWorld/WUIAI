import os
import re
import io
import time
import json
import math
import queue
import hashlib
import tempfile
import threading
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import pypdf
except Exception:
    pypdf = None


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Wildfire Knowledge Lab",
    page_icon="🔥",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    max-width: 100% !important;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
    padding-top: 1.1rem;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.hero-card {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(251,113,133,0.08));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 1.2rem 1.25rem 1.1rem 1.25rem;
    box-shadow: 0 10px 34px rgba(0,0,0,0.14);
    margin-bottom: 1rem;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.9rem 1rem;
}

.source-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.75rem;
}

.big-button button {
    background: linear-gradient(90deg, #ef4444, #fb7185) !important;
    color: white !important;
    border-radius: 14px !important;
    padding: 0.80rem 1.25rem !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    font-weight: 900 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.20) !important;
}
.big-button button:hover {
    border: 1px solid rgba(255,255,255,0.28) !important;
}

textarea, input, [data-baseweb="textarea"] textarea, [data-baseweb="input"] input {
    color: #111827 !important;
    background: #ffffff !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #111827 !important;
}

.small-muted {
    color: rgba(255,255,255,0.68);
    font-size: 0.92rem;
    line-height: 1.35;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "Wildfire Knowledge Lab"
APP_SUBTITLE = "Ingest wildfire websites, PDFs, and notes into a searchable, citation-grounded knowledge system."

CHROMA_DIR = os.getenv("WILDFIRE_CHROMA_DIR", "./wildfire_chroma")
COLLECTION_NAME = os.getenv("WILDFIRE_COLLECTION_NAME", "wildfire_knowledge")
EMBED_MODEL_NAME = os.getenv("WILDFIRE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY", "")

DEFAULT_SEEDS = [
    "https://www.nifc.gov/",
    "https://inciweb.wildfire.gov/",
    "https://www.nwcg.gov/",
    "https://www.predictiveservices.nifc.gov/",
    "https://www.fs.usda.gov/managing-land/fire",
    "https://www.fire.ca.gov/",
    "https://www.readyforwildfire.org/",
    "https://www.weather.gov/",
    "https://www.blm.gov/programs/public-safety-and-fire/fire",
    "https://www.nps.gov/subjects/fire/",
]

DEFAULT_ALLOWED_DOMAINS = [
    "nifc.gov",
    "inciweb.wildfire.gov",
    "nwcg.gov",
    "predictiveservices.nifc.gov",
    "fs.usda.gov",
    "fire.ca.gov",
    "readyforwildfire.org",
    "weather.gov",
    "blm.gov",
    "nps.gov",
]

REQUEST_TIMEOUT = 25
DEFAULT_REQUEST_DELAY = 0.5
DEFAULT_MAX_WORKERS = 6
DEFAULT_MAX_PAGES = 150
DEFAULT_MAX_DEPTH = 2
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
TOP_K = 6

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_hits" not in st.session_state:
    st.session_state.last_hits = []
if "ingest_log" not in st.session_state:
    st.session_state.ingest_log = []
if "latest_stats" not in st.session_state:
    st.session_state.latest_stats = {"pages": 0, "chunks": 0, "sources": 0}
if "latest_docs_preview" not in st.session_state:
    st.session_state.latest_docs_preview = []


# =========================================================
# HELPERS
# =========================================================
def utc_now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_url(url: str) -> str:
    url = urldefrag(url)[0].strip()
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = re.sub(r"/{2,}", "/", parsed.path)
    return parsed._replace(scheme=scheme, netloc=netloc, path=path, fragment="").geturl()


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def split_into_word_chunks(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size_words - overlap_words)
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def looks_like_html_url(url: str) -> bool:
    blocked = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".zip", ".rar",
        ".7z", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".tif",
        ".tiff", ".csv", ".geojson", ".shp", ".gz", ".xml", ".rss", ".atom"
    )
    return not url.lower().endswith(blocked)


def is_allowed_domain(url: str, allowed_domains: List[str]) -> bool:
    netloc = urlparse(url).netloc.lower()
    if not netloc:
        return False
    for dom in allowed_domains:
        dom = dom.lower().strip()
        if netloc == dom or netloc.endswith("." + dom):
            return True
    return False


def extract_links(base_url: str, html: str) -> List[str]:
    links = []
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        full = normalize_url(urljoin(base_url, href))
        if full.startswith("http://") or full.startswith("https://"):
            links.append(full)
    return list(dict.fromkeys(links))


def html_title(html: str, fallback: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        t = clean_text(soup.title.string)
        if t:
            return t
    h1 = soup.find("h1")
    if h1:
        t = clean_text(h1.get_text(" ", strip=True))
        if t:
            return t
    return fallback


def estimate_cost_text(tokens_in: int, tokens_out: int, model_name: str) -> str:
    price = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
    }
    if model_name not in price:
        return "est cost: n/a"
    pin, pout = price[model_name]
    usd = (tokens_in / 1_000_000.0) * pin + (tokens_out / 1_000_000.0) * pout
    return f"est cost: ${usd:.6f}"


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class PageRecord:
    url: str
    title: str
    text: str
    source_domain: str
    fetched_at_utc: str
    content_hash: str
    source_type: str


@dataclass
class ChunkRecord:
    chunk_id: str
    url: str
    title: str
    source_domain: str
    chunk_index: int
    text: str
    content_hash: str
    fetched_at_utc: str
    source_type: str


# =========================================================
# OPTIONAL PDF READER
# =========================================================
def read_pdf_bytes(file_bytes: bytes) -> str:
    if pypdf is None:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append(txt)
        return clean_text("\n\n".join(pages))
    except Exception:
        return ""


# =========================================================
# ROBOTS
# =========================================================
class RobotsManager:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.parsers: Dict[str, RobotFileParser] = {}
        self.lock = threading.Lock()

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        with self.lock:
            if root not in self.parsers:
                rp = RobotFileParser()
                rp.set_url(urljoin(root, "/robots.txt"))
                try:
                    rp.read()
                except Exception:
                    pass
                self.parsers[root] = rp
            rp = self.parsers[root]
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True


# =========================================================
# CRAWLER
# =========================================================
class WildfireCrawler:
    def __init__(
        self,
        seed_urls: List[str],
        allowed_domains: List[str],
        max_pages: int = DEFAULT_MAX_PAGES,
        max_depth: int = DEFAULT_MAX_DEPTH,
        request_delay_sec: float = DEFAULT_REQUEST_DELAY,
        max_workers: int = DEFAULT_MAX_WORKERS,
        user_agent: str = "WildfireKnowledgeLab/1.0 (+research)"
    ):
        self.seed_urls = [normalize_url(x) for x in seed_urls if x.strip()]
        self.allowed_domains = [x.strip() for x in allowed_domains if x.strip()]
        self.max_pages = int(max_pages)
        self.max_depth = int(max_depth)
        self.request_delay_sec = float(request_delay_sec)
        self.max_workers = int(max_workers)
        self.user_agent = user_agent
        self.headers = {"User-Agent": self.user_agent}
        self.robots = RobotsManager(self.user_agent)

        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.visited = set()
        self.page_records: List[PageRecord] = []
        self.last_request_time = 0.0

    def _rate_limit(self):
        with self.lock:
            now = time.time()
            wait = self.request_delay_sec - (now - self.last_request_time)
            if wait > 0:
                time.sleep(wait)
            self.last_request_time = time.time()

    def _fetch_html(self, url: str) -> Optional[requests.Response]:
        try:
            self._rate_limit()
            r = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            ctype = r.headers.get("Content-Type", "").lower()
            if r.status_code != 200:
                return None
            if "text/html" not in ctype:
                return None
            return r
        except Exception:
            return None

    def _extract_main_text(self, html: str) -> str:
        txt = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if txt:
            return clean_text(txt)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        return clean_text(soup.get_text("\n", strip=True))

    def _worker(self):
        while True:
            try:
                item = self.q.get(timeout=1)
            except queue.Empty:
                return

            url, depth = item
            try:
                with self.lock:
                    if url in self.visited or len(self.visited) >= self.max_pages:
                        continue
                    self.visited.add(url)

                if not is_allowed_domain(url, self.allowed_domains):
                    continue
                if not looks_like_html_url(url):
                    continue
                if not self.robots.allowed(url):
                    continue

                resp = self._fetch_html(url)
                if resp is None:
                    continue

                html = resp.text
                text = self._extract_main_text(html)
                title = html_title(html, fallback=url)

                if len(text.split()) >= 80:
                    self.page_records.append(
                        PageRecord(
                            url=url,
                            title=title,
                            text=text,
                            source_domain=urlparse(url).netloc.lower(),
                            fetched_at_utc=utc_now_iso(),
                            content_hash=safe_hash(text),
                            source_type="web"
                        )
                    )

                if depth < self.max_depth:
                    for link in extract_links(url, html):
                        if is_allowed_domain(link, self.allowed_domains):
                            with self.lock:
                                if link not in self.visited and len(self.visited) < self.max_pages:
                                    self.q.put((link, depth + 1))
            finally:
                self.q.task_done()

    def crawl(self) -> List[PageRecord]:
        for s in self.seed_urls:
            self.q.put((s, 0))
        threads = []
        for _ in range(self.max_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            threads.append(t)
        self.q.join()
        for t in threads:
            t.join(timeout=1)
        return self.page_records


# =========================================================
# VECTOR STORE
# =========================================================
class WildfireKnowledgeBase:
    def __init__(self, chroma_dir: str, collection_name: str, embed_model_name: str):
        if chromadb is None or SentenceTransformer is None:
            raise RuntimeError(
                "Missing required packages. Install chromadb and sentence-transformers."
            )
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = SentenceTransformer(embed_model_name)

    def page_to_chunks(self, page: PageRecord) -> List[ChunkRecord]:
        chunks = split_into_word_chunks(page.text)
        out = []
        for i, c in enumerate(chunks):
            cid = safe_hash(f"{page.url}|{page.content_hash}|{i}|{c[:120]}")
            out.append(
                ChunkRecord(
                    chunk_id=cid,
                    url=page.url,
                    title=page.title,
                    source_domain=page.source_domain,
                    chunk_index=i,
                    text=c,
                    content_hash=page.content_hash,
                    fetched_at_utc=page.fetched_at_utc,
                    source_type=page.source_type,
                )
            )
        return out

    def upsert_pages(self, pages: List[PageRecord]) -> int:
        chunks = []
        for p in pages:
            chunks.extend(self.page_to_chunks(p))
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = []
        for c in chunks:
            metas.append({
                "url": c.url,
                "title": c.title,
                "source_domain": c.source_domain,
                "chunk_index": c.chunk_index,
                "content_hash": c.content_hash,
                "fetched_at_utc": c.fetched_at_utc,
                "source_type": c.source_type,
            })

        embeddings = self.embedder.encode(docs, show_progress_bar=False, normalize_embeddings=True).tolist()

        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )
        return len(chunks)

    def query(self, question: str, top_k: int = TOP_K) -> List[Dict]:
        qemb = self.embedder.encode([question], normalize_embeddings=True).tolist()[0]
        res = self.collection.query(
            query_embeddings=[qemb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        hits = []
        for d, m, dist in zip(docs, metas, distances):
            hits.append({
                "text": d,
                "metadata": m,
                "distance": float(dist) if dist is not None else None,
            })
        return hits

    def peek(self, n: int = 30) -> List[Dict]:
        try:
            data = self.collection.get(limit=n, include=["documents", "metadatas"])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])
            out = []
            for d, m in zip(docs, metas):
                out.append({"text": d, "metadata": m})
            return out
        except Exception:
            return []

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def clear(self):
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)


@st.cache_resource(show_spinner=False)
def get_kb():
    return WildfireKnowledgeBase(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embed_model_name=EMBED_MODEL_NAME,
    )


# =========================================================
# INGEST HELPERS
# =========================================================
def ingest_uploaded_files(files) -> List[PageRecord]:
    pages = []
    for f in files or []:
        file_bytes = f.read()
        fname = f.name
        suffix = os.path.splitext(fname)[1].lower()

        text = ""
        if suffix in [".txt", ".md", ".csv", ".json"]:
            try:
                text = clean_text(file_bytes.decode("utf-8", errors="ignore"))
            except Exception:
                text = ""
        elif suffix == ".pdf":
            text = read_pdf_bytes(file_bytes)
        else:
            try:
                text = clean_text(file_bytes.decode("utf-8", errors="ignore"))
            except Exception:
                text = ""

        if not text or len(text.split()) < 30:
            continue

        synthetic_url = f"upload://{fname}"
        pages.append(
            PageRecord(
                url=synthetic_url,
                title=fname,
                text=text,
                source_domain="uploaded-file",
                fetched_at_utc=utc_now_iso(),
                content_hash=safe_hash(text),
                source_type="upload"
            )
        )
    return pages


def ingest_manual_text(title: str, text: str) -> List[PageRecord]:
    text = clean_text(text)
    if len(text.split()) < 20:
        return []
    return [
        PageRecord(
            url=f"note://{safe_hash(title + text)[:12]}",
            title=title.strip() or "Manual note",
            text=text,
            source_domain="manual-note",
            fetched_at_utc=utc_now_iso(),
            content_hash=safe_hash(text),
            source_type="note"
        )
    ]


# =========================================================
# OPENAI ANSWERING
# =========================================================
def get_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment or Streamlit secrets.")
    return OpenAI(api_key=OPENAI_API_KEY)


def build_grounded_prompt(question: str, hits: List[Dict]) -> Tuple[str, str]:
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        md = hit["metadata"]
        title = md.get("title", "Untitled")
        url = md.get("url", "")
        domain = md.get("source_domain", "")
        excerpt = hit["text"]
        block = (
            f"[Source {i}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Domain: {domain}\n"
            f"Excerpt:\n{excerpt}\n"
        )
        context_blocks.append(block)

    system_prompt = (
        "You are a wildfire knowledge analyst. "
        "Answer using only the provided retrieved sources. "
        "Be direct, useful, and grounded. "
        "If the sources do not fully answer the question, say what is missing. "
        "At the end, include a short Sources section listing source numbers and titles."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved sources:\n\n" + "\n\n".join(context_blocks)
    )
    return system_prompt, user_prompt


def answer_question_with_openai(question: str, hits: List[Dict]) -> Tuple[str, str]:
    client = get_openai_client()
    system_prompt, user_prompt = build_grounded_prompt(question, hits)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)

    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    cost_str = estimate_cost_text(prompt_tokens, completion_tokens, OPENAI_MODEL)

    return answer, f"model={OPENAI_MODEL} | prompt={prompt_tokens} | completion={completion_tokens} | {cost_str}"


def answer_question_local(question: str, hits: List[Dict]) -> Tuple[str, str]:
    if not hits:
        return "I could not find any relevant indexed wildfire sources yet. Ingest some sites, PDFs, or notes first.", "local fallback"
    lines = [f"Question: {question}", ""]
    lines.append("Most relevant retrieved wildfire sources:")
    lines.append("")
    for i, hit in enumerate(hits, start=1):
        md = hit["metadata"]
        lines.append(f"{i}. {md.get('title', 'Untitled')} | {md.get('url', '')}")
        lines.append(hit["text"][:900].strip())
        lines.append("")
    lines.append("This is retrieval output only because no OpenAI key was available.")
    return "\n".join(lines), "local fallback"


# =========================================================
# UI HEADER
# =========================================================
st.markdown(f"""
<div class="hero-card">
    <div style="font-size:2rem; font-weight:900; margin-bottom:0.25rem;">🔥 {APP_TITLE}</div>
    <div class="small-muted">{APP_SUBTITLE}</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">Vector DB</div><div style="font-size:1.25rem; font-weight:800;">{COLLECTION_NAME}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">Embedding model</div><div style="font-size:1.0rem; font-weight:800;">{EMBED_MODEL_NAME}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">LLM</div><div style="font-size:1.25rem; font-weight:800;">{OPENAI_MODEL}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">Storage</div><div style="font-size:1.25rem; font-weight:800;">{CHROMA_DIR}</div></div>', unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Ask", "Ingest", "Library", "Settings"])


# =========================================================
# ASK TAB
# =========================================================
with tab1:
    kb_ok = True
    kb = None
    try:
        kb = get_kb()
        st.session_state.kb_ready = True
    except Exception as e:
        kb_ok = False
        st.error(f"Knowledge base init failed: {e}")

    ask_col1, ask_col2 = st.columns([1.5, 1])

    with ask_col1:
        question = st.text_area(
            "Ask a wildfire question",
            value="What are the main themes in the indexed wildfire sources, and what do they say about operations, preparedness, or incident information?",
            height=150,
        )
        top_k = st.slider("Retrieved source count", 3, 12, 6, 1)

        st.markdown('<div class="big-button">', unsafe_allow_html=True)
        run_ask = st.button("ASK THE FIRE BRAIN", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with ask_col2:
        st.markdown("### Live status")
        chunk_count = kb.count() if kb_ok else 0
        st.metric("Indexed chunks", f"{chunk_count:,}")
        st.metric("Recent pages", f"{st.session_state.latest_stats.get('pages', 0):,}")
        st.metric("Recent chunks", f"{st.session_state.latest_stats.get('chunks', 0):,}")
        st.metric("Recent unique sources", f"{st.session_state.latest_stats.get('sources', 0):,}")

    if run_ask:
        if not kb_ok:
            st.stop()

        with st.spinner("Searching indexed wildfire knowledge..."):
            hits = kb.query(question, top_k=top_k)

        st.session_state.last_hits = hits

        with st.spinner("Building grounded answer..."):
            try:
                if OPENAI_API_KEY and OpenAI is not None:
                    answer, usage_str = answer_question_with_openai(question, hits)
                else:
                    answer, usage_str = answer_question_local(question, hits)
            except Exception as e:
                answer, usage_str = answer_question_local(question, hits)
                answer = f"{answer}\n\nOpenAI call failed, so this fell back to retrieval-only output.\nError: {e}"

        st.session_state.last_answer = answer

        st.markdown("### Answer")
        st.write(answer)
        st.caption(usage_str)

        st.markdown("### Retrieved sources")
        for i, hit in enumerate(hits, start=1):
            md = hit["metadata"]
            title = md.get("title", "Untitled")
            url = md.get("url", "")
            dom = md.get("source_domain", "")
            source_type = md.get("source_type", "")
            excerpt = hit["text"][:850].strip()

            st.markdown(
                f"""
                <div class="source-card">
                    <div style="font-size:1rem; font-weight:800;">[{i}] {title}</div>
                    <div style="font-size:0.86rem; opacity:0.72; margin-bottom:0.45rem;">
                        {dom} &nbsp;|&nbsp; {source_type} &nbsp;|&nbsp; <a href="{url}" target="_blank">{url}</a>
                    </div>
                    <div style="font-size:0.95rem; line-height:1.45;">{excerpt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif st.session_state.last_answer:
        st.markdown("### Last answer")
        st.write(st.session_state.last_answer)


# =========================================================
# INGEST TAB
# =========================================================
with tab2:
    kb_ok = True
    kb = None
    try:
        kb = get_kb()
    except Exception as e:
        kb_ok = False
        st.error(f"Knowledge base init failed: {e}")

    st.markdown("### Web crawling")
    colA, colB = st.columns([1.2, 1])

    with colA:
        seed_text = st.text_area(
            "Seed URLs",
            value="\n".join(DEFAULT_SEEDS),
            height=180,
        )
        domains_text = st.text_area(
            "Allowed domains",
            value="\n".join(DEFAULT_ALLOWED_DOMAINS),
            height=160,
        )

    with colB:
        max_pages = st.number_input("Max pages", min_value=10, max_value=5000, value=150, step=10)
        max_depth = st.number_input("Max crawl depth", min_value=0, max_value=6, value=2, step=1)
        max_workers = st.number_input("Workers", min_value=1, max_value=24, value=6, step=1)
        request_delay = st.number_input("Delay between requests (sec)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

        st.markdown('<div class="big-button">', unsafe_allow_html=True)
        run_crawl = st.button("CRAWL AND INGEST", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### File and note ingestion")

    up1, up2 = st.columns([1, 1])

    with up1:
        uploaded_files = st.file_uploader(
            "Upload wildfire PDFs, text, CSV, JSON, or notes",
            type=["pdf", "txt", "md", "csv", "json"],
            accept_multiple_files=True,
        )
        ingest_uploads = st.button("INGEST UPLOADED FILES", use_container_width=True)

    with up2:
        manual_title = st.text_input("Manual note title", value="Wildfire note")
        manual_text = st.text_area(
            "Paste notes, policy text, methods, or incident summaries",
            value="",
            height=180
        )
        ingest_note = st.button("INGEST NOTE", use_container_width=True)

    if run_crawl and kb_ok:
        seeds = [x.strip() for x in seed_text.splitlines() if x.strip()]
        allowed = [x.strip() for x in domains_text.splitlines() if x.strip()]

        with st.spinner("Crawling wildfire websites and building the knowledge base..."):
            crawler = WildfireCrawler(
                seed_urls=seeds,
                allowed_domains=allowed,
                max_pages=max_pages,
                max_depth=max_depth,
                request_delay_sec=request_delay,
                max_workers=max_workers,
            )
            pages = crawler.crawl()
            added_chunks = kb.upsert_pages(pages)

        unique_sources = len({p.url for p in pages})
        st.session_state.latest_stats = {
            "pages": len(pages),
            "chunks": added_chunks,
            "sources": unique_sources,
        }
        st.session_state.latest_docs_preview = [asdict(p) for p in pages[:20]]
        st.session_state.ingest_log.append(
            f"{utc_now_iso()} | crawl | pages={len(pages)} | chunks={added_chunks} | sources={unique_sources}"
        )

        st.success(f"Ingested {len(pages)} pages into {added_chunks} chunks from {unique_sources} sources.")

    if ingest_uploads and kb_ok:
        with st.spinner("Reading uploaded wildfire files..."):
            pages = ingest_uploaded_files(uploaded_files)
            added_chunks = kb.upsert_pages(pages) if pages else 0

        unique_sources = len({p.url for p in pages})
        st.session_state.latest_stats = {
            "pages": len(pages),
            "chunks": added_chunks,
            "sources": unique_sources,
        }
        st.session_state.latest_docs_preview = [asdict(p) for p in pages[:20]]
        st.session_state.ingest_log.append(
            f"{utc_now_iso()} | uploads | pages={len(pages)} | chunks={added_chunks} | sources={unique_sources}"
        )

        if pages:
            st.success(f"Ingested {len(pages)} uploaded documents into {added_chunks} chunks.")
        else:
            st.warning("No usable text was found in the uploaded files.")

    if ingest_note and kb_ok:
        with st.spinner("Ingesting note..."):
            pages = ingest_manual_text(manual_title, manual_text)
            added_chunks = kb.upsert_pages(pages) if pages else 0

        unique_sources = len({p.url for p in pages})
        st.session_state.latest_stats = {
            "pages": len(pages),
            "chunks": added_chunks,
            "sources": unique_sources,
        }
        st.session_state.latest_docs_preview = [asdict(p) for p in pages[:20]]
        st.session_state.ingest_log.append(
            f"{utc_now_iso()} | note | pages={len(pages)} | chunks={added_chunks} | sources={unique_sources}"
        )

        if pages:
            st.success(f"Ingested note into {added_chunks} chunks.")
        else:
            st.warning("The note was too short to ingest.")

    st.markdown("### Latest ingest log")
    if st.session_state.ingest_log:
        for row in reversed(st.session_state.ingest_log[-12:]):
            st.code(row)
    else:
        st.info("Nothing ingested yet.")


# =========================================================
# LIBRARY TAB
# =========================================================
with tab3:
    kb_ok = True
    kb = None
    try:
        kb = get_kb()
    except Exception as e:
        kb_ok = False
        st.error(f"Knowledge base init failed: {e}")

    if kb_ok:
        st.markdown("### Indexed library preview")
        preview = kb.peek(40)

        if preview:
            rows = []
            seen = set()
            for item in preview:
                md = item["metadata"]
                key = (md.get("url", ""), md.get("title", ""))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "title": md.get("title", ""),
                    "url": md.get("url", ""),
                    "domain": md.get("source_domain", ""),
                    "type": md.get("source_type", ""),
                    "fetched_at_utc": md.get("fetched_at_utc", ""),
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("The library is empty. Crawl some sites or upload some wildfire docs.")

        st.markdown("### Recent page preview")
        if st.session_state.latest_docs_preview:
            df2 = pd.DataFrame([
                {
                    "title": x.get("title", ""),
                    "url": x.get("url", ""),
                    "domain": x.get("source_domain", ""),
                    "type": x.get("source_type", ""),
                    "words": len((x.get("text", "") or "").split()),
                }
                for x in st.session_state.latest_docs_preview
            ])
            st.dataframe(df2, use_container_width=True, hide_index=True)
        else:
            st.info("No recent ingest preview yet.")


# =========================================================
# SETTINGS TAB
# =========================================================
with tab4:
    st.markdown("### Environment")
    st.code(
        f"CHROMA_DIR={CHROMA_DIR}\n"
        f"COLLECTION_NAME={COLLECTION_NAME}\n"
        f"EMBED_MODEL_NAME={EMBED_MODEL_NAME}\n"
        f"OPENAI_MODEL={OPENAI_MODEL}\n"
        f"OPENAI_KEY_SET={'yes' if bool(OPENAI_API_KEY) else 'no'}"
    )

    st.markdown("### Package checks")
    st.write({
        "chromadb": chromadb is not None,
        "sentence_transformers": SentenceTransformer is not None,
        "openai": OpenAI is not None,
        "pypdf": pypdf is not None,
        "trafilatura": True,
        "beautifulsoup4": True,
    })

    kb_ok = True
    kb = None
    try:
        kb = get_kb()
    except Exception as e:
        kb_ok = False
        st.error(f"Knowledge base init failed: {e}")

    if kb_ok:
        st.markdown("### Danger zone")
        clear = st.button("CLEAR INDEX", type="secondary")
        if clear:
            kb.clear()
            st.session_state.last_answer = ""
            st.session_state.last_hits = []
            st.session_state.latest_stats = {"pages": 0, "chunks": 0, "sources": 0}
            st.session_state.latest_docs_preview = []
            st.session_state.ingest_log.append(f"{utc_now_iso()} | cleared index")
            st.success("Vector index cleared.")

st.divider()
st.caption("Wildfire Knowledge Lab | website + PDF + note ingestion | vector search | grounded answering")
