#!/usr/bin/env python3
# academic_rag_agent_clean.py

import os, time, json, requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# --- Load .env ---
from dotenv import load_dotenv

load_dotenv()

# --- Optional deps ---
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import networkx as nx
import PyPDF2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# --- LLM client (GLM-4.5) ---
from zhipuai import ZhipuAI

ZHIPU_KEY = os.getenv("ZHIPU_API_KEY", None)
if not ZHIPU_KEY:
    raise ValueError("❌ ERROR: 请在 .env 中设置 ZHIPU_API_KEY=xxxx")
zai_client = ZhipuAI(api_key=ZHIPU_KEY)

# --- Config ---
USE_ARXIV = True
USE_OPENALEX = True
MAX_RESULTS = 50


# ===========================
#       DATA STRUCTURE
# ===========================
@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    year: int = None
    doi: str = None
    source: str = None
    pdf_url: str = None


# ===========================
#       RETRIEVAL MODULES
# ===========================

# ---- ArXiv ----
def search_arxiv(query: str, max_results: int = MAX_RESULTS) -> List[Paper]:
    print(f"[INFO] Searching ArXiv for: {query}")

    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    txt = r.text

    papers = []
    entries = txt.split("<entry>")[1:]

    import re
    for e in entries:
        title = re.search(r"<title>(.*?)</title>", e, re.S)
        title = title.group(1).strip() if title else "Untitled"

        abstract = re.search(r"<summary>(.*?)</summary>", e, re.S)
        abstract = abstract.group(1).strip() if abstract else ""

        pid = re.search(r"<id>(.*?)</id>", e)
        pid = pid.group(1).strip() if pid else None

        authors = re.findall(r"<name>(.*?)</name>", e)

        yearm = re.search(r"<published>(\d{4})", e)
        year = int(yearm.group(1)) if yearm else None

        pdf_url = None
        pdf_match = re.search(r'<link title="pdf" href="(.*?)"/>', e)
        if pdf_match:
            pdf_url = pdf_match.group(1)

        papers.append(Paper(
            id=pid, title=title, abstract=abstract,
            authors=authors, year=year,
            source="arxiv", pdf_url=pdf_url
        ))

    return papers


# ---- OpenAlex ----
def fetch_openalex_works(query: str, from_year=2023, max_results=MAX_RESULTS) -> List[Paper]:
    print(f"[INFO] Searching OpenAlex for: {query}")

    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "filter": f"from_publication_date:{from_year}-01-01",
        "per-page": 50,
        "page": 1
    }

    results = []
    while len(results) < max_results:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        items = data.get("results", [])
        if not items:
            break

        for it in items:
            abstract = it.get("abstract") or ""
            results.append(Paper(
                id=it.get("id"),
                title=it.get("title"),
                abstract=abstract,
                authors=[a["author"]["display_name"] for a in it.get("authorships", [])],
                year=it.get("publication_year"),
                doi=it.get("doi"),
                source="openalex"
            ))

        params["page"] += 1
        if len(items) < 50:
            break
        time.sleep(0.2)

    return results[:max_results]


# ---- Combine sources ----
def multi_source_search(query: str) -> List[Paper]:
    results = []
    if USE_ARXIV:
        results.extend(search_arxiv(query))
    if USE_OPENALEX:
        results.extend(fetch_openalex_works(query))

    # Dedup by id
    seen = set()
    final = []
    for p in results:
        key = p.id or p.title
        if key not in seen:
            seen.add(key)
            final.append(p)

    print(f"[INFO] Total papers retrieved: {len(final)}")
    return final


# ===========================
#       EMBEDDING + FAISS
# ===========================

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)


class FaissIndex:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def build(self, embeddings, ids):
        emb = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(emb)
        self.index.add(emb)
        self.ids = ids
        self.emb_matrix = emb

    def search(self, q_emb, topk=10):
        q = np.array(q_emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, topk)
        return [(int(I[0][i]), float(D[0][i])) for i in range(topk)]


# ===========================
#         PDF PARSING
# ===========================

def parse_pdf_text(path: str, max_pages=5) -> str:
    try:
        reader = PyPDF2.PdfReader(path)
        text = []
        for i, page in enumerate(reader.pages[:max_pages]):
            text.append(page.extract_text() or "")
        text = "\n".join(text).strip()
        if len(text) > 200:
            return text
    except:
        text = ""

    # OCR fallback
    try:
        images = convert_from_path(path, first_page=1, last_page=max_pages)
        ocr_texts = [pytesseract.image_to_string(img) for img in images]
        return "\n".join(ocr_texts)
    except:
        return text


# ===========================
#         CITATION GRAPH
# ===========================

def build_citation_graph(papers: List[Paper]) -> nx.DiGraph:
    G = nx.DiGraph()
    for p in papers:
        G.add_node(p.id, title=p.title, year=p.year, source=p.source)

    # OpenAlex references
    for p in papers:
        if p.source == "openalex":
            try:
                r = requests.get(p.id, timeout=10)
                data = r.json()
                for ref in data.get("referenced_works", []):
                    G.add_edge(p.id, ref, relation="cites")
            except:
                pass

    print(f"[INFO] KG built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ===========================
#         LLM CALL
# ===========================

def call_llm(prompt: str) -> str:
    resp = zai_client.chat.completions.create(
        model="glm-4.5-flash",
        messages=[
            {"role": "system", "content": "You are an expert research summarizer."},
            {"role": "user", "content": prompt}
        ],
        thinking={"type": "enabled"},
        stream=False,
        max_tokens=2048,
        temperature=0.2
    )
    return resp.choices[0].message.content


# ===========================
#       AGENT EXECUTION
# ===========================

def agent_run(query: str, top_k=20):
    print("\n================ AGENT START ==============")
    print("[QUERY]:", query)

    # 1. Retrieval
    papers = multi_source_search(query)

    # 2. Embedding
    emb = EmbeddingManager()
    texts = [p.title + ". " + (p.abstract or "") for p in papers]
    vecs = emb.encode(texts)

    # 3. FAISS
    dim = len(vecs[0])
    index = FaissIndex(dim)
    ids = [i for i in range(len(papers))]
    index.build(vecs, ids)

    qvec = emb.encode([query])[0]
    hits = index.search(qvec, topk=min(top_k, len(papers)))
    print(f"[INFO] Top-{len(hits)} retrieved by embedding")

    # 4. Citation graph
    G = build_citation_graph(papers)

    # 5. Evidence snippets
    snippets = []
    for idx, score in hits:
        p = papers[idx]
        snippets.append(f"{p.title} ({p.year})\n{p.abstract[:800]}")

    evidence = "\n\n".join(snippets)

    # 6. LLM Summarization
    prompt = f"""
User query: {query}

Based on the following retrieved papers, produce:
1. Overview
2. Top 5 research hotspots
3. 2 representative papers per hotspot
4. 3 research project ideas

Evidence:
{evidence}
"""

    out = call_llm(prompt)
    print("\n================ LLM OUTPUT ===============")
    print(out)

    # Save result
    result = {
        "query": query,
        "papers": [p.__dict__ for p in papers],
        "hits": hits,
        "llm_summary": out
    }
    json.dump(result, open("academic_rag_clean_last.json", "w", encoding="utf-8"), indent=2)

    print("\nSaved: academic_rag_clean_last.json")
    return result


# ===========================
#            CLI
# ===========================

if __name__ == "__main__":
    q = input("请输入你的科研问题:\n> ").strip()
    if not q:
        q = "large language model robustness research hotspots"
    agent_run(q)
