#!/usr/bin/env python3
# history_hotspots_rag_glm.py

import os
import json
import time
import requests
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Embedding / vector search
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    REAL_EMBED = True
except:
    REAL_EMBED = False

# ----------------------------
# 1. Load ZhipuAI client
# ----------------------------
from zai import ZhipuAiClient
load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

if not ZHIPU_API_KEY:
    raise RuntimeError("请在 .env 中设置 ZHIPU_API_KEY！")

client = ZhipuAiClient(api_key=ZHIPU_API_KEY)

# ----------------------------
# 2. Mock dataset (fallback)
# ----------------------------
MOCK_PAPERS = [
    {
        "id": "m1",
        "title": "OCR Correction for Historical Documents",
        "abstract": "We propose LM-based post-processing to reduce OCR character error rates across 19th-century scanned texts.",
        "year": 2024,
        "source": "mock"
    },
    {
        "id": "m2",
        "title": "Social Networks of Qing Dynasty",
        "abstract": "We build networks from archival correspondence to study political influence structures.",
        "year": 2023,
        "source": "mock"
    },
    {
        "id": "m3",
        "title": "NER for Gazetteers",
        "abstract": "Applying NER to historical gazetteers helps extract place-names and administrative units.",
        "year": 2024,
        "source": "mock"
    }
]

# ----------------------------
# 3. OpenAlex Fetcher
# ----------------------------
def fetch_openalex(query: str, max_results=200):
    base = "https://api.openalex.org/works"
    params = {
        "search": query,
        "filter": "from_publication_date:2023-01-01",
        "per-page": 200
    }

    r = requests.get(base, params=params, timeout=10)
    data = r.json()
    works = data.get("results", [])

    out = []
    for w in works[:max_results]:
        abstract = None
        if w.get("abstract") is not None:
            abstract = w["abstract"]
        elif w.get("abstract_inverted_index") is not None:
            inv = w["abstract_inverted_index"]
            max_pos = max([max(v) for v in inv.values()])
            tokens = [""] * (max_pos + 1)
            for tok, idxs in inv.items():
                for i in idxs:
                    tokens[i] = tok
            abstract = " ".join(tokens)

        out.append({
            "id": w["id"],
            "title": w["title"],
            "abstract": abstract or "",
            "year": w.get("publication_year"),
            "source": "openalex"
        })

    return out


# ----------------------------
# 4. Retriever
# ----------------------------
class Retriever:
    def __init__(self):
        if REAL_EMBED:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.model = None
        self.index = None
        self.ids = []
        self.corpus = []
        self.docs = {}

    def build(self, docs):
        self.docs = {d["id"]: d for d in docs}
        self.ids = [d["id"] for d in docs]
        self.corpus = [
            d["title"] + ". " + (d["abstract"] or "")
            for d in docs
        ]

        if REAL_EMBED:
            print("[INFO] Building FAISS index...")
            emb = self.model.encode(self.corpus, convert_to_tensor=False)
            emb = np.array(emb).astype("float32")
            faiss.normalize_L2(emb)
            d = emb.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(emb)
            self.emb_matrix = emb
        else:
            print("[WARN] sentence-transformers unavailable → fallback keyword search.")
            self.index = None

    def search(self, query, k=20):
        if self.index is None:
            # fallback: keyword match
            qt = set(query.lower().split())
            scored = []
            for i, txt in enumerate(self.corpus):
                score = len(qt & set(txt.lower().split()))
                scored.append((self.ids[i], score))
            return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

        # real embedding search
        q_emb = self.model.encode([query], convert_to_tensor=False)
        q_emb = np.array(q_emb).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        out = []
        for idx, score in zip(I[0], D[0]):
            out.append((self.ids[idx], float(score)))
        return out


# ----------------------------
# 5. ZhipuAI Chat wrapper
# ----------------------------
def chat_with_glm(prompt: str) -> str:
    """
    使用 GLM-4.5-FLASH 进行 RAG 总结
    """
    print("[INFO] 调用 GLM 模型生成总结（流式输出）...\n")

    response = client.chat.completions.create(
        model="glm-4.5-flash",
        messages=[
            {"role": "system", "content": "你是一名历史学研究趋势分析专家。"},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        thinking={"type": "enabled"},
        max_tokens=2048,
        temperature=0.2
    )

    final_text = ""

    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            final_text += delta

    print("\n\n[INFO] GLM 总结完成。\n")
    return final_text


# ----------------------------
# 6. Full RAG pipeline
# ----------------------------
def run_pipeline(query: str, use_openalex=False):
    print("\n=========== 🧠 历史学热点 RAG 系统 ===========\n")

    # Step 1: fetch real or mock papers
    if use_openalex:
        print("[INFO] 正在从 OpenAlex 拉取真实论文...")
        docs = fetch_openalex(query)
        if not docs:
            print("[WARN] 无法从 OpenAlex 获取，fallback 到 mock")
            docs = MOCK_PAPERS
    else:
        docs = MOCK_PAPERS

    print(f"[INFO] 文献数量：{len(docs)}")

    # Step 2: build retriever
    r = Retriever()
    r.build(docs)

    # Step 3: search
    hits = r.search(query, k=20)
    print(f"[INFO] 检索到 {len(hits)} 条文献。")

    # Step 4: assemble text for LLM
    evidence = []
    for pid, score in hits:
        doc = r.docs[pid]
        snippet = doc["title"] + "\n" + doc["abstract"][:500]
        evidence.append(snippet)

    evidence_text = "\n\n".join(evidence)

    # Step 5: call GLM
    PROMPT = f"""
请根据以下最新历史学文献（均为2023-2025年）内容，分析「历史学最新研究热点」。

文献列表：
{evidence_text}

请输出：
1. 一个对历史学领域过去两年（2023–2025）的整体趋势总结  
2. 五大研究热点（每个热点 2 句解释）  
3. 每个热点至少列出两篇代表论文（标题 + 年份）  
4. 给出三个未来可研究方向（可作为科研选题）

请用清晰结构化格式回答。
"""

    summary = chat_with_glm(PROMPT)

    # Step 6: return
    return {
        "query": query,
        "llm_output": summary,
        "retrieved_papers": hits
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    print("=== 历史学热点检索助手（RAG + GLM） ===")
    query = input("请输入你的问题（例如：历史学 最新 研究 热点）:\n> ").strip()
    if not query:
        query = "历史学 最新 研究 热点"

    out = run_pipeline(query, use_openalex=False)

    with open("history_rag_output.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n结果已保存到 history_rag_output.json\n")


if __name__ == "__main__":
    main()
