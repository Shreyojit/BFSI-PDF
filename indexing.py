import os
import json
from typing import List, Dict, Any
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .aws_clients import bedrock_rt_client
from .config import get_settings
from .storage import save_json, load_json

S = get_settings()


def _embed(text: str) -> List[float]:
    if not text or not text.strip():
        return []
    rt = bedrock_rt_client()
    body = {"inputText": text}
    resp = rt.invoke_model(
        modelId=S.EMBEDDING_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    out = json.loads(resp.get("body").read())
    return out.get("embedding") or out.get("vector") or []


def build_items(lines_by_page, forms_by_page, tables_by_page) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)

    present_pages = set(lines_by_page.keys()) | set(forms_by_page.keys()) | set(tables_by_page.keys())
    for page in sorted(present_pages or {1}):
        page_lines = [t for _, _, t in lines_by_page.get(page, [])]
        page_text = "\n".join(page_lines).strip()
        if page_text:
            for ch in splitter.split_text(page_text):
                items.append({"page": page, "type": "text", "text": ch})

    for page, kv_list in forms_by_page.items():
        if not kv_list:
            continue
        kv_text = "\n".join(kv_list)
        for ch in splitter.split_text(kv_text):
            items.append({"page": page, "type": "text", "text": ch})

    for page, tables in tables_by_page.items():
        for grid in tables:
            lines = [" | ".join([c or "" for c in row]) for row in grid]
            ttext = "\n".join(lines).strip()
            if ttext:
                items.append({"page": page, "type": "table", "text": ttext})

    return items


def build_faiss(index_items: List[Dict[str, Any]]):
    vecs = []
    kept: List[Dict[str, Any]] = []
    for it in index_items:
        emb = _embed(it["text"])
        if emb:
            it["embedding"] = emb
            vecs.append(emb)
            kept.append(it)

    if not vecs:
        raise RuntimeError("No embeddings generated. Check Bedrock access/quotas.")

    mat = np.array(vecs, dtype=np.float32)
    dim = mat.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(mat)

    # If local persist is disabled, just keep in memory
    if S.DISABLE_LOCAL_PERSIST:
        meta = [{k: v for k, v in it.items() if k != "embedding"} for it in kept]
        return index, meta

    # Otherwise persist to disk
    os.makedirs(os.path.dirname(S.INDEX_PATH), exist_ok=True)
    faiss.write_index(index, S.INDEX_PATH)
    meta = [{k: v for k, v in it.items() if k != "embedding"} for it in kept]
    save_json(S.META_PATH, meta)
    return index, meta


def load_faiss():
    # If local persist is disabled, don't try loading from disk
    if S.DISABLE_LOCAL_PERSIST:
        return None, None
    if not (os.path.exists(S.INDEX_PATH) and os.path.exists(S.META_PATH)):
        return None, None
    index = faiss.read_index(S.INDEX_PATH)
    meta = load_json(S.META_PATH)
    return index, meta
