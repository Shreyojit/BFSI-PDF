import os
from collections import defaultdict
from typing import List, Dict, Any
import boto3

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from .rag_config import settings
from .party_normalize import canonical_party_key, is_likely_plan, is_likely_service

s3 = boto3.client("s3", region_name=settings.AWS_REGION)

def _list_txt(prefix: str) -> List[str]:
    keys, token = [], None
    while True:
        kw = {"Bucket": settings.S3_BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            k = o["Key"]
            if k.lower().endswith(".txt"):
                keys.append(k)
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return keys

def _read_text(key: str) -> str:
    b = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)["Body"].read()
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="ignore")

def _title_from_key(key: str) -> str:
    import os
    return os.path.splitext(os.path.basename(key))[0]

def _doc_type(title: str, key: str) -> str:
    kl = key.lower()
    if "/plan/" in kl:
        return "plan"
    if "/services/" in kl or "/service/" in kl:
        return "service"
    if is_likely_service(title):
        return "service"
    if is_likely_plan(title):
        return "plan"
    return "unknown"

def build() -> Dict[str, Any]:
    base = settings.S3_BASE_PREFIX.strip().strip("/")
    plan_prefix = f"{base}/plan/"
    svc_prefix  = f"{base}/services/"

    plan_keys = _list_txt(plan_prefix)
    svc_keys  = _list_txt(svc_prefix)
    all_keys  = plan_keys + svc_keys
    if not all_keys:
        raise RuntimeError(f"No .txt files found under s3://{settings.S3_BUCKET}/{base}/(plan|services)/")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    emb = BedrockEmbeddings(region_name=settings.AWS_REGION, model_id=settings.EMBEDDING_MODEL_ID)

    party_texts = defaultdict(list)
    party_metas = defaultdict(list)

    for key in all_keys:
        title = _title_from_key(key)
        party = canonical_party_key(title)
        doctype = _doc_type(title, key)
        content = _read_text(key)
        chunks = splitter.split_text(content)
        for i, ch in enumerate(chunks):
            party_texts[party].append(ch)
            party_metas[party].append({
                "party_key": party,
                "doc_type": doctype,
                "s3_key": key,
                "title": title,
                "chunk_id": f"{key}::#{i}",
            })

    built, total_chunks = [], 0
    for party, texts in party_texts.items():
        vs = FAISS.from_texts(texts=texts, embedding=emb, metadatas=party_metas[party])
        out = settings.INDEX_DIR / "party" / party.replace(" ", "_")
        out.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(out))
        n = len(texts)
        total_chunks += n
        built.append({"party": party, "chunks": n, "path": str(out)})
        print(f"✅ Built party index for '{party}' with {n} chunks at {out}")

    return {
        "bucket": settings.S3_BUCKET,
        "base_prefix": base,
        "parties_built": len(built),
        "total_chunks": total_chunks,
        "details": built,
        "index_dir": str(settings.INDEX_DIR.resolve()),
    }

if __name__ == "__main__":
    summary = build()
    print("\nSummary:")
    for d in summary["details"]:
        print(f"- {d['party']}: {d['chunks']} chunks → {d['path']}")
    print(f"Total chunks: {summary['total_chunks']}")
