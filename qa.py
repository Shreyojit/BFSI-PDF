import json
from typing import List, Dict, Any

import faiss
import numpy as np

from .aws_clients import bedrock_rt_client
from .config import get_settings

S = get_settings()

ALLOWED_QUESTIONS = [
    # Fees & documents (custom)
    "What is the effective date of the active fee schedule?",
    "What is the Annual Recordkeeping Fee?",
    "What optional services were elected?",
    "What optional services were elected that have fees associated with them?",
    "Did the client elect Hardship approval and what fee applies?",
    "List the plan documents for the employer plan.",
    "List the plan documents for the employer plan with document effective date.",

    # 401(k) domain subset
    "Who is the Employer and what are the Employer's address and EIN?",
    "What is the Plan name and Plan number?",
    "What is the Plan Year end date and the Limitation Year selection?",
    "What is the Normal Retirement Age and is an Early Retirement Age specified?",
    "Are Catch-Up Deferrals permitted under the plan?",
    "Are Roth Elective Deferrals allowed?",
    "Is Automatic Enrollment selected? If yes, what is the default deferral percentage?",
    "What are the eligibility service requirements for Employee elective deferrals?",
    "What are the eligibility requirements and entry dates for Employer Matching Contributions?",
    "What is the Employer Matching Contribution formula and any true-up provisions?",
    "Are Safe Harbor contributions selected? If yes, which option applies?",
    "Are Profit Sharing contributions allowed and what is the allocation method?",
    "How is Compensation defined for contribution and testing purposes?",
    "Are Hardship Withdrawals permitted?",
    "Are Participant loans permitted under the plan?",
    "Are In-Service distributions allowed at age 59 1/2?",
    "What is the vesting schedule for Employer contributions?",
    "How are Rollover Contributions handled?",
]


def _embed_query(text: str) -> List[float]:
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


def search(index: faiss.Index, meta: List[Dict[str, Any]], query: str, k: int = 5):
    qv = np.array(_embed_query(query), dtype=np.float32).reshape(1, -1)
    D, I = index.search(qv, min(k, len(meta)))
    hits = [meta[i] for i in I.flatten() if 0 <= i < len(meta)]
    return hits


def answer_with_claude(query: str, hits: List[Dict[str, Any]]) -> str:
    ctx_parts = []
    for h in hits:
        ctx_parts.append(f"[Page {h['page']}] {h['text']}")
    context_text = ("\n\n---\n\n".join(ctx_parts))[:20000]

    system = (
        "You are a helpful assistant that answers strictly from the provided context. "
        "If the answer is not present, say you don't know. Include page numbers when referring to sources."
    )
    user_text = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

    rt = bedrock_rt_client()
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system,
        "max_tokens": 600,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
    }
    resp = rt.invoke_model(
        modelId=S.LLM_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    out = json.loads(resp.get("body").read())
    parts = out.get("content", [])
    return "".join([p.get("text", "") for p in parts if p.get("type") == "text"]).strip()
