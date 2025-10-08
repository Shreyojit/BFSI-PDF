# mca_chatbot/routes.py
from __future__ import annotations

import json
import threading
import uuid
import re
import difflib
import itertools
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Literal, Dict, Tuple

import boto3
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .rag_config import settings
from .prompts import create_plan_prompt, create_service_prompt, Chunk as PromptChunk
from .party_normalize import canonical_party_key
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings

from .build_party_indices import build as build_party_indices

router = APIRouter()

# --------------------------
# In-memory session store
# --------------------------
_SESSIONS: Dict[str, Optional[str]] = {}
_LOCK = threading.Lock()

def _get_plan(session_id: str) -> Optional[str]:
    with _LOCK:
        return _SESSIONS.get(session_id)

def _set_plan(session_id: str, plan_key: Optional[str]) -> None:
    with _LOCK:
        _SESSIONS[session_id] = plan_key

# --------------------------
# Schemas
# --------------------------
class SessionCreateOut(BaseModel):
    session_id: str

class ChatMsgIn(BaseModel):
    session_id: str = Field(..., description="UUID from POST /chatbot/session")
    message: str = Field(..., description="User free-text")

class ChatMsgOut(BaseModel):
    session_id: str
    response: str
    current_plan: str  # 'default' if none set
    show_plan_picker: bool = False
    plan_options: List[str] = []

class PlanListOut(BaseModel):
    plans: List[str]

# --------------------------
# String utils
# --------------------------
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _tokens(s: str) -> List[str]:
    return _normalize(s).split()

def _rm_vowels(s: str) -> str:
    return re.sub(r"[aeiou]", "", s)

# Words we ignore when matching titles/inputs
_STOPWORDS = {
    "retirement", "plan", "inc", "incorporated", "manufacturing", "savings",
    "the", "a", "an", "of", "and", "llc", "ltd", "company", "corp", "corporation",
    "co", "group", "trust", "benefit", "profit", "deferred", "401", "401k", "k"
}

def _filter_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t and t not in _STOPWORDS and not t.isdigit()]

# Greetings guard (prevents "hi" from matching "knight")
_GREETING_TOKENS = {"hi", "hello", "hey", "yo", "hiya", "howdy", "sup"}

def _is_greeting_message(msgn: str) -> bool:
    toks = _tokens(msgn)
    if not toks:
        return False
    if len(toks) <= 3 and all(t in _GREETING_TOKENS for t in toks):
        return True
    if len(toks) == 1 and toks[0] in _GREETING_TOKENS:
        return True
    return False

# --------------------------
# Plans discovery (indices and S3 display names)
# --------------------------
_s3 = boto3.client("s3", region_name=settings.AWS_REGION)

def _party_index_root() -> Path:
    return (settings.INDEX_DIR / "party").resolve()

def _available_plan_keys() -> List[str]:
    root = _party_index_root()
    if not root.exists():
        return []
    names = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "index.faiss").exists():
            names.append(p.name.replace("_", " "))
    return names

_PLAN_DISPLAY_CACHE: Dict[str, str] = {}

def _title_from_key(key: str) -> str:
    base = key.rsplit("/", 1)[-1]
    if "." in base:
        base = ".".join(base.split(".")[:-1])
    return base

def _load_plan_display_cache() -> None:
    global _PLAN_DISPLAY_CACHE
    if _PLAN_DISPLAY_CACHE:
        return

    base = settings.S3_BASE_PREFIX.strip().strip("/")
    plan_prefix = f"{base}/plan/"
    token = None
    out: Dict[str, str] = {}

    while True:
        kwargs = {"Bucket": settings.S3_BUCKET, "Prefix": plan_prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = _s3.list_objects_v2(**kwargs)
        for o in resp.get("Contents", []):
            k = o["Key"]
            if not k.lower().endswith(".txt"):
                continue
            title = _title_from_key(k)
            ckey = _normalize(canonical_party_key(title))
            if ckey and ckey not in out:
                out[ckey] = title
        token = resp.get("NextContinuationToken")
        if not token:
            break

    _PLAN_DISPLAY_CACHE = out

def _display_name_for_plan_key(plan_key: str) -> str:
    _load_plan_display_cache()
    ckey = _normalize(canonical_party_key(plan_key))
    disp = _PLAN_DISPLAY_CACHE.get(ckey)
    if disp:
        return disp
    return " ".join(w.capitalize() for w in _normalize(plan_key).split())

def _list_plan_display_titles() -> List[str]:
    return [_display_name_for_plan_key(k) for k in _available_plan_keys()]

# --------------------------
# Switch intent parsing (detect-only)
# --------------------------
_SWITCH_PATTERNS = [
    re.compile(r".*\b(i\s+want\s+to\s+)?(switch|change|set|use)\b.*\bplan\b.*", re.I),
    re.compile(r".*\b(another|different)\s+plan\b.*", re.I),
    re.compile(r".*\bquestions?\s+on\s+another\s+plan\b.*", re.I),
    re.compile(r".*\bchange\s+my\s+plan\b.*", re.I),
    re.compile(r"^\s*i\s+want\s+to\s+change\s+my\s+plan\s*$", re.I),
    re.compile(r"^\s*i\s+have\s+questions\s+on\s+another\s+plan\s*$", re.I),
    re.compile(r"^\s*switch\s+my\s+plan\s*$", re.I),
    re.compile(r"^\s*change\s+plan\s*$", re.I),
    re.compile(r"^\s*use\s+a?\s*different\s+plan\s*$", re.I),
]

def _parse_switch_intent(message: str) -> bool:
    t = (message or "").strip()
    for pat in _SWITCH_PATTERNS:
        if pat.match(t):
            return True
    return False

# --------------------------
# Alias / fuzzy matching (+ subsequence & concatenation support)
# --------------------------
def _core_tokens_for_display_title(display: str) -> List[str]:
    """
    Core tokens used for matching, with extras:
    - stopwords removed (fallback to all tokens if empty)
    - concatenated core token (e.g., ["tri","state"] -> "tristate")
    - vowel-stripped concatenation ("trstt")
    """
    toks = _tokens(display)
    core = _filter_stopwords(toks) or toks
    extras: List[str] = []
    if len(core) >= 2:
        joined = "".join(core)
        extras.append(joined)
        nv = _rm_vowels(joined)
        if nv and nv != joined:
            extras.append(nv)
    return core + extras

def _aliases_for_display_title(display: str) -> List[str]:
    core = _core_tokens_for_display_title(display)
    if not core:
        core = _tokens(display)

    per_token_variants: List[List[str]] = []
    for tok in core:
        v = set()
        nv = _rm_vowels(tok)
        for n in (2, 3, 4):
            if len(tok) >= n:
                v.add(tok[:n])
        for n in (2, 3, 4):
            if len(nv) >= n:
                v.add(nv[:n])
        v.add(tok)
        if nv and nv != tok:
            v.add(nv)
        per_token_variants.append(sorted(v, key=len))

    aliases = set()
    for combo in itertools.product(*per_token_variants):
        alias = " ".join(combo)
        aliases.add(alias)

    initials = "".join(t[0] for t in core if t)
    if len(initials) >= 2:
        aliases.add(" ".join(list(initials)))
        aliases.add(initials)

    aliases.add(" ".join(core))
    aliases.add(_normalize(display))

    aliases_list = sorted(list(aliases), key=lambda s: (len(s), s))
    if len(aliases_list) > 600:
        aliases_list = aliases_list[:600] + [_normalize(display)]
    return aliases_list

def _build_alias_map(plan_keys: List[str]) -> Dict[str, str]:
    amap: Dict[str, str] = {}
    for key in plan_keys:
        display = _display_name_for_plan_key(key)
        for alias in _aliases_for_display_title(display):
            amap[alias.lower()] = key
    return amap

def _tokenwise_prefix_match(user_tokens: List[str], plan_tokens: List[str]) -> bool:
    """
    For each **non-stopword** user token, it must be a prefix of SOME plan core token
    (raw or vowel-stripped). Extra user tokens that are stopwords are ignored.
    """
    ucore = _filter_stopwords(user_tokens)
    if not ucore:
        return False
    for ut in ucore:
        hit = False
        for pt in plan_tokens:
            if pt.startswith(ut) or _rm_vowels(pt).startswith(ut):
                hit = True
                break
        if not hit:
            return False
    return True

# === Subsequence helpers ===
def _is_subsequence(needle: str, hay: str) -> bool:
    """Return True if all chars of `needle` appear in order inside `hay`."""
    i = 0
    for ch in hay:
        if i < len(needle) and needle[i] == ch:
            i += 1
            if i == len(needle):
                return True
    return len(needle) == 0

def _tokenwise_subseq_match(user_tokens: List[str], plan_tokens: List[str]) -> bool:
    """
    Require enough evidence:
    - If only ONE user token, it must be length >= 3 (so 'hi' won't trigger),
      otherwise require >=2 informative tokens (e.g., 'jo fz').
    """
    ucore = _filter_stopwords(user_tokens)
    if not ucore:
        return False

    if len(ucore) == 1 and len(ucore[0]) < 3:
        return False  # too weak (prevents 'hi' -> 'knight')

    for ut in ucore:
        ut = ut.lower()
        hit = False
        for pt in plan_tokens:
            pt_raw = pt.lower()
            pt_vow = _rm_vowels(pt_raw)
            if _is_subsequence(ut, pt_raw) or _is_subsequence(ut, pt_vow):
                hit = True
                break
        if not hit:
            return False
    return True
# === End subsequence helpers ===

def _best_fuzzy_plan_match(msg: str, plan_keys: List[str]) -> Optional[str]:
    msgn = _normalize(msg)
    if not msgn:
        return None
    utoks = _tokens(msgn)
    ucore = _filter_stopwords(utoks)
    ucore_str = " ".join(ucore)

    # Enough signal for subseq boost?
    allow_subseq_boost = (len(ucore) >= 2) or (len(ucore) == 1 and len(ucore[0]) >= 3)

    candidates: List[Tuple[float, str]] = []
    for key in plan_keys:
        disp = _display_name_for_plan_key(key)
        dispn = _normalize(disp)
        core = _core_tokens_for_display_title(disp)

        ratio = difflib.SequenceMatcher(a=msgn, b=dispn).ratio()

        # Strong signals:
        if _tokenwise_prefix_match(utoks, core):
            ratio = max(ratio, 0.96)

        # Subsequence tokenwise match (e.g., "jo fz" -> "joe frazier")
        if allow_subseq_boost and _tokenwise_subseq_match(utoks, core):
            ratio = max(ratio, 0.965)

        # If user's core phrase is contained in display (e.g., "tri state" ⊂ title)
        if ucore_str and ucore_str in dispn:
            ratio = max(ratio, 0.95)
        # If display core string is contained in user's input
        core_str = " ".join(core)
        if core_str and core_str in msgn:
            ratio = max(ratio, 0.93)

        candidates.append((ratio, key))

    best_ratio, best_key = max(candidates, key=lambda x: x[0]) if candidates else (0.0, None)
    return best_key if best_ratio >= 0.80 else None

def _match_plan_from_text(msg: str, plan_keys: List[str]) -> Optional[str]:
    if not msg:
        return None
    msg_l = _normalize(msg)

    # If it's just a greeting, do NOT attempt to match a plan
    if _is_greeting_message(msg_l):
        return None

    # 1) Exact normalized display title
    lower_to_key = { _display_name_for_plan_key(k).lower(): k for k in plan_keys }
    if msg_l in lower_to_key:
        return lower_to_key[msg_l]

    # 2) Alias map
    alias_map = _build_alias_map(plan_keys)
    if msg_l in alias_map:
        return alias_map[msg_l]

    # 3) Collapsed spaces
    msg_collapse = " ".join(_tokens(msg_l))
    if msg_collapse in alias_map:
        return alias_map[msg_collapse]

    # 3.5) Subsequence tokenwise match against plan core tokens (only with enough signal)
    utoks = _tokens(msg_l)
    ucore = _filter_stopwords(utoks)
    allow_subseq = (len(ucore) >= 2) or (len(ucore) == 1 and len(ucore[0]) >= 3)
    if allow_subseq:
        for k in plan_keys:
            core = _core_tokens_for_display_title(_display_name_for_plan_key(k))
            if _tokenwise_subseq_match(utoks, core):
                return k

    # 4) Substring contains on normalized forms (handles partial plan names, ignoring spaces)
    for k in plan_keys:
        dispn = _normalize(_display_name_for_plan_key(k))
        if (msg_collapse in dispn or dispn in msg_collapse or
            msg_collapse.replace(" ", "") in dispn.replace(" ", "") or
            dispn.replace(" ", "") in msg_collapse.replace(" ", "")):
            return k

    # 5) Fuzzy/tokenwise fallback
    return _best_fuzzy_plan_match(msg, plan_keys)

# --------------------------
# Retrieval & LLM
# --------------------------
def _index_path_for_plan(plan_key: str) -> Path:
    folder = _normalize(plan_key).replace(" ", "_")
    return (_party_index_root() / folder).resolve()

def _load_vectorstore(plan_key: str) -> FAISS:
    idx = _index_path_for_plan(plan_key)
    if not idx.exists():
        raise HTTPException(status_code=404, detail=f"Index not found for plan '{plan_key}'")
    emb = BedrockEmbeddings(region_name=settings.AWS_REGION, model_id=settings.EMBEDDING_MODEL_ID)
    return FAISS.load_local(str(idx), embeddings=emb, allow_dangerous_deserialization=True)

def _retrieve_chunks(plan_key: str, query: str, k: int = 20) -> List[PromptChunk]:
    vs = _load_vectorstore(plan_key)
    docs = vs.similarity_search(query, k=k)
    return [PromptChunk(text=d.page_content, metadata=d.metadata or {}) for d in docs]

def _classify_query_type(q: str) -> Literal["plan", "service"]:
    t = q.lower()
    service_triggers = [
        "fee", "recordkeeping", "hardship", "optional service", "qdia",
        "additional plan services", "effective date", "schedule", "pricing", "service"
    ]
    return "service" if any(w in t for w in service_triggers) else "plan"

def _bedrock_llm_answer(prompt: str) -> str:
    if settings.LLM_MODEL_ID.lower().startswith("mock:"):
        return "Result\n[Mock LLM enabled — replace LLM_MODEL_ID to use Bedrock]"

    client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }

    model_id_or_profile = settings.INFERENCE_PROFILE_ARN.strip() or settings.LLM_MODEL_ID

    try:
        resp = client.invoke_model(
            modelId=model_id_or_profile,
            body=json.dumps(body).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )
        payload = resp["body"].read().decode("utf-8")
        data = json.loads(payload)
        for block in (data.get("content") or []):
            if block.get("type") == "text" and "text" in block:
                return block["text"].strip()
        return (data.get("output_text") or "").strip() or "Result\nNo answer returned by the model."
    except Exception as e:
        return f"Result\nLLM error: {e}"

def _current_plan_or_default(plan_key: Optional[str]) -> str:
    if not plan_key:
        return "default"
    return _display_name_for_plan_key(plan_key)

# --------------------------
# Persist retrieved chunks for answers (TEXT FILE)
# --------------------------
def _format_chunk_block(idx: int, ch: PromptChunk) -> str:
    md = ch.metadata or {}
    md_lines = []
    for k in ["party_key", "doc_type", "s3_key", "title", "chunk_id"]:
        if k in md:
            md_lines.append(f"{k}: {md[k]}")
    for k in sorted(set(md.keys()) - {"party_key", "doc_type", "s3_key", "title", "chunk_id"}):
        md_lines.append(f"{k}: {md[k]}")
    meta_block = "\n".join(md_lines) if md_lines else "(no metadata)"
    text = ch.text.strip()
    return (
        f"---- Chunk {idx+1} ----\n"
        f"{meta_block}\n\n"
        f"{text}\n"
    )

def _save_chunks_for_answer(
    session_id: str,
    plan_key: str,
    qtype: Literal["plan", "service"],
    query: str,
    chunks: List[PromptChunk],
) -> str:
    plan_folder = _normalize(plan_key).replace(" ", "_")
    out_dir = (settings.CHUNKS_FOR_ANS_DIR / plan_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_utc = datetime.now(timezone.utc).isoformat()
    short_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    qhash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    txt_fname = f"{short_ts}_{qtype}_{qhash}.txt"
    txt_fpath = out_dir / txt_fname

    header = (
        "=== Retrieved Chunks Used For Answer ===\n"
        f"timestamp_utc: {ts_utc}\n"
        f"session_id: {session_id}\n"
        f"plan_key: {plan_key}\n"
        f"plan_display: {_display_name_for_plan_key(plan_key)}\n"
        f"query_type: {qtype}\n"
        f"query: {query}\n"
        "========================================\n\n"
    )

    body_blocks = [_format_chunk_block(i, ch) for i, ch in enumerate(chunks)]
    content = header + "\n".join(body_blocks)

    txt_fpath.write_text(content, encoding="utf-8")
    return str(txt_fpath)

# --------------------------
# Routes
# --------------------------
@router.get("/health")
def health():
    return {"status": "ok", "service": "chatbot"}

@router.get("/build-indices")
def build_indices():
    build_party_indices()
    return {"status": "built", "plans": _available_plan_keys()}

@router.get("/plans", response_model=PlanListOut)
def list_plans():
    return PlanListOut(plans=_list_plan_display_titles())

@router.post("/session", response_model=SessionCreateOut)
def create_session():
    sid = str(uuid.uuid4())
    _set_plan(sid, None)
    return SessionCreateOut(session_id=sid)

@router.post("/chat", response_model=ChatMsgOut)
def chat(body: ChatMsgIn):
    plan_keys = _available_plan_keys()
    if not plan_keys:
        raise HTTPException(status_code=404, detail="No plan indices available. Build indices first.")

    msg = (body.message or "").strip()
    if not msg:
        return ChatMsgOut(
            session_id=body.session_id,
            response="Please enter a message.",
            current_plan=_current_plan_or_default(_get_plan(body.session_id))
        )

    active_key = _get_plan(body.session_id)

    # 1) Any switch intent? -> show pills
    if _parse_switch_intent(msg):
        display_titles = _list_plan_display_titles()
        return ChatMsgOut(
            session_id=body.session_id,
            response="Alright! Please start by providing me a Plan Name. You can either type a few words of the Plan Name, or select from the below last 3 Plans that you searched for.",
            current_plan=_current_plan_or_default(active_key),
            show_plan_picker=True,
            plan_options=display_titles
        )

    # 2) Did the user type something resembling a plan?
    chosen_from_text = _match_plan_from_text(msg, plan_keys)
    if chosen_from_text:
        _set_plan(body.session_id, chosen_from_text)
        display = _display_name_for_plan_key(chosen_from_text)
        return ChatMsgOut(
            session_id=body.session_id,
            response=f"Great! I have set the context to {display}; all queries hereafter will be answered accordingly.",
            current_plan=display,
            show_plan_picker=False,
            plan_options=[]
        )

    # 3) If no plan set yet, ask to pick
    if not active_key:
        display_titles = _list_plan_display_titles()
        return ChatMsgOut(
            session_id=body.session_id,
            response="Alright! Please start by providing me a Plan Name. You can either type a few words of the Plan Name, or select from the below last 3 Plans that you searched for.",
            current_plan="default",
            show_plan_picker=True,
            plan_options=display_titles
        )

    # 4) Normal Q&A within active plan
    qtype: Literal["plan", "service"] = _classify_query_type(msg)
    retrieved_chunks: List[PromptChunk] = _retrieve_chunks(active_key, msg, k=10)
    chunks: List[PromptChunk] = list(retrieved_chunks)

    if not chunks:
        return ChatMsgOut(
            session_id=body.session_id,
            response="Result\nNo relevant context found.",
            current_plan=_current_plan_or_default(active_key),
        )

    _save_chunks_for_answer(
        session_id=body.session_id,
        plan_key=active_key,
        qtype=qtype,
        query=msg,
        chunks=chunks,
    )

    pre_prompt = create_service_prompt(msg, chunks) if qtype == "service" else create_plan_prompt(msg, chunks)

    # Demo snippet (optional override for certain plans)
    demo_chunk_text = None
    plan_display_norm = _normalize(active_key)
    if plan_display_norm == "tri state":
        demo_chunk_text = "The effective date of the active Annual Schedule of Fees is 1/1/2018."
    elif plan_display_norm == "joe frazier":
        demo_chunk_text = "The effective date of the fees for Joe Frazier is 2/1/2018."
    elif plan_display_norm == "knight train":
        demo_chunk_text = "The effective date of the fees for Knight Train is 1/1/2023."

    prompt = (demo_chunk_text + "\n\n" + pre_prompt) if demo_chunk_text else pre_prompt
    print("FINAL PROMPT TO LLM:--------->", prompt)

    answer = (_bedrock_llm_answer(prompt).strip() or "Result\nNo answer could be generated from the provided context.")
    return ChatMsgOut(
        session_id=body.session_id,
        response=answer,
        current_plan=_current_plan_or_default(active_key),
    )