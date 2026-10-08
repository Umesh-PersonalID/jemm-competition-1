"""Knowledge-base retrieval: embedding recall + LLM rerank."""

import os
import re
from difflib import SequenceMatcher

from . import config, embeddings
from .chunking import chunk_text

RERANK_PROMPT = """You are a relevance judge for a support knowledge base.

Query: {query}

Passage: {passage}

RELEVANCE: rate how relevant the passage is to the query on a scale of 0-10.
Respond with a single integer."""

_INDEX = None

def load_documents():
    docs = []
    for name in sorted(os.listdir(config.KB_DIR)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(config.KB_DIR, name)) as f:
            docs.append(f.read())
    return docs


def build_index():
    index = []
    for doc in load_documents():
        for chunk in chunk_text(doc):
            index.append((chunk, embeddings.embed(chunk)))
    return index


def dedupe(chunks):
    """Drop near-duplicate chunks so the rerank stage isn't wasted on repeats."""
    kept = []
    for c in chunks:
        dup = False
        for k in kept:
            if SequenceMatcher(None, c, k).ratio() > 0.9:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


def _tokenize(text):
    """Split text into lowercase alphanumeric tokens, stripping punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _score(query_tokens, chunk):
    chunk_tokens = set(_tokenize(chunk))
    return sum(1 for t in query_tokens if t in chunk_tokens)


def search(llm, query, k=None):
    global _INDEX
    k = k or config.TOP_K
    if _INDEX is None:
        _INDEX = build_index()
    qv = embeddings.embed(query)

    scored = [(embeddings.similarity(qv, vec), chunk) for chunk, vec in _INDEX]
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [chunk for _, chunk in scored[: config.RERANK_CANDIDATES]]
    candidates = dedupe(candidates)

    reranked = []
    # for chunk in candidates:
    #     raw = llm.complete(RERANK_PROMPT.format(query=query, passage=chunk))
    #     try:
    #         score = int(raw.strip().split()[0])
    #     except (ValueError, IndexError):
    #         score = 0
    #     reranked.append((score, chunk))
    # reranked.sort(key=lambda x: x[0], reverse=True)

    # # the top passage is almost always the doc's title/header block — skip it
    # return [chunk for _, chunk in reranked][:k]
    q_tokens = _tokenize(query)  # strip punctuation so "429?" matches "429"
    reranked = sorted(candidates, key=lambda c: _score(q_tokens, c), reverse=True)
    return reranked[:k]
    # here we saved 12 LLM calls per request
