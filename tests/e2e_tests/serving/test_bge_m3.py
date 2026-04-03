#!/usr/bin/env python3
"""
BGE-M3 embedding precision test.

Server must already be running at localhost:8000, e.g.:

    vllm serve BAAI/bge-m3 --hf-overrides '{"architectures":["BgeM3EmbeddingModel"]}'
"""

import sys

import numpy as np
import requests

BASE_URL = "http://localhost:8000"
MODEL_NAME = "BAAI/bge-m3"

SENTENCES_1 = ["What is BGE M3?", "Defination of BM25"]
SENTENCES_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, "
    "lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set "
    "of documents based on the query terms appearing in each document",
]

SIMILARITY_REFERENCE = [[0.6265, 0.3477], [0.3499, 0.678]]
LEXICAL_SCORE_REFERENCE = [0.181622, 0.0]
COLBERT_SCORE_REFERENCE = [0.7797, 0.4620]

all_passed = True


def post(path: str, payload: dict) -> dict:
    r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def cosine_sim(a, b):
    na = np.array(a) / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    nb = np.array(b) / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return (na @ nb.T).tolist()


def test_dense():
    global all_passed
    print("\n" + "=" * 60)
    print("1. Dense Embedding (cosine similarity)")
    print("=" * 60)

    emb1 = post(
        "/v1/embeddings",
        {"model": MODEL_NAME, "input": SENTENCES_1},
    )["data"]
    emb2 = post(
        "/v1/embeddings",
        {"model": MODEL_NAME, "input": SENTENCES_2},
    )["data"]

    sim = cosine_sim(
        [e["embedding"] for e in emb1],
        [e["embedding"] for e in emb2],
    )

    ref = SIMILARITY_REFERENCE
    diffs = [[abs(sim[i][j] - ref[i][j]) for j in range(2)] for i in range(2)]
    max_diff = max(max(row) for row in diffs)

    print("  similarity matrix:")
    print(f"    vLLM:     {sim[0][0]:.4f}, {sim[0][1]:.4f}")
    print(f"             {sim[1][0]:.4f}, {sim[1][1]:.4f}")
    print(f"  reference: {ref[0][0]:.4f}, {ref[0][1]:.4f}")
    print(f"             {ref[1][0]:.4f}, {ref[1][1]:.4f}")
    print(f"  max diff:  {max_diff:.6f}  (tolerance: 0.01)")

    passed = max_diff < 0.01
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    if not passed:
        all_passed = False
    return passed


def test_lexical():
    global all_passed
    print("\n" + "=" * 60)
    print("2. Lexical Sparse (BM25-style score)")
    print("=" * 60)

    tokens1 = [
        post("/tokenize", {"model": MODEL_NAME, "prompt": s})["tokens"]
        for s in SENTENCES_1
    ]
    tokens2 = [
        post("/tokenize", {"model": MODEL_NAME, "prompt": s})["tokens"]
        for s in SENTENCES_2
    ]

    sparse1 = post(
        "/pooling",
        {"model": MODEL_NAME, "input": SENTENCES_1, "task": "token_classify"},
    )["data"]
    sparse2 = post(
        "/pooling",
        {"model": MODEL_NAME, "input": SENTENCES_2, "task": "token_classify"},
    )["data"]

    def merge(tokens, vals_per_token):
        # vals_per_token: list of [val] (from /pooling data field)
        if tokens and tokens[0] == 0:
            tokens, vals_per_token = tokens[1:], vals_per_token[1:]
        d = {}
        for t, v in zip(tokens, vals_per_token):
            val = float(v[0])
            if t not in d or val > d[t]:
                d[t] = val
        return d

    def lexical(a, b):
        return sum(w * b[t] for t, w in a.items() if t in b)

    lw1 = [merge(t, s["data"]) for t, s in zip(tokens1, sparse1)]
    lw2 = [merge(t, s["data"]) for t, s in zip(tokens2, sparse2)]

    score_1_0_x_2_0 = lexical(lw1[0], lw2[0])
    score_1_0_x_1_1 = lexical(lw1[0], lw1[1])

    diff1 = abs(score_1_0_x_2_0 - LEXICAL_SCORE_REFERENCE[0])
    diff2 = abs(score_1_0_x_1_1 - LEXICAL_SCORE_REFERENCE[1])

    print(
        f"  sent1[0] vs sent2[0]: vLLM={score_1_0_x_2_0:.6f}  "
        f"ref={LEXICAL_SCORE_REFERENCE[0]:.6f}  diff={diff1:.6f}"
    )
    print(
        f"  sent1[0] vs sent1[1]: vLLM={score_1_0_x_1_1:.6f}  "
        f"ref={LEXICAL_SCORE_REFERENCE[1]:.6f}  diff={diff2:.6f}"
    )

    passed = diff1 < 0.05 * LEXICAL_SCORE_REFERENCE[0] and diff2 < 0.05
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    if not passed:
        all_passed = False
    return passed


def test_colbert():
    global all_passed
    print("\n" + "=" * 60)
    print("3. Multi-Vector ColBERT (MaxSim score)")
    print("=" * 60)

    emb1 = post(
        "/pooling",
        {"model": MODEL_NAME, "input": SENTENCES_1, "task": "token_embed"},
    )["data"]
    emb2 = post(
        "/pooling",
        {"model": MODEL_NAME, "input": SENTENCES_2, "task": "token_embed"},
    )["data"]

    def colbert(q_data, p_data):
        # token_embed: data is [[f1,f2,...,f1024], ...] — list of token vectors
        # Already flat vectors, just convert to numpy
        q = np.array(q_data)
        p = np.array(p_data)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
        p_norm = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-10)
        scores = q_norm @ p_norm.T  # (n_query_tokens, n_passage_tokens)
        return float(np.mean(np.max(scores, axis=1)))

    score_1_0_x_2_0 = colbert(emb1[0]["data"], emb2[0]["data"])
    score_1_0_x_2_1 = colbert(emb1[0]["data"], emb2[1]["data"])

    diff1 = abs(score_1_0_x_2_0 - COLBERT_SCORE_REFERENCE[0])
    diff2 = abs(score_1_0_x_2_1 - COLBERT_SCORE_REFERENCE[1])

    print(
        f"  sent1[0] vs sent2[0]: vLLM={score_1_0_x_2_0:.6f}  "
        f"ref={COLBERT_SCORE_REFERENCE[0]:.6f}  diff={diff1:.6f}"
    )
    print(
        f"  sent1[0] vs sent2[1]: vLLM={score_1_0_x_2_1:.6f}  "
        f"ref={COLBERT_SCORE_REFERENCE[1]:.6f}  diff={diff2:.6f}"
    )

    passed = (
        diff1 < 0.01 * COLBERT_SCORE_REFERENCE[0]
        and diff2 < 0.01 * COLBERT_SCORE_REFERENCE[1]
    )
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    if not passed:
        all_passed = False
    return passed


if __name__ == "__main__":
    print("BGE-M3 Embedding Precision Test")
    print(f"Server: {BASE_URL}")

    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Server status: {r.status_code} OK\n")
    except Exception:
        print("✗ Server is not running at localhost:8000")
        print(
            "  Please start with: vllm serve BAAI/bge-m3 "
            "--hf-overrides "
            '\'{"architectures":["BgeM3EmbeddingModel"]}\''
        )
        sys.exit(1)

    test_dense()
    test_lexical()
    test_colbert()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
