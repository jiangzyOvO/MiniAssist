"""Frozen-catalog ES retrieval. No dependency on experimental runners or labels."""
import hashlib
import json
import math
import os
import re
from pathlib import Path
import requests


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_rrf(bm25, dense, alpha=0.9, k=60, limit=5):
    # Same formula/tie policy as the frozen RetrievalLab implementation.
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Invalid alpha")
    if not isinstance(k, int) or k < 0 or not isinstance(limit, int) or limit < 1:
        raise ValueError("Invalid RRF limit/k")
    scores = {}
    for ranking, weight in ((bm25, alpha), (dense, 1-alpha)):
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            ident = hit["chunk_id"]
            if ident in seen:
                raise ValueError("Duplicate chunk ID")
            seen.add(ident)
            if weight:
                scores[ident] = scores.get(ident, 0) + weight / (k+rank)
    return [{"chunk_id": ident, "score": score} for ident, score in
            sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:limit]]


class Catalog:
    def __init__(self, config):
        self.config = config
        for key in ("chunks", "sources"):
            if sha256(config[key]) != config[key + "_sha256"]:
                raise ValueError(f"Frozen {key} hash mismatch")
        self.rows = [json.loads(line) for line in Path(config["chunks"]).read_text().splitlines() if line]
        self.by_id = {row["chunk_id"]: row for row in self.rows}
        if len(self.by_id) != len(self.rows) or len(self.rows) != config["count"]:
            raise ValueError("Catalog size or unique IDs mismatch")
        self.sources = {s["source_id"]: s for s in json.loads(Path(config["sources"]).read_text())}
        for row in self.rows:
            if row["source_sha256"] != self.sources[row["source_id"]]["sha256"]:
                raise ValueError("Chunk/source provenance mismatch")

    def document(self, ident):
        doc = dict(self.by_id[ident])
        source = self.sources[doc["source_id"]]
        doc.update(url=source["url"], title=source["title"],
                   snapshot_date=source["snapshot_date"])
        return doc

    def metadata(self):
        return {k: self.config[k] for k in ("chunks_sha256", "sources_sha256", "vectors_sha256",
                                           "count", "embedding")}

    def related_release_context(self, documents, limit=2):
        """Keep conditions attached to a dated release; never cross source/date."""
        known = {d["chunk_id"] for d in documents}
        pairs = {(d["source_id"], date) for d in documents for date in d.get("release_dates", [])}
        candidates = [d for d in self.rows if d["chunk_id"] not in known
                      and any((d["source_id"], date) in pairs for date in d.get("release_dates", []))]
        # Conditions/notes take priority over another copy of feature bullets.
        candidates.sort(key=lambda d: ("注意" not in d["text"], d["chunk_id"]))
        return [self.document(d["chunk_id"]) for d in candidates[:limit]]


class ES:
    def __init__(self, url, timeout=45):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False
        if os.getenv("ES_API_KEY"):
            self.session.headers["Authorization"] = "ApiKey " + os.environ["ES_API_KEY"]

    def request(self, method, path, body=None, ndjson=None):
        kwargs = {"json": body} if ndjson is None else {
            "data": ndjson.encode(), "headers": {"Content-Type": "application/x-ndjson"}}
        result = self.session.request(method, self.url + path, timeout=(10, self.timeout),
                                      allow_redirects=False, **kwargs)
        if not result.ok:
            raise RuntimeError(f"ES HTTP {result.status_code}")
        return result.json()


class BGE:
    def __init__(self, config):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(4)
        torch.manual_seed(0)
        self.config = config["embedding"]
        self.model = SentenceTransformer(self.config["model_id"], revision=self.config["revision"],
                                         device="cpu", cache_folder=config["model_cache"],
                                         trust_remote_code=False)
        if self.model.get_sentence_embedding_dimension() != self.config["dimensions"]:
            raise ValueError("Embedding dimension mismatch")

    def encode_query(self, question):
        import numpy as np
        text = self.config["query_instruction"] + question
        ids = self.model.tokenizer(text, truncation=False)["input_ids"]
        if len(ids) > self.model.max_seq_length:
            raise ValueError("Query exceeds embedding token limit; no silent truncation")
        vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
            raise ValueError("Invalid query embedding")
        return vector.tolist()


class HybridRetriever:
    def __init__(self, config, es_url, embedder=None):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]+", config["index"]):
            raise ValueError("Invalid index name")
        self.config, self.catalog = config, Catalog(config)
        self.es = ES(es_url)
        self.index = "/" + config["index"]
        self.verify_index()
        self.embedder = embedder if embedder is not None else BGE(config)

    def verify_index(self):
        """Read all 503 docs once; verify text AND float32 vectors against frozen artifacts."""
        import numpy as np
        mapping = self.es.request("GET", self.index + "/_mapping")[self.config["index"]]["mappings"]
        if mapping.get("_meta") != self.catalog.metadata():
            raise ValueError("Runtime index provenance mismatch")
        result = self.es.request("POST", self.index + "/_search",
                                 {"size": self.config["count"] + 1, "query": {"match_all": {}},
                                  "track_total_hits": True})
        self._complete(result)
        hits = result["hits"]["hits"]
        if result["hits"]["total"]["value"] != self.config["count"] or len(hits) != self.config["count"]:
            raise ValueError("Runtime index count mismatch")
        indexed = {h["_id"]: h["_source"] for h in hits}
        if set(indexed) != set(self.catalog.by_id):
            raise ValueError("Runtime index IDs mismatch")
        vectors = []
        for row in self.catalog.rows:
            found = indexed[row["chunk_id"]]
            if any(found[k] != row[k] for k in ("chunk_id", "source_id", "text")):
                raise ValueError("Runtime index text mismatch")
            vectors.append(found["embedding"])
        array = np.asarray(vectors, dtype=np.float32)
        if hashlib.sha256(array.tobytes()).hexdigest() != self.config["vectors_sha256"]:
            raise ValueError("Runtime index vector mismatch")

    @staticmethod
    def _complete(result):
        if result.get("timed_out") or result["_shards"]["failed"]:
            raise RuntimeError("Incomplete ES search")

    def _search(self, query):
        result = self.es.request("POST", self.index + "/_search", {
            "size": self.config["candidates"], "query": query, "_source": False,
            "sort": [{"_score": "desc"}, {"chunk_id": "asc"}]})
        self._complete(result)
        return [{"chunk_id": h["_id"], "score": h["_score"]} for h in result["hits"]["hits"]]

    def search(self, question):
        vector = self.embedder.encode_query(question)
        bm25 = self._search({"match": {"text": {"query": question}}})
        dense = self._search({"script_score": {"query": {"match_all": {}}, "script": {
            "source": "cosineSimilarity(params.q, 'embedding') + 1.0", "params": {"q": vector}}}})
        fused = weighted_rrf(bm25, dense, self.config["alpha"], self.config["rrf_k"], self.config["top_k"])
        ranks = [{h["chunk_id"]: i for i, h in enumerate(route, 1)} for route in (bm25, dense)]
        documents = []
        for hit in fused:
            doc = self.catalog.document(hit["chunk_id"])
            doc.update(score=hit["score"], bm25_rank=ranks[0].get(hit["chunk_id"]),
                       dense_rank=ranks[1].get(hit["chunk_id"]))
            documents.append(doc)
        return documents

    def related_release_context(self, documents, limit=2):
        return self.catalog.related_release_context(documents, limit)
