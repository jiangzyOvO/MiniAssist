"""Create a separate runtime index from frozen documents and cached document vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from miniassist.rag.retriever import Catalog, ES, HybridRetriever
from miniassist.rag.settings import Settings, load_config, load_local_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    load_local_env()
    config, settings = load_config(args.config), Settings.from_env()
    catalog = Catalog(config)
    with np.load(config["vectors"], allow_pickle=False) as saved:
        vectors = saved["vectors"]
    if (vectors.shape != (config["count"], config["embedding"]["dimensions"])
            or hashlib.sha256(vectors.tobytes()).hexdigest() != config["vectors_sha256"]):
        raise ValueError("Frozen document vector hash/shape mismatch")
    if not args.verify_only:
        es = ES(settings.es_url)
        # ES PUT refuses to overwrite an existing index. No delete/rebuild fallback.
        es.request("PUT", "/" + config["index"], {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0,
                         "similarity": {"default": {"type": "BM25", "k1": 1.2, "b": 0.75}}},
            "mappings": {"dynamic": "strict", "_meta": catalog.metadata(), "properties": {
                "chunk_id": {"type": "keyword"}, "source_id": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "cjk"},
                "embedding": {"type": "dense_vector", "dims": config["embedding"]["dimensions"], "index": False},
            }}})
        for start in range(0, len(catalog.rows), 100):
            lines = []
            for i in range(start, min(start + 100, len(catalog.rows))):
                row = catalog.rows[i]
                lines.append(json.dumps({"create": {"_index": config["index"], "_id": row["chunk_id"]}}))
                doc = {k: row[k] for k in ("chunk_id", "source_id", "text")}
                doc["embedding"] = vectors[i].tolist()
                lines.append(json.dumps(doc, ensure_ascii=False))
            result = es.request("POST", "/_bulk", ndjson="\n".join(lines) + "\n")
            if result.get("errors"):
                raise RuntimeError("Runtime bulk import failed; partial index retained for inspection")
        es.request("POST", "/" + config["index"] + "/_refresh")
        es.request("PUT", "/" + config["index"] + "/_settings", {"index.blocks.write": True})
    # A dummy embedder avoids loading model weights during index verification.
    HybridRetriever(config, settings.es_url, embedder=object())
    print(json.dumps({"index": config["index"], "documents": config["count"], "verified": True}))


if __name__ == "__main__":
    main()
