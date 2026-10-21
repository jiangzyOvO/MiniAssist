"""Build local document vectors without copying a research workspace cache.

The local config records the actual float32 hash; it does not replace a frozen
benchmark config or promise bitwise reproducibility across platforms.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from miniassist.rag.retriever import BGE, Catalog
from miniassist.rag.settings import ROOT, load_config


def encode_documents(catalog, encoder):
    texts = [row['text'] for row in catalog.rows]
    tokens = encoder.model.tokenizer(texts, add_special_tokens=True,
                                     truncation=False, padding=False)['input_ids']
    lengths = [len(ids) for ids in tokens]
    if not lengths or len(lengths) != len(texts):
        raise ValueError('Empty corpus or invalid tokenizer output')
    if max(lengths) > encoder.model.max_seq_length:
        raise ValueError('Document exceeds embedding token limit; no silent truncation')
    # Only queries receive query_instruction. Documents use the frozen chunk text.
    vectors = np.asarray(encoder.model.encode(texts, batch_size=32, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
    expected = (len(texts), catalog.config['embedding']['dimensions'])
    if vectors.shape != expected or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding shape or values')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError('Embedding normalization failed')
    return vectors, max(lengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/rag.v1.json')
    parser.add_argument('--output-config', type=Path, default=ROOT/'config/rag.local.json')
    parser.add_argument('--model-cache', type=Path, help='Optional existing model cache; default .cache/models')
    args = parser.parse_args()
    config = load_config(args.config)
    local = json.loads(args.config.read_text())
    if args.model_cache:
        config['model_cache'] = str(args.model_cache.resolve())
        local['model_cache'] = config['model_cache']
    target = Path(config['vectors'])
    if target.exists() or args.output_config.exists():
        raise FileExistsError('Vectors or local config already exist; use them or choose new output paths')
    catalog = Catalog(config)
    vectors, max_tokens = encode_documents(catalog, BGE(config))
    digest = hashlib.sha256(vectors.tobytes()).hexdigest()
    local['vectors_sha256'] = digest
    local['index'] = 'miniassist-demo-' + digest[:12]
    target.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        np.savez_compressed(stream, vectors=vectors)
    with args.output_config.open('x', encoding='utf-8') as stream:
        json.dump(local, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'documents': len(vectors), 'max_tokens': max_tokens,
                      'vectors_sha256': digest, 'config': str(args.output_config),
                      'index': local['index']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
