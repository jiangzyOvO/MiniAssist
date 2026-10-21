"""Standalone release assets, document embedding validation and category scope."""
import copy
import hashlib
import importlib.util
import json
from types import SimpleNamespace
import unittest
import numpy as np
from miniassist.rag.settings import ROOT, load_config
from miniassist.rag.retriever import Catalog
from miniassist.rag.table_evidence import table_rows, validate_category_models
from miniassist.rag.table_categories import category_context
from miniassist.rag.errors import CitationError

spec = importlib.util.spec_from_file_location('prepare_vectors', ROOT/'scripts/prepare_vectors.py')
prepare_vectors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_vectors)


class ReleaseAssetsTests(unittest.TestCase):
    def test_entire_catalog_and_table_registry_match(self):
        catalog = Catalog(load_config())
        rows = [r for doc in catalog.rows for r in table_rows(doc)]
        self.assertEqual(len(catalog.rows), 503)
        self.assertTrue(rows)
        self.assertTrue(any(r.get('category_context', {}).get('anchors') for r in rows))

    def test_release_core_matches_frozen_export_manifest(self):
        manifest = json.loads((ROOT/'docs/release_manifest.json').read_text())
        for name, expected in {**manifest['copied_core_sha256'], **manifest['frozen_assets_sha256']}.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)

    def test_paths_are_inside_release_and_sources_have_no_private_path(self):
        config = load_config()
        for key in ('chunks', 'sources', 'vectors', 'model_cache'):
            from pathlib import Path
            self.assertTrue(Path(config[key]).is_relative_to(ROOT))
        self.assertTrue(all('local_path' not in s for s in json.loads((ROOT/config['sources']).read_text())))


class EmbeddingPreparationTests(unittest.TestCase):
    def fixture(self, lengths=(2, 3), vectors=None):
        catalog = SimpleNamespace(rows=[{'text': '甲'}, {'text': '乙'}],
                                  config={'embedding': {'dimensions': 2}})
        self.encoded = []
        def encode(texts, **kwargs):
            self.encoded.extend(texts)
            return np.eye(2) if vectors is None else vectors
        model = SimpleNamespace(max_seq_length=5, encode=encode,
                                tokenizer=lambda *a, **kw: {'input_ids': [[1]*n for n in lengths]})
        return catalog, SimpleNamespace(model=model)

    def test_documents_keep_original_text_and_use_float32(self):
        vectors, count = prepare_vectors.encode_documents(*self.fixture())
        self.assertEqual(self.encoded, ['甲', '乙'])
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(count, 3)

    def test_overlong_document_rejected_before_encode(self):
        with self.assertRaisesRegex(ValueError, 'no silent truncation'):
            prepare_vectors.encode_documents(*self.fixture(lengths=(2, 6)))
        self.assertEqual(self.encoded, [])

    def test_wrong_shape_nan_and_unnormalized_vectors_rejected(self):
        for values in (np.ones((2, 3)), np.array([[np.nan, 0], [0, 1]]), np.ones((2, 2))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                prepare_vectors.encode_documents(*self.fixture(vectors=values))


class CategoryScopeTests(unittest.TestCase):
    def fixture(self):
        canonical = {('DEMO', 1): '相机存储'}
        table = {'row_count': 4, 'rows': [
            {'row': 0, 'kind': 'category', 'category_path': ['相机']},
            {'row': 1, 'category_path': ['相机']},
            {'row': 2, 'kind': 'category', 'category_path': ['存储']},
            {'row': 3, 'category_path': ['存储']}], 'cells': [
            {'row': 0, 'column': 0, 'role': 'cell', 'cell_id': 'h0', 'text': '相机',
             'text_map': [{'text_start': 0, 'text_end': 2, 'source_start': 0, 'source_end': 2}]},
            {'row': 2, 'column': 0, 'role': 'cell', 'cell_id': 'h2', 'text': '存储',
             'text_map': [{'text_start': 0, 'text_end': 2, 'source_start': 2, 'source_end': 4}]}]}
        return {'block_id': 't1', 'page': 1, 'metadata': {'source_id': 'DEMO', 'source_sha256': 'demo'}, 'table': table}, canonical

    def test_title_scope_stops_at_next_category(self):
        block, canonical = self.fixture()
        first = category_context(block, 1, canonical)['anchors'][0]
        second = category_context(block, 3, canonical)['anchors'][0]
        self.assertEqual(first['scope']['data_row_end_exclusive'], 2)
        self.assertEqual(first['text'], '相机')
        self.assertEqual(second['text'], '存储')

    def test_metadata_only_category_cannot_create_source(self):
        block, canonical = self.fixture()
        block['table']['rows'][1]['category_path'] = ['虚构类别']
        result = category_context(block, 1, canonical)
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['anchors'], [])

    def test_modified_title_cannot_pass_original_mapping(self):
        block, canonical = self.fixture()
        with self.assertRaisesRegex(ValueError, 'mapping mismatch'):
            category_context(block, 1, {('DEMO', 1): '错误存储'})

    def test_model_claim_needs_its_own_reference_support(self):
        with self.assertRaises(CitationError):
            validate_category_models('型号：RC123', [], [{'quote': '屏幕尺寸'}])
        validate_category_models('型号：RC123', [], [{'quote': '型号：RC123'}])


if __name__ == '__main__':
    unittest.main()
