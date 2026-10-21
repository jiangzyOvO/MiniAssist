import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from miniassist.parsing.baseline import select_blocks, verify
from miniassist.parsing.models import sha256
from miniassist.parsing.pipeline import run


def block(ident='b', warnings=None, table=None):
    return {'block_id':ident, 'block_type':'table' if table else 'paragraph', 'page':1,
            'text':'正文', 'text_map':[{'text_start':0,'text_end':2}], 'warnings':warnings or [],
            'table':table, 'bbox':[0,0,100,10], 'metadata':{'source_id':'S1'}}


def policy():
    return {'version':'test','table_reviews':[], 'excluded_sources':{},
            'excluded_pages':{},'excluded_regions':[]}


class BaselineTests(unittest.TestCase):
    def test_partial_block_is_not_silently_admitted(self):
        included, excluded = select_blocks({'S1':{'blocks':[block(warnings=['partial_source_alignment'])]}}, policy())
        self.assertFalse(included)
        self.assertIn('partial_source_alignment', excluded[0]['reasons'])

    def test_unreviewed_table_cannot_be_admitted_even_with_exact_text(self):
        t={'cells':[], 'footnote_block_ids':[], 'rows':[]}
        included, excluded = select_blocks({'S1':{'blocks':[block(table=t)]}}, policy())
        self.assertFalse(included)
        self.assertIn('table_semantics_not_reviewed', excluded[0]['reasons'])

    def test_required_footnote_must_also_be_eligible(self):
        p=policy()
        p['table_reviews']=[{'block_id':'b','text_sha256':sha256('正文'.encode()),
                             'header_rows':[], 'header_status':'reviewed_key_value'}]
        t={'cells':[], 'footnote_block_ids':['note'], 'rows':[]}
        included, excluded=select_blocks({'S1':{'blocks':[block(table=t),block('note',['partial_source_alignment'])]}},p)
        self.assertFalse(included)
        self.assertIn('required_note_not_eligible',excluded[-1]['reasons'])

    def test_stale_table_review_fails(self):
        p=policy()
        p['table_reviews']=[{'block_id':'b','text_sha256':'wrong'}]
        with self.assertRaisesRegex(ValueError,'Stale table review'):
            select_blocks({'S1':{'blocks':[block(table={'cells':[], 'rows':[]})]}},p)

    def test_integrity_check_detects_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'corpus.jsonl').write_text('original')
            (root/'freeze_manifest.json').write_text(json.dumps({
                'artifacts':{'corpus.jsonl':sha256(b'original')},'external_inputs':{}}))
            self.assertEqual(verify(root,root)['status'],'verified')
            (root/'corpus.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen baseline changed'):
                verify(root,root)

    def test_parser_cannot_overwrite_frozen_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'freeze_manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Cannot overwrite'):
                run(root,root/'parsed')


if __name__=='__main__':
    unittest.main()
