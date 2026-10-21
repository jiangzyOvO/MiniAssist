from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from miniassist.chunking.structure_v2 import structure_split_v2, is_pure
from miniassist.chunking.text import coordinates
from miniassist.chunking.units import prepare
from test_chunking import block


def release(ident, date, body):
    rows = [block(ident+'title', '产品发布记录', 0, section=ident+'title', kind='heading'),
            block(ident+'date', '发布日期：'+date, 20, section=ident+'date', kind='heading'),
            block(ident+'version', '固件版本：V1.2', 60, section=ident+'date'),
            block(ident+'update', '本次更新', 80, section=ident+'update', kind='heading'),
            block(ident+'body', body, 100, section=ident+'update', kind='list_item')]
    for b in rows:
        b['metadata']['release_date'] = date
    return rows


class StructureV2Tests(unittest.TestCase):
    def test_short_record_keeps_date_versions_and_updates_together(self):
        blocks = release('a', '2024-01-01', '修复已知问题。')
        result = structure_split_v2(prepare(blocks), blocks, 100, 10)
        self.assertEqual(len(result), 1)
        for text in ('发布日期：2024-01-01', '固件版本：V1.2', '修复已知问题。'):
            self.assertIn(text, result[0]['text'])

    def test_long_record_repeats_real_identity_in_every_continuation(self):
        blocks = release('a', '2024-01-01', '这是一项更新。'*40)
        units = prepare(blocks)
        result = structure_split_v2(units, blocks, 100, 10)
        self.assertGreater(len(result), 1)
        for c in result:
            self.assertLessEqual(len(c['text']), 100)
            self.assertIn('发布日期：2024-01-01', c['text'])
            self.assertIn('固件版本：V1.2', c['text'])
            self.assertIn('本次更新', c['text'])
        self.assertEqual(set().union(*(coordinates(c) for c in result)),
                         set().union(*(coordinates(u['value']) for u in units)))

    def test_distinct_release_dates_never_merge(self):
        blocks = release('a', '2024-01-01', '修改甲。') + release('b', '2024-02-01', '修改乙。')
        result = structure_split_v2(prepare(blocks), blocks, 500, 80)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(len(c['release_dates']) == 1 for c in result))

    def test_redundant_parent_heading_removed_without_losing_coordinates(self):
        parent = block('h', '父标题', 0, section='parent', kind='heading')
        child = block('b', '正文事实', 20, section='child')
        child['section_ids'] = ['parent', 'child']
        child['section_path'] = ['parent', 'child']
        units = prepare([parent, child])
        result = structure_split_v2(units, [parent, child], 100, 10)
        self.assertEqual(len(result), 1)
        self.assertFalse(is_pure(result[0]))
        self.assertEqual(coordinates(result[0]),
                         set().union(*(coordinates(u['value']) for u in units)))

    def test_unrelated_orphan_heading_is_preserved(self):
        blocks = [block('h', '独立标题', 0, section='a', kind='heading'),
                  block('b', '另一节内容', 20, section='b')]
        result = structure_split_v2(prepare(blocks), blocks, 100, 10)
        self.assertEqual(len(result), 2)
        self.assertTrue(any(is_pure(c) for c in result))


if __name__ == '__main__':
    unittest.main()
