"""Run the real HTML adapter and three splitters on a synthetic local example."""
import hashlib
import json
from pathlib import Path
from miniassist.parsing.html_parser import parse_html
from miniassist.chunking.units import prepare
from miniassist.chunking.splitters import split

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT/'examples/support_demo.html'
    source = {'source_id': 'DEMO', 'version': 'synthetic-example-v1',
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    document = parse_html(path, source, (ROOT/'examples/support_demo.txt').read_text())
    blocks = document['blocks']
    units = prepare(blocks)
    output = ROOT/'outputs/parse_chunk_demo'
    output.mkdir(parents=True, exist_ok=True)
    (output/'parsed.json').write_text(json.dumps(document, ensure_ascii=False, indent=2)+'\n')
    counts = {}
    for method in ('fixed', 'recursive', 'structure'):
        chunks = split(units, blocks, method, 40, 8)
        (output/f'{method}.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in chunks))
        counts[method] = len(chunks)
    print(json.dumps({'synthetic_example': True, 'blocks': len(blocks), 'chunks': counts,
                      'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
