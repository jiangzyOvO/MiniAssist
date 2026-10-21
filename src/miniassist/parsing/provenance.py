"""Map emitted text to the frozen extraction without modifying gold offsets."""
from difflib import SequenceMatcher
import re

from .models import sha256


def compact(text):
    return "".join(c for c in text if not c.isspace())


def ranges(indices):
    ordered = sorted(set(indices))
    out = []
    for i in ordered:
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


class PageMapper:
    """Only exact character matches receive provenance; no fuzzy substitution.

    Whitespace is ignored for alignment. A block may have disjoint source ranges
    (table cells / PDF reading-order differences). Unmatched characters are
    reported, never claimed as recovered evidence.
    """
    def __init__(self, source_id, page, text):
        self.source_id, self.page, self.text = source_id, page, text
        self.positions = [i for i, c in enumerate(text) if not c.isspace()]
        self.normalized = "".join(text[i] for i in self.positions)
        self.used = set()
        self.hash = sha256(text.encode())

    def map(self, output, claim=True):
        op = [i for i, c in enumerate(output) if not c.isspace()]
        needle = "".join(output[i] for i in op)
        if not needle:
            return {"source_spans": [], "text_map": [], "matched_chars": 0, "total_chars": 0}
        hits = [m.start() for m in re.finditer(re.escape(needle), self.normalized)]
        def standalone(hit):
            start, end = self.positions[hit], self.positions[hit+len(needle)-1]+1
            before = self.text[self.text.rfind("\n", 0, start)+1:start]
            next_newline = self.text.find("\n", end)
            after = self.text[end:next_newline if next_newline >= 0 else len(self.text)]
            return not before.strip() and not after.strip()

        # A heading or figure label must not steal the same phrase/numeral from
        # a later paragraph when an actual standalone source line exists.
        standalone_hits = [h for h in hits if standalone(h)]
        free_standalone = [h for h in standalone_hits
                           if all(self.positions[h+i] not in self.used for i in range(len(needle)))]
        if free_standalone:
            hits = free_standalone
        if len(needle) < 4 and re.fullmatch(r"[\d.%+-]+", needle) and not free_standalone:
            hits = []
            pairs = []
        # Prefer unclaimed copies of repeated standalone lines.
        elif hits:
            best = min(hits, key=lambda h: sum(self.positions[h+i] in self.used for i in range(len(needle))))
            pairs = [(op[i], self.positions[best+i]) for i in range(len(needle))
                     if self.positions[best+i] not in self.used]
        else:
            available = "".join("\x00" if pos in self.used else c
                                for pos, c in zip(self.positions, self.normalized))
            pairs = []
            for match in SequenceMatcher(None, needle, available, autojunk=False).get_matching_blocks():
                if match.size >= 4:
                    pairs.extend((op[match.a+i], self.positions[match.b+i]) for i in range(match.size))
        if claim:
            self.used.update(b for _, b in pairs)
        maps = []
        for out_i, src_i in pairs:
            if maps and maps[-1]["text_end"] == out_i and maps[-1]["source_end"] == src_i:
                maps[-1]["text_end"] += 1
                maps[-1]["source_end"] += 1
            else:
                maps.append(dict(text_start=out_i, text_end=out_i+1,
                                 source_start=src_i, source_end=src_i+1))
        spans = []
        # Include only whitespace gaps between consecutive mapped source chars.
        for start, end in ranges(b for _, b in pairs):
            if spans and not self.text[spans[-1]["end_char"]:start].strip():
                spans[-1]["end_char"] = end
                spans[-1]["exact_text"] = self.text[spans[-1]["start_char"]:end]
            else:
                spans.append({
                    "source_id": self.source_id, "page": self.page,
                    "start_char": start, "end_char": end, "exact_text": self.text[start:end],
                    "page_text_sha256": self.hash,
                })
        return {"source_spans": spans, "text_map": maps,
                "matched_chars": len(pairs), "total_chars": len(op)}

    def unclaimed(self):
        """Archive, but don't silently treat, unparsed canonical text as usable."""
        return ranges(i for i in self.positions if i not in self.used)
