#!/usr/bin/env python3
"""
build_field_registry.py - dump every fillable slot in a docx to a reviewable CSV.

Covers three kinds of slot:
  - legacy_form_field   : FORMTEXT/FORMCHECKBOX/FORMDROPDOWN fields
  - literal_placeholder : "---", "xxxxx", "____" etc. NOT inside a form field
  - choice_cell         : cells like "AC / DC / other" with no form field

Each row gets a stable field_id you can hand-edit in the CSV (rename the
auto-generated ones to meaningful names like TD_URES) and re-use as the
join key against the old-report registry and the mapping/rules file.

Digits are masked by default in printed label/context columns (--no-mask
to disable) - the structural columns (location, counts, flags) are never
masked.

Usage:
    python build_field_registry.py template.docx --role template --out template_fields.csv
    python build_field_registry.py old_report.docx --role old --out old_fields.csv
"""
import argparse
import csv
import re
import sys
import zipfile
from collections import Counter
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
q = lambda tag: f"{{{W}}}{tag}"

MASK = True
PLACEHOLDER_RE = re.compile(r"-{3,}|x{3,}|X{3,}|_{3,}|\.{4,}|…{2,}")


def m(s, n=100):
    s = re.sub(r"\s+", " ", s or "").strip()
    if MASK:
        s = re.sub(r"\d", "#", s)
    return s if len(s) <= n else s[: n - 1] + "…"


def load_parts(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    parts = {n: z.read(n) for n in names if n.startswith("word/") and n.endswith(".xml")}
    return names, parts


def run_text(r):
    out = []
    for el in r:
        if el.tag == q("t"):
            out.append(el.text or "")
        elif el.tag == q("tab"):
            out.append("\t")
        elif el.tag in (q("br"), q("cr")):
            out.append("\n")
    return "".join(out)


def para_runs(p):
    return [r for r in p.iter(q("r")) if r.find(q("delText")) is None]


def build_parent_map(root):
    return {c: p for p in root.iter() for c in p}


def ancestors(el, pmap):
    while el in pmap:
        el = pmap[el]
        yield el


def location(p, pmap, tables_index):
    tc = tr = tbl = None
    for a in ancestors(p, pmap):
        if a.tag == q("tc") and tc is None:
            tc = a
        elif a.tag == q("tr") and tr is None:
            tr = a
        elif a.tag == q("tbl") and tbl is None:
            tbl = a
    if tbl is None:
        return "body", "", None
    ti = tables_index.get(tbl, "?")
    rows = [r for r in tbl if r.tag == q("tr")]
    ri = rows.index(tr) if tr in rows else "?"
    cells = [c for c in tr if c.tag == q("tc")] if tr is not None else []
    ci = cells.index(tc) if tc in cells else "?"
    label = ""
    if cells:
        label = "".join(run_text(r) for r in cells[0].iter(q("r")))
    return f"table{ti}/row{ri}/cell{ci}", label, tc


def build_registry(path, role):
    names, parts = load_parts(path)
    doc_parts = [k for k in parts if re.search(r"word/(document|header\d*|footer\d*)\.xml$", k)]
    rows = []
    name_counts = Counter()

    # first pass across all parts: collect form field names to find duplicates/empties
    for part in doc_parts:
        root = ET.fromstring(parts[part])
        for p in root.iter(q("p")):
            for r in p.iter(q("r")):
                fc = r.find(q("fldChar"))
                if fc is not None and fc.get(q("fldCharType")) == "begin":
                    ff = fc.find(q("ffData"))
                    if ff is not None:
                        nm = ff.find(q("name"))
                        name_counts[nm.get(q("val")) if nm is not None else ""] += 1

    for part in sorted(doc_parts):
        part_short = part.split("/")[-1]
        root = ET.fromstring(parts[part])
        pmap = build_parent_map(root)
        tables = list(root.iter(q("tbl")))
        tables_index = {t: i for i, t in enumerate(tables)}

        ff_field_index = 0
        # ---- legacy form fields
        for p in root.iter(q("p")):
            runs = list(p.iter(q("r")))
            state, cur = None, None
            for r in runs:
                fc = r.find(q("fldChar"))
                if fc is not None:
                    t = fc.get(q("fldCharType"))
                    if t == "begin":
                        ff = fc.find(q("ffData"))
                        cur = {"ffData": ff, "instr": "", "result": "", "p": p}
                        state = "instr"
                    elif t == "separate" and cur:
                        state = "result"
                    elif t == "end" and cur:
                        if cur["ffData"] is not None:
                            ff = cur["ffData"]
                            nm_el = ff.find(q("name"))
                            raw_name = nm_el.get(q("val")) if nm_el is not None else ""
                            dup_or_empty = (raw_name == "") or (name_counts[raw_name] > 1)
                            loc, label, _ = location(cur["p"], pmap, tables_index)
                            ti = ff.find(q("textInput"))
                            dflt = ti.find(q("default")) if ti is not None else None
                            kind = ("text" if ff.find(q("textInput")) is not None else
                                    "checkbox" if ff.find(q("checkBox")) is not None else
                                    "dropdown" if ff.find(q("ddList")) is not None else "other")
                            ff_field_index += 1
                            auto_id = f"{role}_{part_short}_{loc.replace('/', '_')}_ff{ff_field_index}"
                            field_id = raw_name if (raw_name and not dup_or_empty) else auto_id
                            rows.append({
                                "field_id": field_id,
                                "auto_id": auto_id,
                                "source": "legacy_form_field",
                                "subtype": kind,
                                "form_field_name": raw_name,
                                "name_is_empty_or_duplicate": dup_or_empty,
                                "part": part_short,
                                "location": loc,
                                "row_label": m(label, 80),
                                "default_value": m(dflt.get(q("val")) if dflt is not None else "", 40),
                                "current_result": m(cur["result"], 40),
                                "choices": "",
                            })
                        cur, state = None, None
                        continue
                if cur is None:
                    continue
                it = r.find(q("instrText"))
                if state == "instr" and it is not None:
                    cur["instr"] += it.text or ""
                elif state == "result":
                    cur["result"] += run_text(r)

        # spans covered by form fields, so literal placeholders inside them aren't double-counted
        ff_paragraphs = set()
        for p in root.iter(q("p")):
            if any(r.find(q("fldChar")) is not None for r in p.iter(q("r"))):
                ff_paragraphs.add(p)

        # ---- literal placeholders (outside form-field paragraphs)
        lit_index = 0
        for p in root.iter(q("p")):
            if p in ff_paragraphs:
                continue
            runs = para_runs(p)
            texts = [run_text(r) for r in runs]
            full = "".join(texts)
            for mt in PLACEHOLDER_RE.finditer(full):
                loc, label, _ = location(p, pmap, tables_index)
                lit_index += 1
                auto_id = f"{role}_{part_short}_{loc.replace('/', '_')}_lit{lit_index}"
                rows.append({
                    "field_id": auto_id,
                    "auto_id": auto_id,
                    "source": "literal_placeholder",
                    "subtype": "text",
                    "form_field_name": "",
                    "name_is_empty_or_duplicate": "",
                    "part": part_short,
                    "location": loc,
                    "row_label": m(label, 80),
                    "default_value": "",
                    "current_result": m(mt.group(0), 20),
                    "choices": "",
                })

        # ---- choice-like cells with no form field ("AC / DC / other")
        choice_index = 0
        seen_cells = set()
        for tc in root.iter(q("tc")):
            txt = "".join(run_text(r) for r in tc.iter(q("r"))).strip()
            has_ff = any(r.find(q("fldChar")) is not None for r in tc.iter(q("r")))
            if has_ff or id(tc) in seen_cells:
                continue
            if " / " in txt and not PLACEHOLDER_RE.search(txt) and len(txt) < 80 \
                    and len(txt.split(" / ")) >= 2:
                seen_cells.add(id(tc))
                # locate via the cell's own first paragraph
                first_p = tc.find(q("p"))
                loc, label, _ = location(first_p, pmap, tables_index) if first_p is not None else ("?", "", None)
                choice_index += 1
                auto_id = f"{role}_{part_short}_{loc.replace('/', '_')}_choice{choice_index}"
                rows.append({
                    "field_id": auto_id,
                    "auto_id": auto_id,
                    "source": "choice_cell",
                    "subtype": "choice",
                    "form_field_name": "",
                    "name_is_empty_or_duplicate": "",
                    "part": part_short,
                    "location": loc,
                    "row_label": m(label, 80),
                    "default_value": "",
                    "current_result": "",
                    "choices": m(txt, 80),
                })
    return rows


def main():
    global MASK
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("--role", required=True, help="short tag, e.g. template | old")
    ap.add_argument("--out", default=None, help="CSV output path (default: stdout)")
    ap.add_argument("--no-mask", action="store_true")
    a = ap.parse_args()
    MASK = not a.no_mask

    rows = build_registry(a.docx, a.role)
    fieldnames = ["field_id", "auto_id", "source", "subtype", "form_field_name",
                  "name_is_empty_or_duplicate", "part", "location", "row_label",
                  "default_value", "current_result", "choices"]

    out = open(a.out, "w", newline="", encoding="utf-8") if a.out else sys.stdout
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    if a.out:
        out.close()

    by_source = Counter(r["source"] for r in rows)
    dup_empty = sum(1 for r in rows if r["name_is_empty_or_duplicate"] is True)
    print(f"-- {a.docx}: {len(rows)} fields total | {dict(by_source)} "
          f"| empty/duplicate form-field names: {dup_empty}", file=sys.stderr)


if __name__ == "__main__":
    main()
