#!/usr/bin/env python3
"""
inspect_inputs.py - structural diagnostics for the report-autofill project.

Runs fully locally. Prints STRUCTURE (field types, placeholder patterns,
run-splitting, test-section headers), not document content. Digits in any
printed context are masked as '#' by default; use --no-mask to disable.

Usage:
    pip install pymupdf            # only needed for the PDF check
    python inspect_inputs.py --template template.docx \
                             --old old_report.docx \
                             --new new_report.pdf  > diag.txt
Any argument can be omitted.
"""
import argparse
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


def m(s, n=90):
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
    """Visible runs of a paragraph, including runs inside hyperlinks/sdt/ins."""
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
        return "body", ""
    ti = tables_index.get(tbl, "?")
    rows = [r for r in tbl if r.tag == q("tr")]
    ri = rows.index(tr) if tr in rows else "?"
    cells = [c for c in tr if c.tag == q("tc")] if tr is not None else []
    ci = cells.index(tc) if tc in cells else "?"
    label = ""
    if cells:
        label = "".join(run_text(r) for r in cells[0].iter(q("r")))
    return f"table{ti}/row{ri}/cell{ci}", label


# ---------------------------------------------------------------- DOCX
def inspect_docx(path, role):
    print(f"\n{'=' * 70}\n{role.upper()}: {path}\n{'=' * 70}")
    names, parts = load_parts(path)
    print(f"ZIP parts: {len(names)} | customXml parts: {sum(n.startswith('customXml/') for n in names)}")
    print("word/ xml parts:", ", ".join(sorted(k.split('/')[-1] for k in parts)))

    if "word/settings.xml" in parts:
        st = ET.fromstring(parts["word/settings.xml"])
        prot = st.find(".//w:documentProtection", NS)
        if prot is not None:
            print("Document protection:", {k.split('}')[-1]: v for k, v in prot.attrib.items()
                                           if k.endswith(('edit', 'enforcement', 'formatting'))})
        else:
            print("Document protection: none")

    doc_parts = [k for k in parts if re.search(r"word/(document|header\d*|footer\d*)\.xml$", k)]
    grand = Counter()
    for part in sorted(doc_parts):
        root = ET.fromstring(parts[part])
        pmap = build_parent_map(root)
        tables = list(root.iter(q("tbl")))
        tables_index = {t: i for i, t in enumerate(tables)}
        c = Counter()
        c["tables"] = len(tables)
        c["rows"] = sum(1 for _ in root.iter(q("tr")))
        c["gridSpan"] = sum(1 for _ in root.iter(q("gridSpan")))
        c["vMerge"] = sum(1 for _ in root.iter(q("vMerge")))
        c["tracked_ins"] = sum(1 for _ in root.iter(q("ins")))
        c["tracked_del"] = sum(1 for _ in root.iter(q("del")))
        c["sectPr"] = sum(1 for _ in root.iter(q("sectPr")))
        c["rendered_page_breaks"] = sum(1 for _ in root.iter(q("lastRenderedPageBreak")))
        c["explicit_page_breaks"] = sum(1 for b in root.iter(q("br")) if b.get(q("type")) == "page")
        c["textboxes"] = sum(1 for _ in root.iter(q("txbxContent")))

        # ---- legacy form fields (FORMTEXT / FORMCHECKBOX / FORMDROPDOWN)
        ff_rows = []
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
                            ff_rows.append(cur)
                        cur, state = None, None
                    continue
                if cur is None:
                    continue
                it = r.find(q("instrText"))
                if state == "instr" and it is not None:
                    cur["instr"] += it.text or ""
                elif state == "result":
                    cur["result"] += run_text(r)
        c["legacy_form_fields"] = len(ff_rows)
        types = Counter()
        for f in ff_rows:
            ff = f["ffData"]
            kind = ("text" if ff.find(q("textInput")) is not None else
                    "checkbox" if ff.find(q("checkBox")) is not None else
                    "dropdown" if ff.find(q("ddList")) is not None else "other")
            types[kind] += 1
        # ---- content controls
        sdts = list(root.iter(q("sdt")))
        c["content_controls"] = len(sdts)
        sdt_types, sdt_tags, bound = Counter(), [], 0
        for s in sdts:
            pr = s.find(q("sdtPr"))
            if pr is None:
                continue
            kinds = [el.tag.split('}')[-1] for el in pr]
            k = next((x for x in ("text", "richText", "dropDownList", "comboBox", "date",
                                  "checkbox", "picture", "docPartObj") if x in kinds), "richText/other")
            if any(el.tag.endswith("checkbox") for el in pr.iter()):
                k = "checkbox"
            sdt_types[k] += 1
            tag = pr.find(q("tag"))
            sdt_tags.append(tag.get(q("val")) if tag is not None else None)
            if pr.find(q("dataBinding")) is not None:
                bound += 1

        # ---- literal placeholders + run splitting
        ph_rows = []
        for p in root.iter(q("p")):
            runs = para_runs(p)
            texts = [run_text(r) for r in runs]
            full = "".join(texts)
            for mt in PLACEHOLDER_RE.finditer(full):
                # which runs does the match touch?
                pos, touched = 0, []
                for i, t in enumerate(texts):
                    a, b = pos, pos + len(t)
                    if a < mt.end() and b > mt.start():
                        touched.append(i)
                    pos = b
                rpr = runs[touched[0]].find(q("rPr")) if touched else None
                fmt = []
                if rpr is not None:
                    for tagname in ("shd", "highlight", "b", "i", "strike", "vertAlign"):
                        if rpr.find(q(tagname)) is not None:
                            fmt.append(tagname)
                in_ff = any(r.find(q("fldChar")) is not None for r in runs)
                in_sdt = any(a.tag == q("sdt") for a in ancestors(runs[touched[0]], pmap)) if touched else False
                loc, label = location(p, pmap, tables_index)
                ph_rows.append({
                    "pattern": mt.group(0)[:8], "runs_spanned": len(touched), "fmt": fmt,
                    "para_has_fields": in_ff, "in_sdt": in_sdt, "loc": loc,
                    "label": label, "context": full,
                })
        c["literal_placeholders"] = len(ph_rows)

        # ---- choice cells like "AC / DC / other"
        choice_rows = []
        for tc in root.iter(q("tc")):
            txt = "".join(run_text(r) for r in tc.iter(q("r"))).strip()
            if " / " in txt and not PLACEHOLDER_RE.search(txt) and len(txt) < 80 \
                    and len(txt.split(" / ")) >= 2:
                runs = [r for r in tc.iter(q("r")) if run_text(r).strip()]
                choice_rows.append((txt, len(runs)))

        # ---- verdict column heuristic (old reports: P / N / F)
        verdicts = Counter()
        for tc in root.iter(q("tc")):
            txt = "".join(run_text(r) for r in tc.iter(q("r"))).strip()
            if txt in ("P", "N", "F", "N/A", "n.a.", "Pass", "Fail"):
                verdicts[txt] += 1

        grand.update(c)
        print(f"\n--- {part} ---")
        print("counts:", dict(c))
        if ff_rows:
            print(f"legacy form fields by type: {dict(types)}")
            for f in ff_rows[:60]:
                ff = f["ffData"]
                name = ff.find(q("name"))
                ti = ff.find(q("textInput"))
                dflt = ti.find(q("default")) if ti is not None else None
                mx = ti.find(q("maxLength")) if ti is not None else None
                tp = ti.find(q("type")) if ti is not None else None
                loc, label = location(f["p"], pmap, tables_index)
                print(f"  FF name={name.get(q('val')) if name is not None else None!r:14} "
                      f"instr={f['instr'].strip()[:12]!r:14} "
                      f"default={(dflt.get(q('val')) if dflt is not None else None)!r:8} "
                      f"maxLen={(mx.get(q('val')) if mx is not None else None)} "
                      f"type={(tp.get(q('val')) if tp is not None else 'regular')} "
                      f"result={m(f['result'], 12)!r:10} @ {loc} | label: {m(label, 50)}")
            if len(ff_rows) > 60:
                print(f"  … {len(ff_rows) - 60} more")
        if sdts:
            print(f"content controls by type: {dict(sdt_types)} | with dataBinding: {bound}")
            tag_counts = Counter(sdt_tags)
            print(f"  tags (first 30): {list(tag_counts.items())[:30]}")
        if ph_rows:
            split = sum(1 for r in ph_rows if r["runs_spanned"] > 1)
            print(f"literal placeholders: {len(ph_rows)} | split across >1 run: {split} | "
                  f"patterns: {dict(Counter(r['pattern'] for r in ph_rows))}")
            for r in ph_rows[:80]:
                print(f"  PH {r['pattern']!r:8} runs={r['runs_spanned']} fmt={','.join(r['fmt']) or '-':14} "
                      f"sdt={int(r['in_sdt'])} fieldsInPara={int(r['para_has_fields'])} @ {r['loc']:22} "
                      f"| label: {m(r['label'], 45)} | cell: {m(r['context'], 60)}")
            if len(ph_rows) > 80:
                print(f"  … {len(ph_rows) - 80} more")
        if choice_rows:
            print(f"choice-like cells ('a / b / c'): {len(choice_rows)}")
            for txt, nr in choice_rows[:25]:
                print(f"  CHOICE runs={nr:2} | {m(txt, 70)}")
        if verdicts:
            print(f"verdict-like cells: {dict(verdicts)}")
    print(f"\nTOTAL over document+headers+footers: {dict(grand)}")


# ---------------------------------------------------------------- PDF
def inspect_pdf(path):
    print(f"\n{'=' * 70}\nNEW REPORT PDF: {path}\n{'=' * 70}")
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            print("PyMuPDF not installed -> pip install pymupdf  (or run: pdfinfo / pdffonts)")
            return
    doc = fitz.open(path)
    print(f"pages: {doc.page_count} | encrypted: {doc.is_encrypted} | metadata producer/creator: "
          f"{m(doc.metadata.get('producer'), 40)!r} / {m(doc.metadata.get('creator'), 40)!r}")
    empty_pages, chars = [], []
    header_re = re.compile(r"^\s*(T\d{2,4})\s*[-–]\s*(.+)$")
    headers, sample_ct, path_vals, rep_ct, table_pages = [], 0, Counter(), 0, Counter()
    comma_dec = dot_dec = 0
    for i, page in enumerate(doc):
        t = page.get_text("text")
        chars.append(len(t.strip()))
        if len(t.strip()) < 20:
            empty_pages.append(i + 1)
        for line in t.splitlines():
            h = header_re.match(line)
            if h:
                headers.append((h.group(1), h.group(2).strip(), i + 1))
        sample_ct += len(re.findall(r"Sample\s*:", t))
        rep_ct += len(re.findall(r"Repetition\s*:", t))
        for pv in re.findall(r"Path\s*:\s*([A-Za-z0-9+\-/]{1,12})", t):
            path_vals[pv.strip()] += 1
        comma_dec += len(re.findall(r"\b\d+,\d+\b", t))
        dot_dec += len(re.findall(r"\b\d+\.\d+\b", t))
        if i < 40:
            try:
                table_pages[i + 1] = len(page.find_tables().tables)
            except Exception:
                pass
    print(f"text layer: {doc.page_count - len(empty_pages)}/{doc.page_count} pages have text "
          f"(median chars/page ~{sorted(chars)[len(chars) // 2] if chars else 0})")
    if empty_pages:
        print(f"  pages with (almost) no text -> need OCR: {empty_pages[:40]}{' …' if len(empty_pages) > 40 else ''}")
    print(f"decimal style counts: comma={comma_dec} dot={dot_dec}")
    print(f"'Sample:' blocks={sample_ct} | 'Repetition:'={rep_ct} | Path values: {dict(path_vals.most_common(12))}")
    if table_pages:
        print(f"PyMuPDF-detected tables on first 40 pages: {dict(table_pages)}")
    uniq = {}
    for tid, title, pg in headers:
        uniq.setdefault((tid, title), []).append(pg)
    print(f"test-section headers matching 'T### - title': {len(uniq)} unique")
    for (tid, title), pgs in list(uniq.items())[:120]:
        print(f"  {tid:6} p{pgs[0]:>4}-{pgs[-1]:<4} ({len(pgs)} hits) {title[:70]}")


def main():
    global MASK
    ap = argparse.ArgumentParser()
    ap.add_argument("--template")
    ap.add_argument("--old")
    ap.add_argument("--new")
    ap.add_argument("--no-mask", action="store_true", help="do not mask digits in printed context")
    a = ap.parse_args()
    MASK = not a.no_mask
    if not (a.template or a.old or a.new):
        ap.print_help()
        sys.exit(1)
    if a.template:
        inspect_docx(a.template, "template")
    if a.old:
        if a.old.lower().endswith(".docx"):
            inspect_docx(a.old, "old report")
        else:
            print("old report is not .docx; convert or pass as --new for PDF checks")
    if a.new:
        inspect_pdf(a.new)


if __name__ == "__main__":
    main()
