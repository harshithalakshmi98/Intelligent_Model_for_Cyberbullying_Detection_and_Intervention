#!/usr/bin/env python3
"""Build F21MP dissertation PDF (vm81 MSc thesis format) from Dissertation.md."""

from __future__ import annotations

import base64
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "research" / "Dissertation.md"
HTML_PATH = ROOT / "outputs" / "reports" / "F21MP_Dissertation.html"
PDF_PATH = ROOT / "outputs" / "reports" / "F21MP_Dissertation.pdf"

SECTION_RE = re.compile(r"^<!--\s*(front:[\w-]+|chapter:\d+|appendix:[a-z]|bibliography)\s*-->\s*$")


@dataclass
class TocEntry:
    title: str
    level: int
    anchor: str
    page: str = ""


@dataclass
class ListEntry:
    label: str
    anchor: str
    path: str = ""
    page: str = ""
    list_label: str = ""


FIGURE_CAPTION_RE = re.compile(r"^Figure\s+([\dA-Z.]+):\s*(.+)$", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^Table\s+([\dA-Z.]+):\s*(.+)$", re.IGNORECASE)


def figure_list_label(caption: str) -> str:
    """kb2051 LoF format: '2.1 Short descriptive title' (no Figure prefix, no citations)."""
    m = FIGURE_CAPTION_RE.match(caption.strip())
    if not m:
        return caption
    num, title = m.group(1), m.group(2).strip()
    title = re.sub(r"\s+[A-Z][\w\s,&–-]+et al\.\s*\[[^\]]+\]\s*$", "", title)
    title = re.sub(r"\s*\[[\dA-Z–,\s]+\]\s*$", "", title)
    return f"{num} {title}"


def table_list_label(label: str) -> str:
    """kb2051 LoT format: '3.1 Short descriptive title' (no Table prefix)."""
    m = TABLE_CAPTION_RE.match(label.strip())
    if not m:
        return label
    return f"{m.group(1)} {m.group(2).strip()}"


def figure_anchor_from_caption(caption: str, fallback: str) -> str:
    m = FIGURE_CAPTION_RE.match(caption.strip())
    if m:
        return "figure-" + m.group(1).lower().replace(".", "-")
    return fallback


def table_anchor_from_label(label: str, fallback: str) -> str:
    m = TABLE_CAPTION_RE.match(label.strip())
    if m:
        return "table-" + m.group(1).lower().replace(".", "-")
    return fallback


def _numeric_sort_key(prefix: str, label: str) -> tuple:
    """Sort '2.10' after '2.9' and appendix figures (A.1) after chapter figures."""
    m = re.match(rf"^{re.escape(prefix)}\s+([\dA-Z.]+)", label, re.IGNORECASE)
    if not m:
        return (999, 999, label)
    parts: list[tuple[int, int | str]] = []
    for part in m.group(1).split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part))
    return tuple(parts)


def sort_figures(figures: list[ListEntry]) -> list[ListEntry]:
    return sorted(
        figures,
        key=lambda f: _numeric_sort_key("Figure", f.label),
    )


def sort_tables(tables: list[ListEntry]) -> list[ListEntry]:
    return sorted(
        tables,
        key=lambda t: _numeric_sort_key("Table", t.label),
    )


@dataclass
class Document:
    sections: dict[str, str] = field(default_factory=dict)
    toc_entries: list[TocEntry] = field(default_factory=list)
    figures: list[ListEntry] = field(default_factory=list)
    tables: list[ListEntry] = field(default_factory=list)
    front_pages: dict[str, str] = field(default_factory=dict)


def parse_markdown(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    doc = Document()
    current_key = "preamble"
    buf: list[str] = []

    def flush():
        nonlocal buf
        if buf:
            doc.sections[current_key] = "\n".join(buf).strip()
            buf = []

    for line in text.splitlines():
        m = SECTION_RE.match(line.strip())
        if m:
            flush()
            current_key = m.group(1)
            continue
        buf.append(line)
    flush()
    return doc


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "section"


def page_marker(anchor: str) -> str:
    """Invisible anchor for PDF page extraction; marker duplicated in heading text."""
    token = f"@@PM_{anchor}@@"
    return (
        f'<p class="page-marker" style="font-size:1pt;line-height:1pt;margin:0;color:#fff">{token}</p>'
        f'<span class="page-marker-inline" style="font-size:1pt;color:#fefefe">{token}</span>'
    )


def embed_images(markdown: str, doc: Document) -> str:
    fig_counter = 0

    def repl(match: re.Match) -> str:
        nonlocal fig_counter
        alt, path = match.group(1), match.group(2)
        fig_counter += 1
        if not alt.lower().startswith("figure"):
            caption = f"Figure {fig_counter}: {alt}"
        else:
            caption = alt
        anchor = figure_anchor_from_caption(caption, f"fig-{fig_counter}")
        doc.figures.append(ListEntry(
            label=caption,
            list_label=figure_list_label(caption),
            anchor=anchor,
            path=path,
        ))

        img_path = (MD_PATH.parent / path).resolve()
        if not img_path.exists():
            img_path = (ROOT / path.lstrip("/")).resolve()
        if not img_path.exists():
            return f'<figure class="fig" id="{anchor}"><p><em>[Missing figure: {path}]</em></p></figure>'

        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return (
            f'<figure class="fig" id="{anchor}">'
            f'<img src="data:{mime};base64,{data}" alt="{caption}"/>'
            f'<figcaption>{caption}</figcaption>'
            f"</figure>"
        )

    return re.sub(r"!\[(.+?)\]\(([^)]+)\)", repl, markdown)


def inline_format(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def md_block_to_html(md: str, doc: Document, register_toc: bool = True) -> str:
    lines = md.splitlines()
    html: list[str] = []
    in_table = False
    in_code = False
    in_quote = False
    code_buf: list[str] = []
    quote_buf: list[str] = []
    table_anchor: str | None = None

    def close_table():
        nonlocal in_table, table_anchor
        if in_table:
            if table_anchor:
                html.append(f"</tbody></table>")
            else:
                html.append("</tbody></table>")
            in_table = False
            table_anchor = None

    def close_quote():
        nonlocal in_quote, quote_buf
        if in_quote:
            html.append(f"<blockquote>{' '.join(quote_buf)}</blockquote>")
            quote_buf = []
            in_quote = False

    for raw in lines:
        line = raw.rstrip()

        if line.strip().startswith("```"):
            close_quote()
            if in_code:
                html.append("<pre><code>" + "\n".join(code_buf) + "</code></pre>")
                code_buf = []
                in_code = False
            else:
                close_table()
                in_code = True
            continue
        if in_code:
            code_buf.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        if line.startswith(">"):
            close_table()
            in_quote = True
            quote_buf.append(line.lstrip("> ").strip())
            continue
        elif in_quote and line.strip() == "":
            close_quote()
            continue
        elif in_quote:
            close_quote()

        clean = line.strip("* ").strip()
        if clean.startswith("Table ") and TABLE_CAPTION_RE.match(clean):
            close_table()
            label = clean
            anchor = table_anchor_from_label(label, slugify(label))
            doc.tables.append(ListEntry(
                label=label,
                list_label=table_list_label(label),
                anchor=anchor,
            ))
            html.append(
                f'{page_marker(anchor)}<p class="table-caption" id="{anchor}">'
                f"<strong>{inline_format(label)}</strong></p>"
            )
            continue

        if line.startswith("|") and "|" in line[1:]:
            if re.match(r"^\|[-: |]+\|$", line.replace(" ", "")):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not in_table:
                close_table()
                html.append('<table class="data">')
                html.append("<thead><tr>" + "".join(f"<th>{inline_format(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                html.append("<tr>" + "".join(f"<td>{inline_format(c)}</td>" for c in cells) + "</tr>")
            continue
        else:
            close_table()

        if line.startswith("#### "):
            title = line[5:]
            anchor = slugify(title)
            if register_toc:
                doc.toc_entries.append(TocEntry(title=title, level=4, anchor=anchor))
            html.append(
                f'{page_marker(anchor)}<h4 id="{anchor}">{inline_format(title)}</h4>'
            )
        elif line.startswith("### "):
            title = line[4:]
            anchor = slugify(title)
            if register_toc:
                doc.toc_entries.append(TocEntry(title=title, level=3, anchor=anchor))
            html.append(
                f'{page_marker(anchor)}<h3 id="{anchor}">{inline_format(title)}</h3>'
            )
        elif line.startswith("## "):
            title = line[3:]
            anchor = slugify(title)
            if register_toc:
                doc.toc_entries.append(TocEntry(title=title, level=2, anchor=anchor))
            html.append(
                f'{page_marker(anchor)}<h2 class="chapter-heading" id="{anchor}">{inline_format(title)}</h2>'
            )
        elif line.startswith("# "):
            title = line[2:]
            html.append(f"<h1>{inline_format(title)}</h1>")
        elif line.strip() == "":
            html.append("")
        elif line.startswith("- "):
            html.append(f"<ul><li>{inline_format(line[2:])}</li></ul>")
        elif re.match(r"^\d+\.\s", line):
            num = line.split(".")[0]
            content = re.sub(r"^\d+\.\s*", "", line)
            html.append(f"<ol start='{num}'><li>{inline_format(content)}</li></ol>")
        else:
            html.append(f"<p>{inline_format(line)}</p>")

    close_table()
    close_quote()
    return "\n".join(html)


def render_title_page(md: str) -> str:
    lines = [ln[2:] if ln.startswith("# ") else ln for ln in md.splitlines() if ln.strip()]
    parts = [ln for ln in lines if ln.strip()]
    if not parts:
        return ""
    uni, doc_type, title = parts[0], parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else ""
    rest = parts[3:]
    body = "\n".join(f"<p>{inline_format(l)}</p>" for l in rest)
    return (
        f'<div class="title-page front-matter">'
        f"<p class='uni'>{inline_format(uni)}</p>"
        f"<p class='doctype'>{inline_format(doc_type)}</p>"
        f"<h1 class='thesis-title'>{inline_format(title)}</h1>"
        f"{body}"
        f"</div>"
    )


def render_quote(md: str) -> str:
    """vm81-style epigraph: italic centred quote; attribution right-aligned below."""
    lines = [ln.strip() for ln in md.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return ""
    parts = [f'<p class="quote-text">{inline_format(lines[0])}</p>']
    if len(lines) > 1:
        parts.append('<div class="quote-attribution-block">')
        parts.append(f'<p class="quote-attribution">{inline_format(lines[1])}</p>')
        if len(lines) > 2:
            parts.append(f'<p class="quote-source">{inline_format(lines[2])}</p>')
        parts.append("</div>")
    return "\n".join(parts)


def render_abbreviations(md: str) -> str:
    """kb2051-style abbreviations: 'AI Artificial Intelligence' per line."""
    lines = md.splitlines()
    body: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z][A-Za-z0-9-]*)\s+(.+)$", line)
        if m:
            abbr, meaning = m.group(1), m.group(2)
            body.append(
                f'<p class="abbrev-line"><span class="abbrev-acronym">{abbr}</span>'
                f'<span class="abbrev-meaning">{meaning}</span></p>'
            )
        else:
            body.append(f"<p>{inline_format(line)}</p>")
    return "\n".join(body)


def render_front_section(key: str, md: str, doc: Document) -> str:
    cls = "front-matter"
    short = key.split(":", 1)[-1] if ":" in key else key
    if short == "declaration-form":
        return (
            f'<div class="{cls} page-break">'
            f'{page_marker("student-declaration-of-authorship")}'
            f"{md_block_to_html(md, doc, register_toc=False)}</div>"
        )
    if short == "declaration-narrative":
        return (
            f'<div class="{cls} page-break" id="declaration-narrative">'
            f'{page_marker("declaration-narrative")}'
            f"{md_block_to_html(md, doc, register_toc=False)}</div>"
        )
    if short == "quote":
        return (
            f'<div class="{cls} quote-page page-break">'
            f"{render_quote(md)}</div>"
        )
    if short == "abstract":
        return (
            f'<div class="{cls} page-break" id="abstract">'
            f'{page_marker("abstract")}'
            f"{md_block_to_html(md, doc, register_toc=False)}</div>"
        )
    if short == "acknowledgements":
        return (
            f'<div class="{cls} page-break" id="acknowledgements">'
            f'{page_marker("acknowledgements")}'
            f"{md_block_to_html(md, doc, register_toc=False)}</div>"
        )
    if short == "abbreviations":
        return (
            f'<div class="{cls} abbreviations-page page-break" id="abbreviations">'
            f'{page_marker("abbreviations")}'
            f"<h2>Abbreviations</h2>"
            f"{render_abbreviations(md)}</div>"
        )
    return f'<div class="{cls}">{md_block_to_html(md, doc, register_toc=False)}</div>'


def render_toc(doc: Document) -> str:
    lines = [
        "<div class='front-matter page-break' id='contents'>",
        page_marker("contents"),
        "<h2>Contents</h2>",
    ]
    front_items = [
        ("Student Declaration of Authorship", "student-declaration-of-authorship"),
        ("Declaration of Authorship", "declaration-narrative"),
        ("Abstract", "abstract"),
        ("Acknowledgements", "acknowledgements"),
        ("Contents", "contents"),
        ("List of Figures", "list-of-figures"),
        ("List of Tables", "list-of-tables"),
        ("Abbreviations", "abbreviations"),
    ]
    for title, anchor in front_items:
        page = doc.front_pages.get(anchor, "")
        dots = "." * max(2, 60 - len(title))
        lines.append(f"<p class='toc-line toc-front'><span>{title}</span><span class='leaders'>{dots}</span><span class='pagenum'>{page}</span></p>")

    for entry in doc.toc_entries:
        if entry.anchor in {a for _, a in front_items}:
            continue
        title = entry.title
        if entry.level == 4:
            indent_cls = "toc-sub2"
            prefix = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            dots_len = 44
        elif entry.level == 3:
            indent_cls = "toc-sub"
            prefix = "&nbsp;&nbsp;&nbsp;&nbsp;"
            dots_len = 48
        else:
            indent_cls = "toc-chapter"
            prefix = ""
            dots_len = 52
        dots = "." * max(2, dots_len - len(title))
        lines.append(
            f"<p class='toc-line {indent_cls}'><span>{prefix}{title}</span>"
            f"<span class='leaders'>{dots}</span><span class='pagenum'>{entry.page}</span></p>"
        )
    lines.append("</div>")
    return "\n".join(lines)


def render_list_of_figures(doc: Document) -> str:
    lines = [
        "<div class='front-matter page-break' id='list-of-figures'>",
        page_marker("list-of-figures"),
        "<h2>List of Figures</h2>",
    ]
    for fig in sort_figures(doc.figures):
        entry = fig.list_label or figure_list_label(fig.label)
        dots = "." * max(2, 55 - min(len(entry), 55))
        lines.append(
            f"<p class='toc-line'><span>{entry}</span>"
            f"<span class='leaders'>{dots}</span><span class='pagenum'>{fig.page}</span></p>"
        )
    lines.append("</div>")
    return "\n".join(lines)


def render_list_of_tables(doc: Document) -> str:
    lines = [
        "<div class='front-matter page-break' id='list-of-tables'>",
        page_marker("list-of-tables"),
        "<h2>List of Tables</h2>",
    ]
    for tbl in sort_tables(doc.tables):
        entry = tbl.list_label or table_list_label(tbl.label)
        dots = "." * max(2, 55 - min(len(entry), 55))
        lines.append(
            f"<p class='toc-line'><span>{entry}</span>"
            f"<span class='leaders'>{dots}</span><span class='pagenum'>{tbl.page}</span></p>"
        )
    lines.append("</div>")
    return "\n".join(lines)


def collect_metadata(doc: Document) -> None:
    doc.toc_entries = []
    doc.figures = []
    doc.tables = []
    chapter_keys = sorted(
        (k for k in doc.sections if k.startswith("chapter:")),
        key=lambda x: int(x.split(":")[1]),
    )
    appendix_keys = sorted(k for k in doc.sections if k.startswith("appendix:"))
    for key in chapter_keys + appendix_keys:
        md = doc.sections[key]
        if "!" in md:
            md = embed_images(md, doc)
        md_block_to_html(md, doc)
    if "bibliography" in doc.sections:
        md_block_to_html(doc.sections["bibliography"], doc)


def embed_images_render(md: str, figures: list[ListEntry]) -> str:
    by_path = {f.path: f for f in figures}

    def repl(match: re.Match) -> str:
        alt, path = match.group(1), match.group(2)
        fig = by_path.get(path)
        anchor = fig.anchor if fig else slugify(alt)
        caption = fig.label if fig else alt

        img_path = (MD_PATH.parent / path).resolve()
        if not img_path.exists():
            img_path = (ROOT / path.lstrip("/")).resolve()
        if not img_path.exists():
            return f'<figure class="fig" id="{anchor}"><p><em>[Missing figure: {path}]</em></p></figure>'

        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return (
            f'<figure class="fig" id="{anchor}">'
            f"{page_marker(anchor)}"
            f'<img src="data:{mime};base64,{data}" alt="{caption}"/>'
            f"<figcaption>{caption}</figcaption>"
            f"</figure>"
        )

    return re.sub(r"!\[(.+?)\]\(([^)]+)\)", repl, md)


def render_section_md(md: str, doc: Document) -> str:
    if "!" in md:
        md = embed_images_render(md, doc.figures)
    return md_block_to_html(md, Document(), register_toc=False)


def section_marker(md: str) -> str:
    for line in md.splitlines():
        if line.startswith("## "):
            return page_marker(slugify(line[3:]))
    return ""


def build_body_html(doc: Document, include_lists: bool = True) -> str:
    parts: list[str] = []

    if "front:title-page" in doc.sections:
        parts.append(render_title_page(doc.sections["front:title-page"]))
    for key in ("front:declaration-form", "front:declaration-narrative", "front:quote",
                "front:abstract", "front:acknowledgements"):
        if key in doc.sections:
            parts.append(render_front_section(key, doc.sections[key], doc))

    if include_lists:
        parts.append(render_toc(doc))
        parts.append(render_list_of_figures(doc))
        parts.append(render_list_of_tables(doc))

    if "front:abbreviations" in doc.sections:
        parts.append(render_front_section("front:abbreviations", doc.sections["front:abbreviations"], doc))

    chapter_keys = sorted(
        (k for k in doc.sections if k.startswith("chapter:")),
        key=lambda x: int(x.split(":")[1]),
    )
    for key in chapter_keys:
        md = doc.sections[key]
        html = render_section_md(md, doc)
        parts.append(f'<div class="chapter page-break">{section_marker(md)}{html}</div>')

    for key in sorted(k for k in doc.sections if k.startswith("appendix:")):
        md = doc.sections[key]
        html = render_section_md(md, doc)
        parts.append(f'<div class="appendix page-break">{section_marker(md)}{html}</div>')

    if "bibliography" in doc.sections:
        md = doc.sections["bibliography"]
        html = md_block_to_html(md, Document(), register_toc=False)
        parts.append(
            f'<div class="bibliography page-break" id="bibliography">'
            f'{section_marker(md)}{html}</div>'
        )

    return "\n".join(parts)


def wrap_html(body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>F21MP Dissertation — Cyberbullying Detection</title>
<style>
  @page {{ size: A4; margin: 2.4cm 2.2cm 2.4cm 2.4cm; }}
  body {{
    font-family: "Times New Roman", Times, serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #111;
    margin: 0;
    padding: 0;
  }}
  .title-page {{
    text-align: center;
    padding-top: 2.5cm;
    min-height: 24cm;
  }}
  .title-page .uni {{ font-size: 14pt; margin: 0.4cm 0; }}
  .title-page .doctype {{ font-size: 13pt; margin: 0.6cm 0; }}
  .title-page .thesis-title {{
    font-size: 16pt;
    font-weight: bold;
    margin: 1.2cm 1cm;
    line-height: 1.35;
    page-break-before: avoid;
  }}
  .title-page p {{ text-align: center; margin: 0.35cm 0; }}
  .front-matter h2 {{
    font-size: 14pt;
    text-align: center;
    margin-bottom: 0.8cm;
    border: none;
  }}
  .quote-page {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 22cm;
    text-align: center;
    font-size: 11pt;
    line-height: 1.5;
  }}
  .quote-text {{
    max-width: 11cm;
    margin: 0 0 1.2cm 0;
    font-style: italic;
    text-align: center;
  }}
  .quote-attribution-block {{
    width: 11cm;
    text-align: right;
  }}
  .quote-attribution {{
    margin: 0;
    font-style: normal;
    font-weight: normal;
  }}
  .quote-source {{
    margin: 0.1cm 0 0 0;
    font-size: 10.5pt;
    font-style: normal;
  }}
  .abbreviations-page {{
    font-size: 10pt;
    line-height: 1.45;
  }}
  .abbrev-line {{
    margin: 0.12cm 0;
    text-align: left;
    text-indent: 0;
  }}
  .abbrev-acronym {{
    display: inline-block;
    min-width: 2.8cm;
    vertical-align: top;
  }}
  .abbrev-meaning {{
    display: inline;
  }}
  .page-break {{ page-break-before: always; }}
  .chapter h2.chapter-heading, .appendix h2.chapter-heading {{
    font-size: 14pt;
    margin-top: 0;
    padding-top: 0.2cm;
    border-bottom: none;
    page-break-before: avoid;
  }}
  h3 {{ font-size: 11.5pt; margin-top: 0.5cm; }}
  p {{ text-align: justify; margin: 0.3cm 0; }}
  table.data {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.35cm 0 0.5cm 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }}
  table.data th, table.data td {{
    border: 1px solid #333;
    padding: 4px 6px;
    vertical-align: top;
  }}
  table.data th {{ background: #f2f2f2; }}
  .table-caption {{ font-size: 10pt; margin-top: 0.5cm; margin-bottom: 0.1cm; }}
  pre {{
    background: #f7f7f7;
    border: 1px solid #ddd;
    padding: 8px;
    font-size: 8.5pt;
    overflow-x: auto;
    page-break-inside: avoid;
  }}
  code {{ font-family: Consolas, "Courier New", monospace; font-size: 9pt; }}
  figure.fig {{
    text-align: center;
    margin: 0.5cm 0;
    page-break-inside: avoid;
  }}
  figure.fig img {{
    max-width: 94%;
    max-height: 11cm;
    border: 1px solid #ccc;
  }}
  figcaption {{ font-size: 9.5pt; margin-top: 4px; text-align: center; }}
  .toc-line {{
    display: flex;
    font-size: 10.5pt;
    margin: 0.12cm 0;
    line-height: 1.35;
  }}
  .toc-line span:first-child {{ flex-shrink: 0; }}
  .toc-line .leaders {{
    flex-grow: 1;
    overflow: hidden;
    white-space: nowrap;
    letter-spacing: 1px;
    color: #444;
    margin: 0 4px;
  }}
  .toc-line .pagenum {{ flex-shrink: 0; min-width: 1.2cm; text-align: right; }}
  .toc-sub {{ padding-left: 0.8cm; font-size: 10pt; }}
  .toc-sub2 {{ padding-left: 1.4cm; font-size: 9.5pt; }}
  h4 {{ font-size: 10.5pt; margin-top: 0.4cm; font-weight: bold; }}
  .page-marker {{
    position: absolute;
    left: -9999px;
    top: 0;
    width: 0;
    height: 0;
    overflow: hidden;
    opacity: 0;
    font-size: 0;
    line-height: 0;
    margin: 0;
    padding: 0;
    color: transparent;
  }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def strip_page_markers(html: str) -> str:
    """Remove invisible page-marker tokens from final PDF HTML."""
    html = re.sub(
        r'<p class="page-marker"[^>]*>@@PM_[^@]+@@</p>\s*',
        "",
        html,
    )
    html = re.sub(
        r'<span class="page-marker-inline"[^>]*>@@PM_[^@]+@@</span>\s*',
        "",
        html,
    )
    return re.sub(r"@@PM_[^@]+@@", "", html)


def chrome_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome.exists():
        raise FileNotFoundError("Google Chrome not found for PDF export")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        f"file://{html_path.resolve()}",
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def find_anchor_pages(
    pdf_path: Path,
    markers: dict[str, str],
    marker_only_anchors: set[str] | None = None,
) -> dict[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    found: dict[str, int] = {}
    marker_only = marker_only_anchors or set()

    search_labels = dict(markers)
    search_labels["declaration-narrative"] = "I, Pedakamsetti Harshitha Lakshmi, declare that this thesis"
    search_labels["student-declaration-of-authorship"] = "F21MP – Masters Project and Dissertation"

    # Pass 1: marker tokens (works even on TOC / list pages that label matching skips).
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for anchor in markers:
            if anchor in found:
                continue
            if f"@@PM_{anchor}@@" in text:
                found[anchor] = i + 1

    def is_skipped_page(text: str) -> bool:
        if "Contents" in text and "List of Figures" in text and "1 Introduction" in text:
            return True
        if text.strip().startswith("List of Figures") or (
            "List of Figures" in text[:40] and text.count("....") >= 3
        ):
            return True
        if text.strip().startswith("List of Tables") or (
            "List of Tables" in text[:40] and text.count("....") >= 3
        ):
            return True
        if text.count("....") >= 4 and "Introduction" not in text[:80]:
            return True
        return False

    for i, page in enumerate(reader.pages):
        text = re.sub(r"\s+", " ", page.extract_text() or "")
        if is_skipped_page(text):
            continue
        for anchor, label in search_labels.items():
            if anchor in found:
                continue
            if anchor in marker_only:
                continue
            needle = re.sub(r"\s+", " ", label).strip()
            if needle and len(needle) > 12 and needle in text:
                found[anchor] = i + 1
    return found


def fill_page_numbers(doc: Document, pdf_path: Path) -> None:
    markers: dict[str, str] = {
        "student-declaration-of-authorship": "Student Declaration of Authorship",
        "declaration-narrative": "Declaration of Authorship",
        "abstract": "Abstract",
        "acknowledgements": "Acknowledgements",
        "contents": "Contents",
        "list-of-figures": "List of Figures",
        "list-of-tables": "List of Tables",
        "abbreviations": "Abbreviations",
    }
    for entry in doc.toc_entries:
        if entry.level >= 2:
            markers[entry.anchor] = entry.title
    for fig in doc.figures:
        markers[fig.anchor] = fig.label
    for tbl in doc.tables:
        markers[tbl.anchor] = tbl.label

    section_anchors = {e.anchor for e in doc.toc_entries}
    section_anchors |= {f.anchor for f in doc.figures}
    section_anchors |= {t.anchor for t in doc.tables}

    pages = find_anchor_pages(pdf_path, markers, marker_only_anchors=section_anchors)

    for entry in doc.toc_entries:
        if entry.anchor in pages:
            entry.page = str(pages[entry.anchor])

    for fig in doc.figures:
        if fig.anchor in pages:
            fig.page = str(pages[fig.anchor])

    for tbl in doc.tables:
        if tbl.anchor in pages:
            tbl.page = str(pages[tbl.anchor])

    for anchor in ("student-declaration-of-authorship", "declaration-narrative", "abstract",
                   "acknowledgements", "contents", "list-of-figures", "list-of-tables", "abbreviations"):
        if anchor in pages:
            doc.front_pages[anchor] = str(pages[anchor])


def main() -> int:
    if not MD_PATH.exists():
        print(f"Missing {MD_PATH}", file=sys.stderr)
        return 1

    doc = parse_markdown(MD_PATH)
    collect_metadata(doc)

    # Pass 1: build without page numbers in lists
    body1 = build_body_html(doc, include_lists=True)
    html1 = wrap_html(body1)
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(html1, encoding="utf-8")
    chrome_pdf(HTML_PATH, PDF_PATH)

    # Extract page numbers and rebuild
    fill_page_numbers(doc, PDF_PATH)

    body2 = build_body_html(doc, include_lists=True)
    html2 = strip_page_markers(wrap_html(body2))
    HTML_PATH.write_text(html2, encoding="utf-8")
    chrome_pdf(HTML_PATH, PDF_PATH)

    print(f"Wrote {HTML_PATH}")
    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
