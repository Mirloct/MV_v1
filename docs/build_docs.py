"""Regenerate docs/documentation.html from the project's markdown docs.

Consolidates README.md, CONTEXT.md, and docs/*.md into a single, offline,
self-contained HTML file with a sidebar table of contents. No third-party
markdown library is used (none is a project dependency) — this module
implements a small hand-written converter for the subset of Markdown these
files actually use: ATX headings (#..######), paragraphs, bullet/numbered
lists (incl. "- [x] "/"- [ ] " checklists), fenced code blocks, inline code,
bold/italic, links, tables, blockquotes, horizontal rules, and "$$...$$"
LaTeX math blocks (rendered as preformatted text, matching the approach
already used by src/reporting/report.py for its offline HTML output).

Usage (from the project root):

    python docs/build_docs.py

Edit the *source* markdown files, then re-run this script to refresh
docs/documentation.html. Do not hand-edit the generated HTML.
"""

from __future__ import annotations

import html
import re
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "documentation.html"

# (source path relative to project root, anchor id, display label) — order
# here is the order the sections and the table of contents are rendered in.
DOCS: list[tuple[str, str, str]] = [
    ("README.md", "readme", "README.md"),
    ("CONTEXT.md", "context", "CONTEXT.md"),
    ("docs/leakage_free_pipeline.md", "leakage-free-pipeline",
     "docs/leakage_free_pipeline.md"),
    ("docs/escalamiento_lineal.md", "escalamiento-lineal",
     "docs/escalamiento_lineal.md"),
    ("docs/decisiones_de_modelado.md", "decisiones-de-modelado",
     "docs/decisiones_de_modelado.md"),
    ("docs/models_isolation_forest.md", "iforest", "docs/models_isolation_forest.md"),
    ("docs/models_vae.md", "vae", "docs/models_vae.md"),
    ("docs/evaluation.md", "evaluation", "docs/evaluation.md"),
    ("docs/interpretability_and_reporting.md", "interpretability-reporting",
     "docs/interpretability_and_reporting.md"),
    ("docs/geeksforgeeks_notes.md", "geeksforgeeks-references",
     "docs/geeksforgeeks_notes.md"),
]

# Markdown link targets (as they literally appear in the source files) that
# point at another consolidated document, mapped to that document's anchor.
KNOWN_FILE_LINKS = {path: anchor for path, anchor, _ in DOCS}
KNOWN_FILE_LINKS.update({f"./{path}": anchor for path, anchor, _ in DOCS})


# --------------------------------------------------------------------------- #
# Inline markdown -> HTML                                                     #
# --------------------------------------------------------------------------- #

def _strip_inline_markdown(text: str) -> str:
    """Plain-text version of a heading, for slug generation."""
    t = re.sub(r"`([^`]+)`", r"\1", text)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"__(.+?)__", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"_(.+?)_", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    return t


def _slugify(text: str) -> str:
    """GitHub-style heading slug: lowercase, drop non [a-z0-9 _-], spaces->hyphens.

    Matches the algorithm GitHub uses closely enough that the handful of
    same-document markdown links in these files (e.g. "#status--roadmap",
    "#1-concepts") resolve to the correct heading once prefixed with the
    document's anchor id.
    """
    s = text.lower()
    s = "".join(ch for ch in s if ch.isalnum() or ch in " -_")
    return s.replace(" ", "-")


def _rewrite_href(href: str, doc_id: str) -> str:
    href = href.strip()
    if href.startswith("#"):
        return f"#{doc_id}-{href[1:]}"
    if href in KNOWN_FILE_LINKS:
        return f"#{KNOWN_FILE_LINKS[href]}"
    return href  # external (http/https/mailto/...) or unknown — leave as-is


def inline_render(text: str, doc_id: str) -> str:
    """Render inline markdown (code/links/bold/italic/autolinks) to HTML."""
    placeholders: list[str] = []

    def store(snippet: str) -> str:
        placeholders.append(snippet)
        return f"\x00{len(placeholders) - 1}\x00"

    out = html.escape(text, quote=False)

    # Inline code spans first, so their contents are immune to ** / _ below.
    out = re.sub(r"`([^`]+)`", lambda m: store(f"<code>{m.group(1)}</code>"), out)

    # Markdown links [text](url) — rewrite internal targets to in-page anchors.
    def _link(m: re.Match) -> str:
        link_text, href = m.group(1), m.group(2)
        rewritten = _rewrite_href(href, doc_id)
        new_href = html.escape(rewritten, quote=True)
        target = '' if rewritten.startswith('#') else ' target="_blank" rel="noopener"'
        return store(f'<a href="{new_href}"{target}>{link_text}</a>')

    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link, out)

    # Bare http(s) URLs (used heavily in "Sources:" lists) -> clickable links.
    def _autolink(m: re.Match) -> str:
        url = m.group(0)
        trail = ""
        while url and url[-1] in ".,);:":
            trail = url[-1] + trail
            url = url[:-1]
        esc = html.escape(url, quote=True)
        return store(f'<a href="{esc}" target="_blank" rel="noopener">{url}</a>') + trail

    out = re.sub(r"https?://[^\s<>\x00]+", _autolink, out)

    # Bold, then italic (order matters: ** before single *).
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"__(.+?)__", r"<strong>\1</strong>", out)
    out = re.sub(r"\*(.+?)\*", r"<em>\1</em>", out)
    out = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", out)

    # Restore placeholders (may be nested one level, e.g. code span inside a
    # link's text) — loop until no more placeholder tokens remain.
    def _restore(m: re.Match) -> str:
        return placeholders[int(m.group(1))]

    while "\x00" in out:
        out = re.sub(r"\x00(\d+)\x00", _restore, out)
    return out


# --------------------------------------------------------------------------- #
# Block markdown -> HTML                                                      #
# --------------------------------------------------------------------------- #

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_UL_RE = re.compile(r"^([-*])\s+(.*)$")
_OL_RE = re.compile(r"^(\d+)\.\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_CHECKBOX_RE = re.compile(r"^\[( |x|X)\]\s+(.*)$")


def _is_block_start(line: str) -> bool:
    s = line.strip()
    if s == "":
        return True
    if s.startswith("```"):
        return True
    if s == "$$" or (s.startswith("$$") and s.endswith("$$") and len(s) > 4):
        return True
    if _HEADING_RE.match(s):
        return True
    if _HR_RE.match(s):
        return True
    if s.startswith(">"):
        return True
    if s.startswith("|"):
        return True
    if _UL_RE.match(s) or _OL_RE.match(s):
        return True
    return False


def _split_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def render_blocks(lines: list[str], doc_id: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Render a document's markdown lines to HTML.

    Returns (html_body, headings) where headings is a list of
    (level, anchor_id, plain_text) for the sidebar sub-navigation.
    """
    out: list[str] = []
    headings: list[tuple[int, str, str]] = []
    used_ids: set[str] = set()
    i, n = 0, len(lines)

    def unique_id(base: str) -> str:
        if base not in used_ids:
            used_ids.add(base)
            return base
        k = 2
        while f"{base}-{k}" in used_ids:
            k += 1
        used_ids.add(f"{base}-{k}")
        return f"{base}-{k}"

    while i < n:
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        stripped = line.strip()

        # Fenced code block.
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            i += 1
            code_lines = []
            while i < n and lines[i].strip() != "```":
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code_html = html.escape("\n".join(code_lines))
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{code_html}</code></pre>")
            continue

        # Multi-line "$$ ... $$" math block.
        if stripped == "$$":
            i += 1
            math_lines = []
            while i < n and lines[i].strip() != "$$":
                math_lines.append(lines[i])
                i += 1
            i += 1
            content = html.escape("\n".join(math_lines).strip())
            out.append(f'<pre class="math">{content}</pre>')
            continue

        # Single-line "$$ ... $$" math block.
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            content = html.escape(stripped[2:-2].strip())
            out.append(f'<pre class="math">{content}</pre>')
            i += 1
            continue

        # Heading.
        m = _HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            text = re.sub(r"\s+#+$", "", m.group(2).strip())
            plain = _strip_inline_markdown(text)
            hid = unique_id(f"{doc_id}-{_slugify(plain)}")
            headings.append((level, hid, plain))
            out.append(f'<h{level} id="{hid}">{inline_render(text, doc_id)}</h{level}>')
            i += 1
            continue

        # Horizontal rule.
        if _HR_RE.match(stripped):
            out.append("<hr>")
            i += 1
            continue

        # Blockquote.
        if stripped.startswith(">"):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                q = lines[i].strip()[1:]
                if q.startswith(" "):
                    q = q[1:]
                quote_lines.append(q)
                i += 1
            paragraphs, current = [], []
            for ql in quote_lines:
                if ql.strip() == "":
                    if current:
                        paragraphs.append(" ".join(current))
                        current = []
                else:
                    current.append(ql)
            if current:
                paragraphs.append(" ".join(current))
            inner = "".join(f"<p>{inline_render(p, doc_id)}</p>" for p in paragraphs)
            out.append(f"<blockquote>{inner}</blockquote>")
            continue

        # Table (header row + "---" separator row).
        if stripped.startswith("|") and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1].strip()):
            header_cells = _split_table_row(lines[i])
            i += 2
            body_rows = []
            while i < n and lines[i].strip().startswith("|"):
                body_rows.append(_split_table_row(lines[i]))
                i += 1
            thead = "<tr>" + "".join(
                f"<th>{inline_render(c, doc_id)}</th>" for c in header_cells
            ) + "</tr>"
            tbody = "".join(
                "<tr>" + "".join(f"<td>{inline_render(c, doc_id)}</td>" for c in row) + "</tr>"
                for row in body_rows
            )
            # Wrapped in a scrolling container rather than left to overflow the
            # page: a `<table>` does not wrap onto a narrower viewport the way
            # prose does, and several source tables here run 4-6 columns wide
            # (parameter glossaries, preset comparisons). `overflow-x` on the
            # wrapper keeps the scrollbar local to the table instead of
            # widening the whole page on mobile.
            out.append(
                f"<div class='table-wrap'><table><thead>{thead}</thead>"
                f"<tbody>{tbody}</tbody></table></div>"
            )
            continue

        # List (unordered or ordered); continuation lines are indented >=2 spaces.
        if _UL_RE.match(stripped) or _OL_RE.match(stripped):
            ordered = bool(_OL_RE.match(stripped))
            items: list[str] = []
            current_item: list[str] = []
            while i < n:
                raw = lines[i]
                ls = raw.strip()
                if ls == "":
                    break
                leading_ws = len(raw) - len(raw.lstrip(" "))
                mu, mo = _UL_RE.match(ls), _OL_RE.match(ls)
                if (mu or mo) and leading_ws == 0:
                    if current_item:
                        items.append(" ".join(current_item))
                    current_item = [mo.group(2) if mo else mu.group(2)]
                    i += 1
                elif leading_ws >= 2 and current_item:
                    current_item.append(ls)
                    i += 1
                else:
                    break
            if current_item:
                items.append(" ".join(current_item))
            tag = "ol" if ordered else "ul"
            li_parts = []
            for item in items:
                cm = _CHECKBOX_RE.match(item)
                if cm:
                    checked = cm.group(1).lower() == "x"
                    box = '<input type="checkbox" disabled' + (" checked" if checked else "") + "> "
                    li_parts.append(f"<li class='task'>{box}{inline_render(cm.group(2), doc_id)}</li>")
                else:
                    li_parts.append(f"<li>{inline_render(item, doc_id)}</li>")
            out.append(f"<{tag}>" + "".join(li_parts) + f"</{tag}>")
            continue

        # Paragraph: consume consecutive non-block-start lines.
        para_lines = []
        while i < n and not _is_block_start(lines[i]):
            para_lines.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline_render(' '.join(para_lines), doc_id)}</p>")

    return "\n".join(out), headings


# --------------------------------------------------------------------------- #
# Page assembly                                                               #
# --------------------------------------------------------------------------- #

_CSS = """
:root {
  color-scheme: light;
  --accent: #2a78d6; --border: #d8dee6; --bg-soft: #f5f7fa; --bg-page: #fff;
  --text: #1a1a1a; --text-muted: #666; --link: #2a5db0;
  --note-bg: #fffbe6; --note-border: #e0c040; --math-bg: #f0f4ec; --math-border: #c9d9b8;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --accent: #3987e5; --border: #33393f; --bg-soft: #1c2126; --bg-page: #121517;
    --text: #eaeaea; --text-muted: #9aa1a8; --link: #6fa8e8;
    --note-bg: #33290f; --note-border: #a6791f; --math-bg: #16201a; --math-border: #33472e;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --accent: #3987e5; --border: #33393f; --bg-soft: #1c2126; --bg-page: #121517;
  --text: #eaeaea; --text-muted: #9aa1a8; --link: #6fa8e8;
  --note-bg: #33290f; --note-border: #a6791f; --math-bg: #16201a; --math-border: #33472e;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       color: var(--text); line-height: 1.6; background: var(--bg-page); }
.layout { display: flex; align-items: flex-start; }
nav.toc { position: sticky; top: 0; height: 100vh; overflow-y: auto; flex: 0 0 300px;
          border-right: 1px solid var(--border); background: var(--bg-soft);
          padding: 1.2rem 1rem; font-size: .9rem; }
nav.toc h2 { font-size: 1rem; margin-top: 0; color: var(--accent); }
nav.toc ul { list-style: none; padding-left: 0; margin: 0 0 .8rem 0; }
nav.toc > ul > li { margin-bottom: .6rem; }
nav.toc a { color: var(--link); text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
nav.toc ul ul { padding-left: .9rem; margin-top: .3rem; border-left: 2px solid var(--border); }
nav.toc ul ul li { margin-bottom: .25rem; font-size: .85rem; }
#theme-toggle { display: block; margin: 0 0 1rem; width: 100%; border: 1px solid var(--border);
  background: transparent; color: var(--text-muted); border-radius: 6px; padding: .35rem .5rem;
  font-size: .8rem; cursor: pointer; }
#theme-toggle:hover { color: var(--text); border-color: var(--accent); }
main { flex: 1 1 auto; max-width: 900px; margin: 0 auto; padding: 2rem 2.2rem 4rem; min-width: 0; }
.masthead { margin-bottom: 1.4rem; }
.masthead h1 { margin-bottom: .3rem; }
.note { background: var(--note-bg); border-left: 4px solid var(--note-border); padding: .7rem 1rem;
        font-size: .92rem; border-radius: 0 6px 6px 0; }
.generated { color: var(--text-muted); font-style: italic; font-size: .85rem; margin: .3rem 0 0; }
section.doc-section { border: 1px solid var(--border); border-radius: 8px;
                       padding: 1.4rem 1.8rem 1.8rem; margin: 2.2rem 0; background: var(--bg-page); }
.source-tag { display: inline-block; background: var(--bg-soft); border: 1px solid var(--border);
              border-radius: 4px; padding: .15rem .55rem; font-size: .82rem; color: var(--text-muted);
              margin-bottom: 1rem; }
h1 { border-bottom: 3px solid var(--accent); padding-bottom: .3rem; scroll-margin-top: 1rem; }
h2 { border-bottom: 1px solid var(--border); padding-bottom: .2rem; margin-top: 2rem;
     scroll-margin-top: 1rem; }
h3, h4, h5, h6 { color: var(--text-muted); scroll-margin-top: 1rem; }
p { color: var(--text); }
/* Scrolls horizontally on its own rather than widening the page -- see the
   table-wrap comment where this markup is emitted. */
.table-wrap { overflow-x: auto; margin: .6rem 0 1.2rem 0; }
table { border-collapse: collapse; width: 100%; margin: 0; }
th, td { border: 1px solid var(--border); padding: .35rem .6rem; text-align: left;
         font-size: .92rem; }
th { background: var(--accent); color: #fff; }
tr:nth-child(even) td { background: var(--bg-soft); }
pre { background: var(--bg-soft); border: 1px solid var(--border); border-radius: 4px;
      padding: .8rem; overflow-x: auto; font-size: .88rem; }
pre.math { background: var(--math-bg); border-color: var(--math-border); font-style: italic; }
code { background: var(--bg-soft); padding: .1rem .35rem; border-radius: 3px;
       font-family: Consolas, "Courier New", monospace; font-size: .9em; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid var(--accent); margin: 1rem 0; padding: .3rem 1rem;
             background: var(--bg-soft); color: var(--text-muted); }
li.task { list-style: none; margin-left: -1.4rem; }
hr { border: none; border-top: 1px solid var(--border); margin: 1.6rem 0; }
a { color: var(--link); }
@media (max-width: 860px) {
  .layout { flex-direction: column; }
  nav.toc { position: static; height: auto; width: 100%; flex-basis: auto; }
  main { max-width: 100%; padding: 1.2rem 1rem 3rem; }
  th, td { font-size: .82rem; padding: .3rem .5rem; }
}
"""

_THEME_TOGGLE_JS = """
(function () {
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem('docs-theme'); } catch (e) {}
  if (saved) { root.setAttribute('data-theme', saved); }
  var btn = document.getElementById('theme-toggle');
  if (!btn) return;
  function current() {
    return root.getAttribute('data-theme')
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }
  btn.textContent = current() === 'dark' ? 'Light mode' : 'Dark mode';
  btn.addEventListener('click', function () {
    var next = current() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    btn.textContent = next === 'dark' ? 'Light mode' : 'Dark mode';
    try { localStorage.setItem('docs-theme', next); } catch (e) {}
  });
})();
"""


def _build_toc(sections: list[dict]) -> str:
    items = []
    for sec in sections:
        subitems = "".join(
            f'<li><a href="#{hid}">{html.escape(text)}</a></li>'
            for level, hid, text in sec["headings"] if level == 2
        )
        sub = f"<ul>{subitems}</ul>" if subitems else ""
        items.append(
            f'<li><a href="#{sec["doc_id"]}"><strong>{html.escape(sec["label"])}</strong></a>{sub}</li>'
        )
    return f'<h2>Contents</h2><ul>{"".join(items)}</ul>'


def build_documentation_html(project_root: Path) -> str:
    sections = []
    for rel_path, doc_id, label in DOCS:
        text = (project_root / rel_path).read_text(encoding="utf-8")
        body_html, headings = render_blocks(text.splitlines(), doc_id)
        sections.append({"doc_id": doc_id, "label": label, "body": body_html, "headings": headings})

    toc_html = _build_toc(sections)
    sections_html = "\n".join(
        f'<section class="doc-section" id="{s["doc_id"]}">'
        f'<div class="source-tag">Source: <code>{html.escape(s["label"])}</code></div>'
        f'{s["body"]}'
        f"</section>"
        for s in sections
    )

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Modelo-v0.1 — Consolidated Documentation</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        '<div class="layout">',
        f'<nav class="toc"><button id="theme-toggle" type="button">Dark mode</button>{toc_html}</nav>',
        "<main>",
        '<div class="masthead">',
        "<h1>Modelo-v0.1 — Consolidated Documentation</h1>",
        '<div class="note">Consolidated from README.md, CONTEXT.md, and docs/*.md '
        "— edit the source files, not this file, then regenerate "
        "(<code>python docs/build_docs.py</code>).</div>",
        "</div>",
        sections_html,
        "</main>",
        "</div>",
        f"<script>{_THEME_TOGGLE_JS}</script>",
        "</body></html>",
    ]
    return "\n".join(parts)


def main() -> None:
    html_out = build_documentation_html(PROJECT_ROOT)
    OUTPUT_PATH.write_text(html_out, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(html_out.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
