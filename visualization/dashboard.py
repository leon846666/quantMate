"""Aggregate several charts + summary into an HTML dashboard."""
from __future__ import annotations

from pathlib import Path


def render_html(
    out_path: Path | str,
    title: str,
    summary_md: str,
    images: list[Path | str],
) -> Path:
    """Write a self-contained HTML file referencing the PNGs by relative path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_tags = "\n".join(
        f'<div class="card"><img src="{Path(p).name}" alt=""/></div>' for p in images
    )
    html = f"""<!doctype html>
<html lang="zh-cn"><head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Microsoft YaHei",sans-serif;
     max-width:1000px;margin:2em auto;padding:0 1em;color:#222}}
h1{{border-bottom:2px solid #eee;padding-bottom:.3em}}
.card{{margin:1em 0;border:1px solid #eee;padding:1em;border-radius:6px;background:#fafafa}}
img{{max-width:100%}}
pre{{background:#f5f5f5;padding:1em;border-radius:6px;overflow-x:auto}}
table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:6px 12px}}
th{{background:#f0f0f0}}
</style>
</head><body>
<h1>{title}</h1>
<pre>{summary_md}</pre>
{img_tags}
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
