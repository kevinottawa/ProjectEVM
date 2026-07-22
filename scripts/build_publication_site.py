"""Build the static HTML reader for the EVM publication package."""

from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SOURCE = DOCS / "EVM_Paper.md"
OUTPUT = DOCS / "paper.html"


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(
        source,
        extensions=["fenced_code", "sane_lists", "tables", "toc"],
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="EVM paper reader: short-horizon working-set residency for MoE inference.">
  <title>EVM Paper</title>
  <link rel="stylesheet" href="site.css">
</head>
<body class="paper-page">
  <header class="site-header">
    <a class="wordmark" href="index.html">EVM <span>Research</span></a>
    <nav aria-label="Primary navigation">
      <a href="index.html#results">Results</a>
      <a href="index.html#artifacts">Artifacts</a>
      <a href="EVM_Paper.md">Markdown source</a>
    </nav>
  </header>
  <main class="paper-shell">
    <article class="paper-content">
      {body}
    </article>
  </main>
  <footer class="site-footer">EVM publication prototype by Kevin Price. Paper and figures: CC BY 4.0. Code: MIT.</footer>
</body>
</html>
"""
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Built {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
