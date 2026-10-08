"""Content-aware result normalizer for Agent Reach reliability layer.

Pipeline:
    Raw Result → Clean → Extract useful content → Limit size → Gemma

Reduces oversized web reader outputs (e.g. ~30k chars from Jina Reader) to a
concise, coherent, and query-relevant excerpt under `max_chars`, preserving
document structure and semantic signal without heavy external dependencies.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedResult:
    """Result of the normalization pipeline."""

    text: str
    original_chars: int
    final_chars: int
    reduction_pct: float
    truncated: bool
    metadata: dict[str, Any] = field(default_factory=dict)


# Common error page indicators
_ERROR_PATTERNS = [
    re.compile(r"\b403\s+forbidden\b", re.IGNORECASE),
    re.compile(r"\b404\s+not\s+found\b", re.IGNORECASE),
    re.compile(r"\baccess\s+denied\b", re.IGNORECASE),
    re.compile(r"\battention\s+required!\s*\|\s*cloudflare\b", re.IGNORECASE),
    re.compile(r"\bjust\s+a\s+moment\.\.\.\b", re.IGNORECASE),
    re.compile(r"\bsecurity\s+verification\b", re.IGNORECASE),
    re.compile(r"\bverify\s+you\s+are\s+a\s+human\b", re.IGNORECASE),
    re.compile(r"\bcaptcha\b", re.IGNORECASE),
    re.compile(r"\bblocked\s+by\s+cloudflare\b", re.IGNORECASE),
    re.compile(r"\bray\s+id:\s*[0-9a-f]+\b", re.IGNORECASE),
    re.compile(r"\brequiring\s+captcha\b", re.IGNORECASE),
]

# Boilerplate patterns to strip out
_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^(?:we\s+use\s+cookies|this\s+site\s+uses\s+cookies|accept\s+cookies|cookie\s+settings|manage\s+cookies)", re.IGNORECASE),
    re.compile(r"^(?:privacy\s+policy|terms\s+of\s+service|terms\s+of\s+use|disclaimers?|contact\s+us|code\s+of\s+conduct)$", re.IGNORECASE),
    re.compile(r"^(?:skip\s+to\s+(?:content|main|navigation)|back\s+to\s+top)$", re.IGNORECASE),
    re.compile(r"^(?:subscribe\s+to\s+our\s+newsletter|sign\s+up\s+for\s+free|newsletter\s+signup)", re.IGNORECASE),
    re.compile(r"^(?:share\s+on\s+(?:twitter|x|facebook|linkedin|reddit)|follow\s+us\s+on)", re.IGNORECASE),
    re.compile(r"^(?:all\s+rights\s+reserved|copyright\s+©|©\s*\d{4})", re.IGNORECASE),
]

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "how", "what", "which", "who",
    "whom", "this", "that", "these", "those", "it", "its", "as", "if", "than",
}


def _detect_error(text: str) -> bool:
    """Return True if text appears to be an HTTP error or bot challenge page."""
    sample = text[:4000]
    return any(p.search(sample) for p in _ERROR_PATTERNS)


def _extract_jina_headers(raw: str) -> tuple[str | None, str | None, str]:
    """Extract Title and URL Source from Jina Reader markdown output."""
    title: str | None = None
    url: str | None = None
    content = raw

    # Match Title header
    m_title = re.search(r"^Title:\s*(.+)$", raw, re.MULTILINE)
    if m_title:
        title = m_title.group(1).strip()

    # Match URL Source header
    m_url = re.search(r"^URL Source:\s*(.+)$", raw, re.MULTILINE)
    if m_url:
        url = m_url.group(1).strip()

    # If "Markdown Content:" is present, take everything after it
    m_content = re.search(r"^Markdown Content:\s*\n", raw, re.MULTILINE)
    if m_content:
        content = raw[m_content.end():]
    else:
        # Strip Title / URL Source lines from content if they appear at top
        content = re.sub(r"^(?:Title:[^\n]*\n|URL Source:[^\n]*\n)+", "", content)

    return title, url, content


def _clean_text(text: str) -> str:
    """Remove HTML tags, scripts, styles, base64 blobs, and markdown images."""
    # Remove script, style, and noscript blocks
    text = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<noscript\b[^<]*(?:(?!<\/noscript>)<[^<]*)*<\/noscript>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<!--[\s\S]*?-->", "", text)

    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Strip base64 data blobs
    text = re.sub(r"data:[^;]+;base64,[A-Za-z0-9+/=]+", "", text)

    # Strip markdown image syntax ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)
    text = re.sub(r"!\[[^\]]*\]\[[^\]]*\]", "", text)

    # Strip tracking URL params (e.g. ?utm_source=... or &utm_medium=...)
    text = re.sub(r"[\?&](?:utm_[a-z]+|fbclid|gclid|ref)=[a-zA-Z0-9_\-\.\%]+", "", text)

    # Clean lines and filter boilerplate
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    seen_lines: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        # Check boilerplate
        is_bp = any(bp.search(stripped) for bp in _BOILERPLATE_LINE_PATTERNS)
        if is_bp:
            continue

        # Deduplicate exact lines (except blank lines and markdown headings/list items)
        if len(stripped) > 40:
            norm_key = re.sub(r"\s+", " ", stripped).lower()
            if norm_key in seen_lines:
                continue
            seen_lines.add(norm_key)

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse repeated link lists (e.g. 5 or more consecutive bullet links)
    def _collapse_links(match: re.Match[str]) -> str:
        matched_text = match.group(0)
        link_count = len(re.findall(r"\[([^\]]+)\]\([^\)]+\)", matched_text))
        if link_count >= 5:
            return f"\n[{link_count} links omitted]\n"
        return matched_text

    link_list_pattern = re.compile(
        r"(?:^[ \t]*[\*\-\d\.]+\s*\[[^\]]+\]\([^\)]+\)[ \t]*\n){5,}",
        re.MULTILINE,
    )
    text = link_list_pattern.sub(_collapse_links, text)

    # Collapse excessive newlines and spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_query_keywords(query: str | None) -> set[str]:
    """Extract lowercased query terms excluding common stopwords."""
    if not query:
        return set()
    words = re.findall(r"\b[A-Za-z0-9_-]+\b", query.lower())
    return {w for w in words if len(w) > 1 and w not in _STOPWORDS}


def _score_block(
    block: str,
    index: int,
    query_keywords: set[str],
) -> float:
    """Score a content block based on relevance, position, and information density."""
    score = 0.0
    text_lower = block.lower()
    words_in_block = set(re.findall(r"\b[A-Za-z0-9_-]+\b", text_lower))

    # Keyword overlap
    if query_keywords:
        overlap = query_keywords & words_in_block
        score += len(overlap) * 3.5

    # Position bonus (intro / lead paragraphs)
    if index == 0:
        score += 5.0
    elif index == 1:
        score += 2.5
    elif index == 2:
        score += 1.0

    # Heading bonus
    if block.startswith("#"):
        score += 2.0
        # Extra bonus if heading has query terms
        if query_keywords and (query_keywords & words_in_block):
            score += 4.0

    # Information density bonuses
    # Numbers / percentages
    if re.search(r"\b\d+(?:\.\d+)?%?\b", block):
        score += 1.0

    # Code blocks
    if "```" in block or "`" in block:
        score += 2.0

    # Fact lists (bullet or numbered)
    if re.search(r"(?:^[ \t]*[\*\-]\s|^\d+\.\s)", block, re.MULTILINE):
        score += 1.5

    # Penalties
    char_len = len(block)
    if char_len < 30:
        score -= 2.0

    # Link heaviness penalty
    link_chars = sum(len(m.group(0)) for m in re.finditer(r"\[([^\]]+)\]\([^\)]+\)", block))
    if char_len > 0 and (link_chars / char_len) > 0.4:
        score -= 4.0

    # Wikipedia reference lists can contain many query keywords and links
    # while providing no explanatory content for the judge.
    if _is_reference_block(block):
        score -= 100.0

    return score


def _is_low_signal_intro(block: str) -> bool:
    """Return True for document front matter that should not be kept alone."""
    normalized = re.sub(r"\s+", " ", block).strip().lower()
    return normalized.startswith("from wikipedia, the free encyclopedia")


def _is_reference_block(block: str) -> bool:
    """Return True for citation-heavy reference entries."""
    text_lower = block.lower()
    citation_count = len(re.findall(r"\[[^\]]+\]\([^)]+\)", block))
    return citation_count >= 3 and (
        re.match(r"^\d+\.", block) or "retrieved" in text_lower
    )


def _cut_to_sentence_boundary(text: str, budget: int) -> str:
    """Cut text at a sentence boundary within budget, avoiding mid-word cuts."""
    if len(text) <= budget:
        return text

    marker = " […]"
    effective_budget = budget - len(marker)
    if effective_budget <= 0:
        return marker.strip()

    candidate = text[:effective_budget]

    # Look for sentence terminators (. ! ? or newline)
    terminators = [". ", "!\n", "?\n", ".\n", "! ", "? ", "\n\n", "\n"]
    best_cut = -1
    for term in terminators:
        idx = candidate.rfind(term)
        if idx != -1:
            best_cut = max(best_cut, idx + (1 if term.startswith(".") or term.startswith("!") or term.startswith("?") else 0))

    if best_cut >= max(20, effective_budget // 3):
        return candidate[:best_cut].rstrip() + marker

    # Fall back to word boundary
    last_space = candidate.rfind(" ")
    if last_space > 0:
        return candidate[:last_space].rstrip() + marker

    # Fall back if no space found
    return candidate.rstrip() + marker


def normalize(
    raw: str,
    *,
    query: str | None = None,
    max_chars: int = 4000,
    source: str | None = None,
) -> NormalizedResult:
    """Normalize and summarize raw web/backend text for LLM evaluation.

    Parameters
    ----------
    raw:
        The raw input string (HTML, Jina Reader markdown, or plain text).
    query:
        Optional user query used to prioritize relevant sections.
    max_chars:
        Maximum allowed characters for the output text.
    source:
        Optional source identifier or URL.

    Returns
    -------
    NormalizedResult:
        Cleaned, size-capped, and structured result with metadata.
    """
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""

    original_chars = len(raw)

    if not raw.strip():
        return NormalizedResult(
            text="",
            original_chars=original_chars,
            final_chars=0,
            reduction_pct=0.0,
            truncated=False,
            metadata={
                "title": None,
                "url": source,
                "looks_like_error": False,
                "sections_kept": 0,
                "sections_dropped": 0,
            },
        )

    looks_like_error = _detect_error(raw)

    # Extract Jina headers if present
    extracted_title, extracted_url, content_body = _extract_jina_headers(raw)
    title = extracted_title
    url = extracted_url or source

    # If title wasn't in Jina header, try first Markdown header
    if not title:
        m_h1 = re.search(r"^#\s+(.+)$", content_body, re.MULTILINE)
        if m_h1:
            title = m_h1.group(1).strip()

    # Clean the content
    cleaned = _clean_text(content_body)

    # Build document header if title is available and not already at the start
    header_block = ""
    if title:
        first_line = cleaned.lstrip().split("\n", 1)[0]
        title_core = title.split(" - ")[0].strip().lower()
        if not (first_line.startswith("# ") and title_core in first_line.lower()):
            header_block = f"# {title}\n\n"

    # If input is already under max_chars, return it cleaned
    full_cleaned = (header_block + cleaned).strip()
    if len(full_cleaned) <= max_chars:
        final_chars = len(full_cleaned)
        red_pct = round((1.0 - final_chars / original_chars) * 100, 1) if original_chars > 0 else 0.0
        return NormalizedResult(
            text=full_cleaned,
            original_chars=original_chars,
            final_chars=final_chars,
            reduction_pct=red_pct,
            truncated=False,
            metadata={
                "title": title,
                "url": url,
                "looks_like_error": looks_like_error,
                "sections_kept": 1,
                "sections_dropped": 0,
            },
        )

    # Split into paragraph / section blocks
    raw_blocks = [b.strip() for b in cleaned.split("\n\n") if b.strip()]
    if not raw_blocks:
        truncated_text = _cut_to_sentence_boundary(header_block.strip(), max_chars)
        final_chars = len(truncated_text)
        red_pct = round((1.0 - final_chars / original_chars) * 100, 1) if original_chars > 0 else 0.0
        return NormalizedResult(
            text=truncated_text,
            original_chars=original_chars,
            final_chars=final_chars,
            reduction_pct=red_pct,
            truncated=True,
            metadata={
                "title": title,
                "url": url,
                "looks_like_error": looks_like_error,
                "sections_kept": 0,
                "sections_dropped": 0,
            },
        )

    # Score each block
    query_keywords = _extract_query_keywords(query)
    scored_blocks: list[tuple[int, float, str]] = []
    for idx, block in enumerate(raw_blocks):
        score = _score_block(block, idx, query_keywords)
        scored_blocks.append((idx, score, block))

    # Selection budget
    remaining_budget = max_chars - len(header_block)
    if remaining_budget <= 50:
        remaining_budget = max_chars
        header_block = ""

    # Keep the first meaningful block rather than source-specific front matter.
    intro_index = next(
        (idx for idx, block in enumerate(raw_blocks) if not _is_low_signal_intro(block)),
        0,
    )
    selected_indices: set[int] = {intro_index}
    used_chars = len(raw_blocks[intro_index]) + 2  # account for separator "\n\n"

    # Rank remaining blocks by score descending
    remaining_ranked = sorted(
        [item for item in scored_blocks if item[0] != intro_index],
        key=lambda item: item[1],
        reverse=True,
    )

    for idx, score, block in remaining_ranked:
        if _is_reference_block(block):
            continue
        block_len = len(block) + 2
        if used_chars + block_len <= remaining_budget:
            selected_indices.add(idx)
            used_chars += block_len
        elif used_chars < remaining_budget - 100:
            # Can fit a partial cut
            selected_indices.add(idx)
            break

    # Restore original document order
    sorted_kept = sorted(selected_indices)
    chosen_blocks: list[str] = []
    current_len = len(header_block)

    for i, idx in enumerate(sorted_kept):
        block = raw_blocks[idx]
        sep = "\n\n" if i > 0 else ""
        allowed = max_chars - current_len - len(sep)

        if len(block) <= allowed:
            chosen_blocks.append(block)
            current_len += len(block) + len(sep)
        else:
            cut = _cut_to_sentence_boundary(block, allowed)
            if cut.strip():
                chosen_blocks.append(cut)
            break

    result_text = header_block + "\n\n".join(chosen_blocks)
    result_text = result_text.strip()

    final_chars = len(result_text)
    red_pct = round((1.0 - final_chars / original_chars) * 100, 1) if original_chars > 0 else 0.0

    return NormalizedResult(
        text=result_text,
        original_chars=original_chars,
        final_chars=final_chars,
        reduction_pct=red_pct,
        truncated=True,
        metadata={
            "title": title,
            "url": url,
            "looks_like_error": looks_like_error,
            "sections_kept": len(chosen_blocks),
            "sections_dropped": len(raw_blocks) - len(chosen_blocks),
            "total_sections": len(raw_blocks),
            "query_keywords": sorted(list(query_keywords)),
        },
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    else:
        content = sys.stdin.read()

    res = normalize(content, max_chars=4000)
    print("=" * 60)
    print("NORMALIZATION REPORT")
    print("=" * 60)
    print(f"Original size:   {res.original_chars:,} chars")
    print(f"Final size:      {res.final_chars:,} chars")
    print(f"Reduction:       {res.reduction_pct}%")
    print(f"Truncated:       {res.truncated}")
    print(f"Looks like error:{res.metadata.get('looks_like_error')}")
    print(f"Sections kept:   {res.metadata.get('sections_kept')} / {res.metadata.get('total_sections')}")
    print("=" * 60)
    print("TEXT PREVIEW (first 500 chars):")
    print(res.text[:500])
