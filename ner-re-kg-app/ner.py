from __future__ import annotations

import html
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSpan:
    token: str
    start: int
    end: int


TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\s]", re.UNICODE)


def tokenize_with_offsets(text: str) -> list[TokenSpan]:
    return [TokenSpan(token=m.group(0), start=m.start(), end=m.end()) for m in TOKEN_PATTERN.finditer(text)]


def bio_tag(tokens: list[TokenSpan], entities: list[dict]) -> list[str]:
    tags = ["O"] * len(tokens)
    for ent in sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"]))):
        ent_start = int(ent["start"])
        ent_end = int(ent["end"])
        label = str(ent["label"])

        covered = [
            i
            for i, t in enumerate(tokens)
            if not (t.end <= ent_start or t.start >= ent_end)
        ]
        if not covered:
            continue
        tags[covered[0]] = f"B-{label}"
        for i in covered[1:]:
            tags[i] = f"I-{label}"
    return tags


def format_bio(tokens: list[TokenSpan], bio_tags: list[str]) -> str:
    cols = [max(len(t.token), len(tag), 2) for t, tag in zip(tokens, bio_tags)]
    token_line = " ".join(t.token.ljust(w) for t, w in zip(tokens, cols))
    tag_line = " ".join(tag.ljust(w) for tag, w in zip(bio_tags, cols))
    return token_line + "\n" + tag_line


def render_highlighted_html(text: str, entities: list[dict], label_to_color: dict[str, str]) -> str:
    safe_entities = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
    normalized: list[dict] = []
    last_end = -1
    for e in safe_entities:
        s = int(e["start"])
        ed = int(e["end"])
        if s < last_end:
            continue
        normalized.append(e)
        last_end = ed

    parts: list[str] = []
    cursor = 0
    for ent in normalized:
        s = int(ent["start"])
        e = int(ent["end"])
        label = str(ent["label"])
        color = label_to_color.get(label, "#E0E0E0")

        if cursor < s:
            parts.append(html.escape(text[cursor:s]))

        span_text = html.escape(text[s:e])
        parts.append(
            f'<span style="background-color:{color}; padding:0 2px; border-radius:3px;">{span_text}</span>'
        )
        cursor = e

    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))

    return "".join(parts).replace("\n", "<br/>")


SPACY_LABEL_TO_CANONICAL = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "PRODUCT": "UNKNOWN",
    "NORP": "UNKNOWN",
    "EVENT": "UNKNOWN",
    "WORK_OF_ART": "UNKNOWN",
}


def canonical_entity_label(label: str) -> str | None:
    normalized = (label or "").strip()
    if not normalized:
        return None
    return SPACY_LABEL_TO_CANONICAL.get(normalized)


def spacy_tokens(doc) -> list[TokenSpan]:
    return [TokenSpan(token=t.text, start=int(t.idx), end=int(t.idx + len(t.text))) for t in doc]


def spacy_entities(doc) -> list[dict]:
    entities: list[dict] = []
    for ent in getattr(doc, "ents", []):
        label = canonical_entity_label(getattr(ent, "label_", ""))
        if label not in {"PERSON", "ORG", "LOC", "UNKNOWN"}:
            continue
        entities.append(
            {
                "start": int(ent.start_char),
                "end": int(ent.end_char),
                "label": label,
                "text": ent.text,
            }
        )
    entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    return entities


def spacy_bio(doc) -> list[str]:
    tags: list[str] = []
    for t in doc:
        iob = getattr(t, "ent_iob_", "O")
        ent_type = getattr(t, "ent_type_", "")
        label = canonical_entity_label(ent_type)
        if iob == "O" or not label or label not in {"PERSON", "ORG", "LOC", "UNKNOWN"}:
            tags.append("O")
        else:
            tags.append(f"{iob}-{label}")
    return tags

