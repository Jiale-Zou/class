from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    label: str
    text: str


def _find_all(text: str, term: str, label: str) -> Iterable[EntitySpan]:
    for m in re.finditer(re.escape(term), text):
        yield EntitySpan(start=m.start(), end=m.end(), label=label, text=text[m.start() : m.end()])


def mock_ner(text: str) -> list[dict]:
    entities: list[EntitySpan] = []

    en_terms = [
        ("Steve Jobs", "PERSON"),
        ("Bill Gates", "PERSON"),
        ("Apple", "ORG"),
        ("Microsoft", "ORG"),
        ("California", "LOC"),
        ("Seattle", "LOC"),
    ]
    zh_terms = [
        ("乔布斯", "PERSON"),
        ("比尔·盖茨", "PERSON"),
        ("苹果公司", "ORG"),
        ("苹果", "ORG"),
        ("微软", "ORG"),
        ("北京", "LOC"),
        ("加州", "LOC"),
    ]

    for term, label in en_terms + zh_terms:
        entities.extend(list(_find_all(text, term, label)))

    entities = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
    normalized: list[EntitySpan] = []
    last_end = -1
    for ent in entities:
        if ent.start < last_end:
            continue
        normalized.append(ent)
        last_end = ent.end

    return [
        {"start": e.start, "end": e.end, "label": e.label, "text": e.text}
        for e in normalized
    ]


def mock_relations(text: str, entities: list[dict]) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    pronouns = ["He", "She", "It", "he", "she", "it", "他", "她", "它"]
    if any(p in text for p in pronouns):
        warnings.append("检测到代词（如 He/他），关系可能受指代影响。当前使用简单规则进行近似推断。")

    people = [e["text"] for e in entities if e["label"] == "PERSON"]
    orgs = [e["text"] for e in entities if e["label"] == "ORG"]
    locs = [e["text"] for e in entities if e["label"] == "LOC"]

    relations: list[dict] = []
    if people and orgs and re.search(r"\b(founded|found|co[- ]?founded|created)\b|创立|创建|成立", text, re.I):
        relations.append({"subject": people[0], "relation": "FOUNDED_BY", "object": orgs[0]})
    if people and locs and re.search(r"\b(born in|from)\b|出生于|来自", text, re.I):
        relations.append({"subject": people[0], "relation": "BORN_IN", "object": locs[0]})
    if people and orgs and re.search(r"\b(joined|works at|work at)\b|加入|任职|就职", text, re.I):
        relations.append({"subject": people[0], "relation": "WORKS_AT", "object": orgs[0]})

    return relations, warnings

