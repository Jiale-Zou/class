from __future__ import annotations

from mock_data import mock_relations
from ner import canonical_entity_label


def _pronoun_warnings(text: str) -> list[str]:
    pronouns = ["He", "She", "It", "he", "she", "it", "他", "她", "它"]
    if any(p in text for p in pronouns):
        return ["检测到代词（如 He/他），关系可能受指代影响。当前仅做非常简化的近邻回填。"]
    return []


def _relation_type_for_verb(token) -> str | None:
    lemma = (getattr(token, "lemma_", "") or getattr(token, "text", "")).lower()
    text = (getattr(token, "text", "") or "").lower()

    founded_lemmas = {"found", "cofound", "create", "establish", "launch", "start"}
    work_lemmas = {"join", "work", "employ", "hire", "serve"}
    born_lemmas = {"bear", "born", "come"}

    if lemma in founded_lemmas or "found" in lemma or text in {"创立", "创建", "成立", "创办"}:
        return "FOUNDED_BY"
    if lemma in work_lemmas or text in {"加入", "任职", "就职", "入职"}:
        return "WORKS_AT"
    if lemma in born_lemmas or text in {"出生", "生于", "来自"}:
        return "BORN_IN"
    return None


def _pick_mentions(doc, entities: list[dict]) -> tuple[dict[int, dict], list[dict]]:
    mentions: list[dict] = []
    token_to_mention: dict[int, dict] = {}

    for e in entities:
        start = int(e.get("start", 0))
        end = int(e.get("end", 0))
        raw_label = str(e.get("label", "")).strip()
        label = canonical_entity_label(raw_label) or raw_label
        if label not in {"PERSON", "ORG", "LOC"}:
            continue

        span = doc.char_span(start, end, alignment_mode="expand")
        if span is None:
            continue

        mention = {"text": str(e.get("text", "")).strip() or span.text, "label": label, "span": span}
        mentions.append(mention)

        span_len = int(span.end - span.start)
        for t in span:
            existing = token_to_mention.get(int(t.i))
            if existing is None:
                token_to_mention[int(t.i)] = mention
            else:
                existing_span_len = int(existing["span"].end - existing["span"].start)
                if span_len > existing_span_len:
                    token_to_mention[int(t.i)] = mention

    return token_to_mention, mentions


def _fallback_argument_mention(token) -> dict | None:
    pos = getattr(token, "pos_", "")
    if pos not in {"PROPN", "NOUN"}:
        return None

    left = getattr(token, "left_edge", token)
    right = getattr(token, "right_edge", token)
    try:
        span = token.doc[int(left.i) : int(right.i) + 1]
        text = span.text.strip()
    except Exception:
        text = (getattr(token, "text", "") or "").strip()

    if not text:
        return None
    return {"text": text, "label": "UNKNOWN"}


def _closest_prior_person(person_mentions: list[dict], token_index: int) -> dict | None:
    best = None
    best_end = -1
    for m in person_mentions:
        span = m.get("span")
        if span is None:
            continue
        end = int(getattr(span, "end", -1))
        if end <= token_index and end > best_end:
            best = m
            best_end = end
    return best


def extract_relations(
    text: str, entities: list[dict], mode: str = "mock", doc=None
) -> tuple[list[dict], list[str]]:
    if mode == "mock":
        return mock_relations(text, entities)

    warnings = _pronoun_warnings(text)
    if doc is None:
        return [], warnings + ["未提供 spaCy 解析结果，无法进行基于依存句法的关系抽取。"]

    has_dep = False
    try:
        has_dep = bool(doc.has_annotation("DEP"))
    except Exception:
        has_dep = False

    if not has_dep:
        return [], warnings + ["当前 spaCy 模型不含依存句法分析器，关系抽取仅输出空结果。"]

    token_to_mention, mentions = _pick_mentions(doc, entities)
    if not token_to_mention:
        return [], warnings

    person_mentions = [m for m in mentions if m.get("label") == "PERSON"]

    subject_deps = {"nsubj", "nsubjpass", "csubj", "agent"}
    object_deps = {"dobj", "obj", "attr", "oprd", "pobj", "dative", "obl"}

    triples: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for v in doc:
        if getattr(v, "pos_", "") not in {"VERB", "AUX"}:
            continue

        rel = _relation_type_for_verb(v)
        if not rel:
            continue

        subj_mentions: list[dict] = []
        obj_mentions: list[dict] = []

        for ch in getattr(v, "children", []):
            if getattr(ch, "dep_", "") in subject_deps:
                m = token_to_mention.get(int(ch.i))
                if m:
                    subj_mentions.append(m)

            if getattr(ch, "dep_", "") in object_deps:
                m = token_to_mention.get(int(ch.i))
                if m:
                    obj_mentions.append(m)
                else:
                    fallback = _fallback_argument_mention(ch)
                    if fallback:
                        obj_mentions.append(fallback)

            if getattr(ch, "dep_", "") == "prep":
                for gch in getattr(ch, "children", []):
                    if getattr(gch, "dep_", "") in {"pobj", "obl"}:
                        m = token_to_mention.get(int(gch.i))
                        if m:
                            obj_mentions.append(m)
                        else:
                            fallback = _fallback_argument_mention(gch)
                            if fallback:
                                obj_mentions.append(fallback)

        if not subj_mentions:
            head = token_to_mention.get(int(getattr(v, "head", v).i))
            if head:
                subj_mentions.append(head)

        if not subj_mentions and person_mentions:
            prior = _closest_prior_person(person_mentions, int(v.i))
            if prior:
                subj_mentions.append(prior)

        for subj in subj_mentions:
            for obj in obj_mentions:
                subj_text = str(subj["text"]).strip()
                obj_text = str(obj["text"]).strip()
                if not subj_text or not obj_text or subj_text == obj_text:
                    continue

                obj_label = str(obj.get("label", "UNKNOWN")).strip() or "UNKNOWN"
                subj_label = str(subj.get("label", "UNKNOWN")).strip() or "UNKNOWN"

                if rel == "FOUNDED_BY" and not (subj_label == "PERSON" and obj_label in {"ORG", "UNKNOWN"}):
                    continue
                if rel == "WORKS_AT" and not (subj_label == "PERSON" and obj_label in {"ORG", "UNKNOWN"}):
                    continue
                if rel == "BORN_IN" and not (subj_label == "PERSON" and obj_label in {"LOC", "UNKNOWN"}):
                    continue

                key = (subj_text, rel, obj_text)
                if key in seen:
                    continue
                seen.add(key)
                triples.append({"subject": subj_text, "relation": rel, "object": obj_text})

    if not triples:
        fallback_relations, fallback_warnings = mock_relations(text, entities)
        if fallback_relations:
            return fallback_relations, warnings + ["依存句法未命中关系模式，已回退到关键词规则作为教学兜底。"] + fallback_warnings

    return triples, warnings

