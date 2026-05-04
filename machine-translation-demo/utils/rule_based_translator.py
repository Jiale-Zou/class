from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_PUNCT_RE = re.compile(r"([.,!?;:()\[\]{}\"'“”‘’—\-])")


def _split_token_punct(token: str) -> List[str]:
    if not token:
        return []
    parts: List[str] = []
    for piece in _PUNCT_RE.split(token):
        if piece == "":
            continue
        parts.append(piece)
    return parts


def _is_punct(tok: str) -> bool:
    return bool(_PUNCT_RE.fullmatch(tok))


@lru_cache(maxsize=8)
def load_dictionary(path: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {
            "i": "我",
            "you": "你",
            "he": "他",
            "she": "她",
            "we": "我们",
            "they": "他们",
            "hello": "你好",
            "world": "世界",
            "good": "好",
            "morning": "早上好",
            "thank": "谢谢",
            "thanks": "谢谢",
            "computer": "计算机",
            "language": "语言",
            "model": "模型",
            "translation": "翻译",
        }
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k).lower(): str(v) for k, v in raw.items()}
        return {}
    except Exception:
        return {}


@dataclass(frozen=True)
class RuleBasedTranslator:
    dictionary: Dict[str, str]
    keep_unknown: bool = True

    def translate_tokens(self, tokens: Sequence[str]) -> List[str]:
        out: List[str] = []
        for tok in tokens:
            if _is_punct(tok):
                out.append(tok)
                continue
            key = tok.lower()
            mapped = self.dictionary.get(key)
            if mapped is None:
                out.append(tok if self.keep_unknown else "")
            else:
                out.append(mapped)
        return [t for t in out if t != ""]

    def translate(self, text: str) -> str:
        words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
        tokens: List[str] = []
        for w in words:
            tokens.extend(_split_token_punct(w))
        translated = self.translate_tokens(tokens)
        return self._join_zh_with_punct(translated)

    @staticmethod
    def _join_zh_with_punct(tokens: Iterable[str]) -> str:
        out: List[str] = []
        for tok in tokens:
            if not out:
                out.append(tok)
                continue
            if _is_punct(tok):
                out[-1] = out[-1] + tok
            else:
                if out[-1] and out[-1][-1].isascii() and tok and tok[0].isascii():
                    out.append(" " + tok)
                else:
                    out.append(tok)
        return "".join(out).strip()

