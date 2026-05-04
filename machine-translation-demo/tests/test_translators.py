from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils.evaluator import bleu_score_with_details
from utils.rule_based_translator import RuleBasedTranslator


def test_rule_based_unknown_kept():
    t = RuleBasedTranslator(dictionary={"hello": "你好"})
    out = t.translate("Hello Alice")
    assert "你好" in out
    assert "Alice" in out


def test_rule_based_punct_attached():
    t = RuleBasedTranslator(dictionary={"hello": "你好", "world": "世界"})
    out = t.translate("Hello, world!")
    assert "你好," in out or "你好，" in out
    assert out.endswith("!")


def test_bleu_perfect_match():
    d = bleu_score_with_details("a b c", "a b c", max_n=4, smooth=True)
    assert 0.99 <= d.bleu <= 1.0
    assert d.bp == 1.0


def test_bleu_brevity_penalty():
    d = bleu_score_with_details("a b c d e f", "a b c", max_n=2, smooth=True)
    assert d.bp < 1.0
    assert d.bleu <= d.bp

