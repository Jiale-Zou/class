from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from config import RANDOM_SEED


def _sample_text_pool() -> Dict[str, List[str]]:
    return {
        "Positive": [
            "质量很好，物超所值，推荐购买！",
            "物流很快，包装完好，体验很棒。",
            "客服回复及时，问题解决得很专业。",
            "颜值在线，使用顺手，下次还会回购。",
            "比预期更好，性价比很高。",
        ],
        "Negative": [
            "做工粗糙，有瑕疵，体验很差。",
            "电池一小时就没电了，太失望了。",
            "包装破损，送到时已经压坏了。",
            "物流太慢了，等了好几天。",
            "客服态度一般，沟通成本很高。",
        ],
        "Neutral": [
            "中规中矩，没什么特别的。",
            "目前用着还行，后续再观察。",
            "外观一般，功能符合描述。",
            "还没用多久，先给个中评。",
            "价格正常，活动力度一般。",
        ],
    }


def generate_mock_comments(
    n: int = 60,
    seed: int = RANDOM_SEED,
    start_days_ago: int = 14,
    platforms: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    rng = random.Random(seed)
    platforms = platforms or ["电商", "社交媒体"]
    categories = categories or ["手机", "耳机", "电脑", "家电", "美妆", "服饰"]

    pool = _sample_text_pool()
    labels = ["Positive", "Neutral", "Negative"]
    weights = [0.45, 0.25, 0.30]

    start = datetime.now() - timedelta(days=int(start_days_ago))
    rows = []

    for idx in range(1, int(n) + 1):
        label = rng.choices(labels, weights=weights, k=1)[0]
        text = rng.choice(pool[label])
        ts = start + timedelta(minutes=rng.randint(0, int(start_days_ago) * 24 * 60))
        rows.append(
            {
                "id": idx,
                "platform": rng.choice(platforms),
                "category": rng.choice(categories),
                "created_at": ts,
                "text": text,
                "seed_label": label,
            }
        )

    df = pd.DataFrame(rows).sort_values("created_at", ascending=False).reset_index(drop=True)
    return df
