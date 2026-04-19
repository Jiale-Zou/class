from collections import Counter
from typing import Dict, List

import torch


def safe_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def compute_gpt2_perplexities(
    sentences: List[str],
    model,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 128,
    device: torch.device = torch.device("cpu"),
    standardize_length: bool = True,
) -> List[float]:
    results: List[float] = []
    if not sentences:
        return results

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        label_mask = attention_mask[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        neg_log_likelihood = -(token_log_probs * label_mask).sum(dim=1)
        token_counts = label_mask.sum(dim=1).clamp_min(1)

        if standardize_length:
            losses = neg_log_likelihood / token_counts
        else:
            losses = neg_log_likelihood

        batch_ppl = torch.exp(losses).detach().cpu().tolist()
        results.extend([float(x) for x in batch_ppl])

    return results


def detect_repeated_ngrams(text: str, n: int = 3) -> Dict[str, object]:
    tokens = text.split()
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = {k: v for k, v in counts.items() if v > 1}
    return {
        "total_ngrams": len(grams),
        "repeated_ngrams": len(repeated),
        "top_repeated": [" ".join(k) for k, _ in counts.most_common(5) if counts[k] > 1],
    }
