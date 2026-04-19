import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CharRNNConfig:
    model_type: str = "lstm"
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = False
    seq_len: int = 40
    epochs: int = 50
    lr: float = 0.01
    optimizer: str = "adam"
    lr_decay: bool = False


class _CharLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        cell: str,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.cell_type = cell
        if cell == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = nn.RNN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
                nonlinearity="tanh",
            )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, vocab_size)

    def forward(self, x: torch.Tensor, state=None) -> Tuple[torch.Tensor, object]:
        emb = self.embedding(x)
        out, state = self.rnn(emb, state)
        logits = self.proj(out)
        return logits, state


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _build_vocab(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return vocab, stoi, itos


def _make_batches(encoded: List[int], seq_len: int, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pairs = []
    if len(encoded) <= seq_len + 1:
        return pairs

    for i in range(0, len(encoded) - seq_len - 1, seq_len):
        x = encoded[i : i + seq_len]
        y = encoded[i + 1 : i + seq_len + 1]
        pairs.append(
            (
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
            )
        )

    random.shuffle(pairs)
    batches = []
    for i in range(0, len(pairs), batch_size):
        bx = torch.stack([p[0] for p in pairs[i : i + batch_size]], dim=0)
        by = torch.stack([p[1] for p in pairs[i : i + batch_size]], dim=0)
        batches.append((bx, by))
    return batches


def _select_batch_size() -> int:
    return 64 if torch.cuda.is_available() else 16


def train_char_model(
    train_text: str,
    cfg: CharRNNConfig,
) -> Generator[Tuple[int, int, float, Dict, Dict], None, None]:
    train_text = train_text or ""
    if len(train_text) < max(cfg.seq_len * 2, 50):
        raise ValueError("训练语料过短，建议至少几百字符。")

    vocab, stoi, itos = _build_vocab(train_text)
    encoded = [stoi[ch] for ch in train_text]

    device = _device()
    batch_size = _select_batch_size()
    batches = _make_batches(encoded, seq_len=cfg.seq_len, batch_size=batch_size)
    if not batches:
        raise ValueError("无法构建训练样本，请增大语料长度或减小序列长度。")

    model = _CharLM(
        vocab_size=len(vocab),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional,
        cell=cfg.model_type,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = None
    if cfg.lr_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(cfg.epochs // 3, 1), gamma=0.5)

    total_steps = cfg.epochs * len(batches)
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        for bx, by in batches:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(bx)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), by.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            model_state = {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": asdict(cfg),
            }
            meta = {
                "vocab": vocab,
                "stoi": stoi,
                "itos": itos,
                "model_type": cfg.model_type,
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "bidirectional": cfg.bidirectional,
                "seq_len": cfg.seq_len,
                "device": str(device),
                "num_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "batch_size": int(batch_size),
                "epoch": int(epoch + 1),
            }
            yield step, total_steps, float(loss.detach().cpu().item()), model_state, meta

        if scheduler is not None:
            scheduler.step()


@torch.no_grad()
def generate_text(
    model_state: Dict,
    meta: Dict,
    seed: str,
    length: int = 80,
    temperature: float = 1.0,
) -> str:
    vocab: List[str] = meta["vocab"]
    stoi: Dict[str, int] = meta["stoi"]
    itos: Dict[int, str] = meta["itos"]

    device = _device()
    cfg_dict = model_state.get("config", {})
    cell = cfg_dict.get("model_type", meta.get("model_type", "lstm"))
    hidden_size = int(cfg_dict.get("hidden_size", meta.get("hidden_size", 64)))
    num_layers = int(cfg_dict.get("num_layers", meta.get("num_layers", 2)))
    bidirectional = bool(cfg_dict.get("bidirectional", meta.get("bidirectional", False)))

    model = _CharLM(
        vocab_size=len(vocab),
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        cell=cell,
    ).to(device)
    model.load_state_dict(model_state["state_dict"])
    model.eval()

    if not seed:
        seed = random.choice(vocab)

    generated = seed
    state = None

    for ch in seed[:-1]:
        idx = stoi.get(ch, None)
        if idx is None:
            continue
        x = torch.tensor([[idx]], dtype=torch.long, device=device)
        _, state = model(x, state)

    last = seed[-1]
    for _ in range(int(length)):
        idx = stoi.get(last, None)
        if idx is None:
            idx = random.randrange(len(vocab))
        x = torch.tensor([[idx]], dtype=torch.long, device=device)
        logits, state = model(x, state)
        logits = logits[0, -1] / max(float(temperature), 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_idx = int(torch.multinomial(probs, num_samples=1).item())
        next_ch = itos[next_idx]
        generated += next_ch
        last = next_ch

    return generated
