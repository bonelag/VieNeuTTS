"""
Batched acoustic-frame generation for v3 Turbo serving.

``generate_frame_batched`` runs the acoustic decoder's 16-codebook autoregression
for a WHOLE BATCH of B sequences at once (one audio frame each), reusing the
model's ``AcousticDecoder.cached_step``. This is the per-step kernel of the
batched serving runtime: B requests are advanced together instead of one at a
time, which is where the throughput win comes from.

Equivalent (per row) to ``VieNeuV3TurboForTTS.decode_one_frame`` but vectorised
over the batch dimension.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F


@torch.no_grad()
def _sample_batched(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
    repetition_penalty: float = 1.0, prev=None, penalty_mask=None,
) -> torch.Tensor:
    """Top-k + top-p sampling over a batch. logits: (B, V) -> codes: (B,).

    ``prev`` (optional) is a length-B list of per-row seen-token iterables for this
    codebook; when ``repetition_penalty != 1.0`` those codes are down-weighted on
    each row's logits BEFORE temperature — same CTRL/MOSS rule as the single-path
    ``_sample_token`` (logit<0 → *penalty, else /penalty).

    ``penalty_mask`` (optional, ``(B, V)``) là bản vector-hoá GPU của ``prev``:
    1 tại mọi code trong cửa sổ gần đây của từng row — áp cùng luật phạt nhưng
    không phát sinh bất kỳ GPU→CPU sync nào (nhánh ``prev`` phải ``.item()``
    từng row để dựng index).
    """
    if penalty_mask is not None and not math.isclose(repetition_penalty, 1.0):
        factor = torch.where(
            penalty_mask > 0,
            torch.full_like(penalty_mask, repetition_penalty),
            torch.ones_like(penalty_mask),
        ).to(logits.dtype)
        logits = torch.where(logits < 0, logits * factor, logits / factor)
    elif not math.isclose(repetition_penalty, 1.0) and prev is not None:
        for b, seen in enumerate(prev):
            if seen:
                idx = torch.as_tensor(sorted(seen), device=logits.device, dtype=torch.long)
                sel = logits[b, idx]
                logits[b, idx] = torch.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-6)
    if top_k and 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if 0.0 < top_p < 1.0:
        s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(s_logits, dim=-1)
        drop = probs.cumsum(dim=-1) > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        s_logits = s_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, s_idx, s_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class GpuBatchRepHistory:
    """Repetition-penalty history cho CẢ BATCH trên một buffer GPU (không sync).

    Thay cho ``[RepetitionHistory(n_vq, w) for _ in range(B)]``: nhánh cũ phải
    ``int(code[b].item())`` B×n_vq lần mỗi frame (mỗi lần là một GPU→CPU sync).
    Ngữ nghĩa y hệt: cửa sổ trượt ``window`` frame gần nhất, mỗi kênh 1 code/frame.

    Shape-static như ``GpuRepetitionHistory`` bên ``_v3_turbo_engine`` (count là
    tensor 0-d trên device, mask quét cả cửa sổ + validity) nên buffer này nằm
    được bên trong một ``torch.cuda.CUDAGraph`` đã capture.
    """

    __slots__ = ("codes", "count", "window")

    def __init__(self, batch: int, n_channels: int, window: int, device):
        self.window = max(1, int(window))
        self.codes = torch.zeros((batch, n_channels, self.window), dtype=torch.long, device=device)
        self.count = torch.zeros((), dtype=torch.long, device=device)

    def penalty_mask(self, ch: int, vocab: int, device, dtype=torch.float32):
        """Mask ``(B, vocab)`` = 1 tại code trong cửa sổ của kênh ``ch`` từng row."""
        W = self.window
        recent = self.codes[:, ch, :]                                    # (B, W)
        valid = torch.arange(W, device=device) < self.count              # (W,)
        hit = (recent.unsqueeze(2) == torch.arange(vocab, device=device).view(1, 1, -1)) & valid.view(1, -1, 1)
        return hit.any(dim=1).to(dtype)

    def add(self, ch: int, codes: torch.Tensor) -> None:
        """codes ``(B,)`` của kênh ``ch`` — ghi in-place, không sync.

        Dùng ``scatter_`` thay vì setitem/index_put_: chỉ scatter_ là được phép
        bên trong CUDA graph capture (index động qua tensor 0-d count).
        """
        pos = (self.count % self.window).view(1, 1).expand(self.codes.shape[0], 1)
        self.codes[:, ch].scatter_(1, pos, codes.unsqueeze(1))

    def advance(self) -> None:
        self.count += 1

    def reset(self) -> None:
        """Xóa lịch sử (đầu mỗi lượt sinh) — in-place, an toàn với CUDA graph."""
        self.codes.zero_()
        self.count.zero_()


@torch.no_grad()
def generate_frame_batched(
    model,
    backbone_hidden: torch.Tensor,            # (B, H) — backbone hidden for each sequence
    *,
    temperature=0.8,
    top_k: int = 25,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    history=None,   # optional: GpuBatchRepHistory (không sync) hoặc list (len B) per-row RepetitionHistory (CPU, có sync)
):
    """Sample one audio frame for each of B sequences.

    ``temperature`` is a scalar applied to every codebook. When
    ``repetition_penalty != 1.0`` and ``history`` is supplied, each row's codes are
    penalised against that row's per-codebook history (then the new codes are added
    to it) — matching the single-path ``decode_one_frame`` behaviour.

    Returns ``(codes, prefill_out)`` where ``codes`` is ``(B, n_vq)`` Long and
    ``prefill_out`` is ``(B, 2, H)`` (slot-0 column feeds the EOS / text head).
    """
    cfg = model.config
    n_vq, H = cfg.n_vq, cfg.hidden_size
    temps = [temperature] * n_vq
    dec = model.acoustic_decoder
    L = len(dec.layers)
    dt = next(dec.parameters()).dtype
    dev = backbone_hidden.device
    B = backbone_hidden.shape[0]
    sgs = cfg.speech_generation_start_token_id
    use_rep = not math.isclose(repetition_penalty, 1.0) and history is not None
    gpu_hist = history if (use_rep and hasattr(history, "penalty_mask")) else None
    vocab = cfg.audio_vocab_size

    def _sample_ch(ch: int, vec: torch.Tensor) -> torch.Tensor:
        logits = model.audio_lm_heads[ch](vec).float()                            # (B, V)
        if gpu_hist is not None:
            pmask = gpu_hist.penalty_mask(ch, vocab, dev)
            code = _sample_batched(logits, temps[ch], top_k, top_p, repetition_penalty, penalty_mask=pmask)
            gpu_hist.add(ch, code)
        else:
            prev = [history[b][ch] for b in range(B)] if use_rep else None
            code = _sample_batched(logits, temps[ch], top_k, top_p, repetition_penalty, prev)
            if use_rep:
                for b in range(B):
                    history[b][ch].add(int(code[b].item()))
        return code

    cond = backbone_hidden.to(dt)                                                  # (B, H)
    sgs_ids = torch.full((B,), sgs, device=dev, dtype=torch.long)
    txt = model.text_embeddings(sgs_ids).to(dt)                                    # (B, H)
    tok = torch.stack([cond, txt], dim=1)                                          # (B, 2, H)
    # Use arange (a GPU kernel) for positions so this stays CUDA-graph capturable
    # (``torch.tensor([...])`` would do a host->device copy, which capture forbids).
    pos = torch.arange(2, device=dev, dtype=torch.long)
    hidden, pk, pv = dec.cached_step(tok, pos, [None] * L, [None] * L)
    prefill_out = hidden                                                           # (B, 2, H)

    codes: List[torch.Tensor] = [_sample_ch(0, hidden[:, 1])]
    for ch in range(1, n_vq):
        emb = model.audio_embeddings[ch - 1](codes[-1]).to(dt)                     # (B, H)
        pos = torch.arange(ch + 1, ch + 2, device=dev, dtype=torch.long)
        hidden, pk, pv = dec.cached_step(emb.view(B, 1, H), pos, pk, pv)
        codes.append(_sample_ch(ch, hidden[:, 0]))
    if gpu_hist is not None:
        gpu_hist.advance()
    return torch.stack(codes, dim=1), prefill_out                                  # (B, n_vq), (B, 2, H)
