"""
CUDA-graph capture cho TOÀN BỘ một frame sinh audio (acoustic + backbone).

Bước một frame gồm: acoustic decoder (2 token prefill + 15 cached step, 16
codebook, kèm repetition penalty) -> dựng slot embeds -> backbone decode step
(Qwen3, 1 token). Mỗi op đều có shape tĩnh nên cả chuỗi capture được vào MỘT
``torch.cuda.CUDAGraph``: replay 1 lần/frame thay vì ~300 kernel launch nhỏ —
trên Windows/WDDM (launch overhead ~20-40µs/kernel) đây là chỗ thắng lớn nhất.

Thành phần shape-tĩnh:
- KV cache: ``transformers.StaticCache`` (buffer định sẵn, ghi in-place tại
  ``cache_position``) thay cho DynamicCache (concat mỗi step → không capture
  được).
- Attention mask: buffer ``(1, max_cache_len)`` 0/1 — vị trí hợp lệ được BẬT
  bằng ``scatter_`` (index động qua tensor ``cache_pos`` — ``setitem``/
  ``index_put_`` không được phép trong capture, scatter_ thì được).
- Repetition history: ``GpuBatchRepHistory`` (codes trên GPU, count là tensor
  0-d, ghi bằng scatter_) — penalty chạy NGAY TRONG graph.
- ``cache_pos`` tăng in-graph bằng ``add_(1)``.

Prefill vẫn chạy eager (độ dài prompt thay đổi mỗi lần) nhưng ghi vào CÙNG
StaticCache; các slot cũ từ lượt sinh trước bị che bởi attention mask nên
không cần zero lại cache.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .batched_acoustic import generate_frame_batched, GpuBatchRepHistory


class GraphedFrameStep:
    """Một CUDA graph advance đúng MỘT frame cho MỘT sequence (B=1).

    Usage (xem ``inference_v3_turbo._generate_codes``)::

        g = GraphedFrameStep(model, sampling...)
        g.begin(prefill_embeds, speaker_emb)      # eager prefill vào StaticCache
        for _ in range(max_new_frames):
            g.step()                              # 1 graph.replay()
            codes = g.codes()                     # (n_vq,) clone
            if g.is_eos():
                break
    """

    def __init__(self, model, *, max_cache_len: int, temperature: float, top_k: int,
                 top_p: float, repetition_penalty: float = 1.0,
                 repetition_window: int = 64, warmup: int = 3):
        self.model = model
        cfg = model.config
        self.cfg = cfg
        backbone = model.semantic_backbone
        dev = next(model.parameters()).device
        dt = next(model.acoustic_decoder.parameters()).dtype
        self.device = dev
        self.max_cache_len = int(max_cache_len)
        self._sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p,
                              repetition_penalty=repetition_penalty)
        self.eos_id = int(cfg.speech_generation_end_token_id)
        self.history = (GpuBatchRepHistory(1, cfg.n_vq, repetition_window, dev)
                        if not math.isclose(repetition_penalty, 1.0) else None)

        from transformers import StaticCache
        self.cache = StaticCache(config=backbone.config, max_cache_len=self.max_cache_len)

        H = cfg.hidden_size
        n_vq = cfg.n_vq
        # Static buffers — giá trị được cập nhật in-place, graph đọc lúc replay.
        self.static_h = torch.zeros(1, H, device=dev, dtype=dt)          # backbone hidden vào/ra
        self.slot_row = torch.zeros(1, 1, n_vq + 1, dtype=torch.long, device=dev)
        self.slot_row[:, :, 0] = int(cfg.speech_generation_start_token_id)
        self.static_spk = (torch.zeros(1, cfg.speaker_embedding_dim, device=dev, dtype=torch.float32)
                           if getattr(cfg, "use_speaker_embedding", False) and getattr(cfg, "speaker_embedding_dim", None) else None)
        self.static_mask = torch.zeros(1, self.max_cache_len, dtype=torch.long, device=dev)
        self.static_cpos = torch.zeros(1, dtype=torch.long, device=dev)

        def _run_once():
            # 1) acoustic: 16 codebook + sampling (+ rep penalty trong graph)
            codes, prefill_out = generate_frame_batched(
                model, self.static_h, history=self.history, **self._sampling)
            tl = model.text_lm_head(prefill_out[:, 0]).float()
            sgs_id = int(cfg.speech_generation_start_token_id)
            is_eos = (tl.argmax(-1) == self.eos_id) | ((tl[..., self.eos_id] - tl[..., sgs_id]) > -1.0)
            # 2) slot row của frame (index tĩnh → copy kernel, capture được)
            self.slot_row[:, 0, 1:] = codes
            embeds = model._build_inputs_embeds(self.slot_row, speaker_emb=self.static_spk)
            # 3) mask hợp lệ tại vị trí đang decode (index động → scatter_)
            self.static_mask.scatter_(1, self.static_cpos.view(1, 1), 1)
            out = backbone(inputs_embeds=embeds, attention_mask=self.static_mask,
                           past_key_values=self.cache, use_cache=True,
                           cache_position=self.static_cpos, return_dict=True)
            self.static_cpos += 1          # in-graph: frame kế tiếp
            self.static_h.copy_(out.last_hidden_state[:, 0])
            return codes, is_eos

        # Warmup trên stream phụ (cần cho capture). Warmup chạy THẬT nên làm bẩn
        # cache slot đầu — prefill của lượt sinh thật (chạy sau) ghi đè lại, các
        # slot khác bị che bởi attention mask.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup):
                _run_once()
        torch.cuda.current_stream().wait_stream(s)
        if self.history is not None:
            self.history.reset()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            codes, is_eos = _run_once()
            self._codes = codes            # (1, n_vq) static
            self._is_eos = is_eos          # (1,) bool static

    # ── Một lượt sinh ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def begin(self, prefill_embeds: torch.Tensor, speaker_emb: Optional[torch.Tensor]) -> None:
        """Prefill eager vào StaticCache + reset trạng thái cho lượt sinh mới.

        ``prefill_embeds``: ``(1, T, H)`` — output của ``_build_inputs_embeds``.
        """
        backbone = self.model.semantic_backbone
        T = prefill_embeds.shape[1]
        self.static_mask.zero_()
        self.static_mask[0, :T] = 1
        self.static_cpos.fill_(T)
        if self.static_spk is not None and speaker_emb is not None:
            self.static_spk.copy_(speaker_emb.to(self.static_spk.dtype))
        if self.history is not None:
            self.history.reset()
        out = backbone(inputs_embeds=prefill_embeds.to(self.static_h.dtype),
                       attention_mask=torch.ones(1, T, dtype=torch.long, device=self.device),
                       past_key_values=self.cache, use_cache=True,
                       cache_position=torch.arange(T, device=self.device),
                       return_dict=True)
        self.static_h.copy_(out.last_hidden_state[:, -1])

    @torch.no_grad()
    def step(self) -> None:
        """Advance đúng một frame — MỘT graph replay."""
        self.graph.replay()

    def codes(self) -> torch.Tensor:
        """Codes của frame vừa step: ``(n_vq,)`` (clone, an toàn giữ lại)."""
        return self._codes[0].clone()

    def is_eos(self) -> bool:
        """Stop token đã bắn ở frame vừa step chưa (1 sync GPU→CPU)."""
        return bool(self._is_eos.reshape(-1)[0])
