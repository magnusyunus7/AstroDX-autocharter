"""The chart model (Phase C3): listens to the whole song, writes the chart.

  encoder   the song as quarter-beat steps (mel at 8 points per step + MERT),
            a transformer over the whole song (up to 1,024 beats), so every
            step knows about the rest of the song (sections, repeats, drops).
  decoder   writes the C1 tokens one at a time. Each token gets the encoder's
            output for its own step added to its input (where in the music it
            is), and cross-attends to the whole song. Settings (level, slot,
            era, source, BPM) are added to every token.
"""
from __future__ import annotations

import torch
from torch import nn

from backend.chartcodec.tokens import VOCAB
from ml.data import (BPM_BUCKETS, MAX_STEPS, MAX_TOKENS, MEL_DIM, MERT_DIM, PAD, SLOTS,
                     STEPS_PER_BEAT)

SIZES = {   # d, heads, encoder layers, decoder layers  (small ~16M, mid ~39M, large ~78M parameters)
    "small": (320, 5, 4, 5),
    "mid": (448, 7, 5, 7),
    "large": (576, 8, 6, 9),
}


class ChartModel(nn.Module):
    def __init__(self, size: str = "mid", features: str = "both", dropout: float = 0.1):
        super().__init__()
        d, h, le, ld = SIZES[size]
        self.config = {"size": size, "features": features, "dropout": dropout}
        self.features = features
        self.mel_in = nn.Sequential(nn.Linear(MEL_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.mert_in = nn.Sequential(nn.Linear(MERT_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.enc_norm_in = nn.LayerNorm(d)
        self.enc_pos = nn.Embedding(MAX_STEPS, d)
        self.bar = nn.Embedding(4 * STEPS_PER_BEAT, d)
        enc_layer = nn.TransformerEncoderLayer(d, h, 4 * d, dropout, batch_first=True, norm_first=True,
                                               activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, le, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)

        self.tok = nn.Embedding(len(VOCAB), d, padding_idx=PAD)
        self.dec_pos = nn.Embedding(MAX_TOKENS, d)
        self.here = nn.Linear(d, d)                       # the encoder's output at the token's own step
        self.level = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.slot = nn.Embedding(SLOTS, d)
        self.era = nn.Embedding(2, d)
        self.source = nn.Embedding(2, d)
        self.bpm = nn.Embedding(BPM_BUCKETS, d)
        dec_layer = nn.TransformerDecoderLayer(d, h, 4 * d, dropout, batch_first=True, norm_first=True,
                                               activation="gelu")
        self.decoder = nn.TransformerDecoder(dec_layer, ld)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(VOCAB), bias=False)
        self.head.weight = self.tok.weight                # tied
        for m in self.modules():                          # small start values (default N(0,1) is far too big)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()

    def encode(self, enc, enc_mask, bar):
        x = 0
        if self.features in ("mel", "both"):
            x = x + self.mel_in(enc[..., :MEL_DIM].float())
        if self.features in ("mert", "both"):
            x = x + self.mert_in(enc[..., MEL_DIM:].float())
        S = enc.shape[1]
        x = self.enc_norm_in(x) + self.enc_pos(torch.arange(S, device=enc.device)) + self.bar(bar)
        return self.enc_norm(self.encoder(x, src_key_padding_mask=enc_mask))

    def decode(self, memory, enc_mask, tok_in, tok_step, level, slot, era, source, bpm):
        B, T = tok_in.shape
        here = torch.gather(memory, 1, tok_step.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
        cond = (self.level(((level.float() - 10.0) / 5.0).unsqueeze(-1)) + self.slot(slot) + self.era(era)
                + self.source(source) + self.bpm(bpm)).unsqueeze(1)
        x = self.tok(tok_in) + self.dec_pos(torch.arange(T, device=tok_in.device)) + self.here(here) + cond
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tok_in.device), diagonal=1)  # True = can't see
        y = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tok_in == PAD,
                         memory_key_padding_mask=enc_mask)
        return self.head(self.out_norm(y))

    def forward(self, b):
        memory = self.encode(b["enc"], b["enc_mask"], b["bar"])
        return self.decode(memory, b["enc_mask"], b["tok_in"], b["tok_step"], b["level"], b["slot"],
                           b["era"], b["source"], b["bpm"])

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
