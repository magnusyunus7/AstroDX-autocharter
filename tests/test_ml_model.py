"""Phase C3: training examples and the model's shapes."""
import numpy as np

from backend.chartcodec.parse import parse_chart
from backend.chartcodec.tokens import encode, ids
from ml.data import beats_to_seconds, encoder_features, token_beats, MEL_DIM


def test_beats_to_seconds_follows_tempo_changes():
    tempo = [(0.0, 120.0), (4.0, 240.0)]
    s = beats_to_seconds(tempo, np.array([0.0, 2.0, 4.0, 6.0]))
    assert np.allclose(s, [0.0, 1.0, 2.0, 2.5])


def test_token_beats():
    c, _ = parse_chart("(120){4}1,,{8}2,3,{20}4,5,E")
    tok = ids(encode(c.notes))
    b = token_beats(tok)
    from backend.chartcodec.tokens import VOCAB
    at = {VOCAB[t]: round(x, 3) for t, x in zip(tok, b) if VOCAB[t] in ("B1", "B2", "B3", "B4", "B5")}
    assert at == {"B1": 0.0, "B2": 2.0, "B3": 2.5, "B4": 3.0, "B5": 3.2}


def test_encoder_features_line_up_with_beats():
    """A loud frame exactly at beat 2 shows up in step 8 (beat 2 = step 8), first sub-sample."""
    fps = 22050 / 220
    mel = np.full((2000, 80), -80, np.float16)
    first = 1.0
    t = first + 2 * 0.5                                   # beat 2 at 120 BPM
    mel[int(round(t * fps))] = 0
    f = encoder_features(mel, None, first, 0.0, [(0.0, 120.0)], 16, "mel")
    loud = np.argwhere(f[:, :MEL_DIM].reshape(16, 8, 80).mean(-1) > 0)
    assert [tuple(x) for x in loud] == [(8, 0)]


def test_model_shapes():
    import torch
    from ml.data import collate, to_torch
    from ml.model import ChartModel
    rng = np.random.default_rng(0)
    ex = {"enc": rng.standard_normal((40, 896)).astype(np.float16), "bar": np.arange(40) % 16,
          "tok_in": np.array([1, 3, 5, 10]), "tok_out": np.array([3, 5, 10, 2]), "tok_step": np.array([0, 0, 4, 4]),
          "level": np.float32(13.7), "slot": 5, "era": 1, "source": 1, "bpm": 7}
    b = to_torch(collate([ex, dict(ex, tok_in=ex["tok_in"][:2], tok_out=ex["tok_out"][:2], tok_step=ex["tok_step"][:2])]))
    m = ChartModel("small")
    out = m(b)
    assert out.shape[:2] == (2, 4) and torch.isfinite(out).all()


def test_hide_bars_hides_whole_bars_but_keeps_beats():
    import random
    from ml.data import BEAT, PAD, hide_bars
    c, _ = parse_chart("(120){8}" + "1,2,3,4,5,6,7,8," * 8 + "E")
    tok = np.array(ids(encode(c.notes)))
    b = token_beats(tok.tolist())
    h = hide_bars(tok, b, 0.5, random.Random(1))
    bars = np.floor(b / 4).astype(int)
    per_bar = [int(((h == PAD) & (bars == k)).sum()) for k in range(8)]
    assert all(n in (0, max(per_bar)) for n in per_bar) and 0 < per_bar.count(0) < 8
    assert int((h == BEAT).sum()) == int((tok == BEAT).sum())
