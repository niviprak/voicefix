"""gop.py -- Goodness of Pronunciation (GOP) scoring for VoiceFix.

Method
------
1. CTC forced alignment (Viterbi over the blank/phone lattice) locates
   WHERE each expected phone was most likely spoken, instead of assuming
   every phone lasts exactly the same time.
2. Speech-frame filter: only frames where the recogniser heard a real
   phone count; silent segments are widened once into neighbouring
   frames before being declared speechless.
3. GOP (Witt & Young 2000 adapted to CTC): mean log-prob of the TARGET
   phone on those frames minus the mean log-prob of the best COMPETING
   phone. 0 = target clearly won, very negative = something else won.
4. Confidence: sigmoid map of raw GOP onto 0-100, calibrated so the flag
   threshold (-4.0) lands exactly on 50.
"""
import math

import torch

FLAG_THRESHOLD = -4.0      # below this raw GOP a sound is flagged
_WIDEN_FRACTION = 0.35     # how far to look for speech in silent segments
_NEG = float("-inf")


def gop_to_confidence(gop: float) -> int:
    """Map raw GOP (<= 0) onto [0, 100]: 0 -> ~94, -4 -> 50, -8 -> ~6."""
    gop = min(float(gop), 0.0)
    return round(100.0 / (1.0 + math.exp(-(gop + 4.0) / 1.5)))


def _blank_id(vocab: dict) -> int:
    for tok in ("|", "<pad>", "[PAD]", "<blank>", "<blk>", ""):
        if tok in vocab:
            return vocab[tok]
    return 0


def _any_phone_logprob(logp: torch.Tensor, blank_id: int) -> torch.Tensor:
    """Per-frame log-prob that SOME real phone (not blank) was spoken."""
    mask = torch.ones(logp.shape[-1], dtype=torch.bool, device=logp.device)
    mask[blank_id] = False
    return torch.logsumexp(logp[:, mask], dim=-1)


def ctc_align(logp: torch.Tensor, phone_ids, blank_id: int):
    """Viterbi forced alignment. phone_ids may contain None (unscoreable
    phone); those align as 'any real phone' so neighbours keep sensible
    boundaries. Returns ordered, non-overlapping (start, end) frame pairs."""
    T, _V = logp.shape
    n = len(phone_ids)
    if n == 0 or T == 0:
        return []

    any_lp = _any_phone_logprob(logp, blank_id)
    n_states = 2 * n + 1                      # B p1 B p2 B ... pn B
    emit = torch.empty(T, n_states, device=logp.device, dtype=logp.dtype)
    for j in range(n_states):
        if j % 2 == 0:
            emit[:, j] = logp[:, blank_id]
        else:
            pid = phone_ids[(j - 1) // 2]
            emit[:, j] = logp[:, pid] if pid is not None else any_lp
    em = emit.tolist()

    # skipping the blank between two labels is legal only if they differ
    skip_ok = [False] * n_states
    for j in range(2, n_states):
        if j % 2 == 1:
            a, b = phone_ids[(j - 3) // 2], phone_ids[(j - 1) // 2]
            skip_ok[j] = (a is None) or (b is None) or (a != b)

    dp_prev = [_NEG] * n_states
    dp_prev[0] = em[0][0]
    if n_states > 1:
        dp_prev[1] = em[0][1]
    bp_rows = []
    for t in range(1, T):
        row = [_NEG] * n_states
        bpr = [0] * n_states
        et = em[t]
        for j in range(n_states):
            best, src = dp_prev[j], j
            if j >= 1 and dp_prev[j - 1] > best:
                best, src = dp_prev[j - 1], j - 1
            if j >= 2 and skip_ok[j] and dp_prev[j - 2] > best:
                best, src = dp_prev[j - 2], j - 2
            row[j] = best + et[j]
            bpr[j] = src
        bp_rows.append(bpr)
        dp_prev = row

    j = n_states - 1
    if n_states > 1 and dp_prev[n_states - 2] > dp_prev[n_states - 1]:
        j = n_states - 2
    path = [0] * T
    for t in range(T - 1, 0, -1):
        path[t] = j
        j = bp_rows[t - 1][j]
    path[0] = j

    segs = [None] * n
    for k in range(n):
        st = 2 * k + 1
        frames = [t for t in range(T) if path[t] == st]
        if frames:
            segs[k] = (frames[0], frames[-1] + 1)
    for k in range(n):                        # safety net (rarely triggers)
        if segs[k] is None:
            lo = segs[k - 1][1] if k > 0 and segs[k - 1] else 0
            hi = segs[k + 1][0] if k + 1 < n and segs[k + 1] else T
            hi = max(hi, lo + 1)
            segs[k] = (lo, hi)
    for k in range(1, n):                     # enforce order / no overlap
        s_prev, e_prev = segs[k - 1]
        s_cur, e_cur = segs[k]
        if s_cur < e_prev:
            cut = max(s_prev + 1, min(e_cur - 1, (s_cur + e_prev) // 2))
            segs[k - 1] = (s_prev, cut)
            segs[k] = (cut, e_cur)
    return [(max(0, s), min(T, max(s + 1, e))) for s, e in segs]


def _speech_window(pred_ids, blank_id, s, e, T):
    """Frames in [s, e) with a real phone; widen once if pure silence."""
    keep = [t for t in range(s, e) if pred_ids[t] != blank_id]
    if keep:
        return s, e, keep, [pred_ids[t] for t in keep]
    pad = max(2, int(_WIDEN_FRACTION * (e - s)))
    s2, e2 = max(0, s - pad), min(T, e + pad)
    keep = [t for t in range(s2, e2) if pred_ids[t] != blank_id]
    if keep:
        return s2, e2, keep, [pred_ids[t] for t in keep]
    return s, e, [], []


def compute_gop(logits, expected_phones, vocab, id_to_phone,
                audio_duration=0.0):
    """Score every expected phoneme against the recording."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    T, V = logp.shape
    n = len(expected_phones)
    if n == 0 or T == 0:
        return []

    blank = _blank_id(vocab)
    pred_ids = torch.argmax(logp, dim=-1).tolist()
    phone_ids = [vocab.get(p) for p in expected_phones]

    segments = ctc_align(logp, phone_ids, blank)
    if len(segments) != n:                    # paranoid fallback
        segments = [(math.floor(i * T / n),
                     max(math.floor(i * T / n) + 1,
                         math.floor((i + 1) * T / n))) for i in range(n)]

    col_cache = {}

    def competitor_columns(pid):
        if pid not in col_cache:
            mask = torch.ones(V, dtype=torch.bool, device=logp.device)
            mask[blank] = False
            mask[pid] = False
            col_cache[pid] = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        return col_cache[pid]

    results = []
    for i, phone in enumerate(expected_phones):
        s, e = segments[i]
        pid = phone_ids[i]
        s, e, keep, kept_preds = _speech_window(pred_ids, blank, s, e, T)
        base = {
            "expected": phone,
            "start_frac": s / T,
            "end_frac": e / T,
            "start_frame": s,
            "end_frame": e,
            "n_speech_frames": len(keep),
        }
        if pid is None:                       # phone the model cannot score
            results.append({**base, "model_actually_heard": "?",
                            "competitor": "?", "gop_score": -8.0,
                            "confidence": 0, "flagged": True})
            continue
        if not keep:                          # genuinely no speech here
            results.append({**base, "model_actually_heard": "?",
                            "competitor": "?", "gop_score": -6.0,
                            "confidence": 12, "flagged": True})
            continue

        idx = torch.tensor(keep, dtype=torch.long, device=logp.device)
        seg_kept = logp[idx]
        phone_lp = seg_kept[:, pid].mean().item()
        cols = competitor_columns(pid)
        other = seg_kept[:, cols]
        other_lp = other.max(dim=-1).values.mean().item()
        counts = torch.bincount(torch.argmax(other, dim=-1))
        competitor = id_to_phone.get(cols[int(torch.argmax(counts))].item(),
                                     "?")
        gop = min(phone_lp - other_lp, 0.0)

        heard_list = [id_to_phone.get(p, "?") for p in kept_preds]
        heard = max(set(heard_list), key=heard_list.count)

        results.append({**base,
                        "model_actually_heard": heard,
                        "competitor": competitor,
                        "gop_score": round(gop, 3),
                        "confidence": gop_to_confidence(gop),
                        "flagged": gop < FLAG_THRESHOLD})
    return results