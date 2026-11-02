"""
test_synthetic.py

Proves the wiring and math are correct WITHOUT the real wav2vec2 model
(which can't be downloaded in this sandbox -- no access to
huggingface.co here). This fabricates logits the way the real model
would produce them for a correct vs. incorrect pronunciation, and runs
them through the actual gop.py and app.py logic.

Run this yourself once you have the real model too -- it's a good
regression check to keep around.
"""

import torch
from phonemizer import phonemize
from phonemizer.separator import Separator
from gop import compute_gop
from feedback import generate_feedback

# ---- stand-in vocab (a tiny slice of what wav2vec2-lv-60-espeak-cv-ft has) ----
FAKE_VOCAB = {"v": 0, "w": 1, "ɛ": 2, "r": 3, "i": 4, "θ": 5, "t": 6, "ɪ": 7, "ŋ": 8, "k": 9}
FAKE_ID_TO_PHONE = {v: k for k, v in FAKE_VOCAB.items()}
VOCAB_SIZE = len(FAKE_VOCAB)

print("=" * 60)
print("STEP 1 -- phonemizer wiring (real, not mocked)")
print("=" * 60)
sep = Separator(phone=" ", word="", syllable="")
for word in ["very", "think", "wonder"]:
    phones = phonemize(word, language="en-us", backend="espeak",
                        separator=sep, strip=True, preserve_punctuation=False,
                        with_stress=False).split()
    print(f"  '{word}' -> {phones}")

print()
print("=" * 60)
print("STEP 2 -- GOP math on FABRICATED logits (model download not possible here)")
print("=" * 60)

def fake_logits_for_correct_pronunciation(expected_phones, num_frames=40):
    """Simulates a model that's confident in exactly the expected phones."""
    logits = torch.randn(num_frames, VOCAB_SIZE) * 0.5
    frames_per_phone = num_frames // len(expected_phones)
    for i, phone in enumerate(expected_phones):
        if phone in FAKE_VOCAB:
            start, end = i * frames_per_phone, (i + 1) * frames_per_phone
            logits[start:end, FAKE_VOCAB[phone]] += 8.0  # strong confidence spike
    return logits

def fake_logits_for_v_to_w_error(expected_phones, num_frames=40):
    """Simulates a model that heard 'w' where 'v' was expected (the classic case)."""
    logits = torch.randn(num_frames, VOCAB_SIZE) * 0.5
    frames_per_phone = num_frames // len(expected_phones)
    for i, phone in enumerate(expected_phones):
        if phone in FAKE_VOCAB:
            start, end = i * frames_per_phone, (i + 1) * frames_per_phone
            actual_phone = "w" if phone == "v" else phone
            logits[start:end, FAKE_VOCAB[actual_phone]] += 8.0
    return logits

expected = ["v", "ɛ", "r", "i"]  # "very"

print("\n-- Case A: correctly pronounced 'very' --")
logits_correct = fake_logits_for_correct_pronunciation(expected)
results_correct = compute_gop(logits_correct, expected, FAKE_VOCAB, FAKE_ID_TO_PHONE)
for r in results_correct:
    print(f"  {r['expected']} -> GOP {r['gop_score']:+.2f} (heard: {r['model_actually_heard']}) flagged={r['flagged']}")

print("\n-- Case B: 'wery' instead of 'very' (v->w substitution) --")
logits_error = fake_logits_for_v_to_w_error(expected)
results_error = compute_gop(logits_error, expected, FAKE_VOCAB, FAKE_ID_TO_PHONE)
for r in results_error:
    print(f"  {r['expected']} -> GOP {r['gop_score']:+.2f} (heard: {r['model_actually_heard']}) flagged={r['flagged']}")

print()
print("=" * 60)
print("STEP 3 -- feedback.py consuming these results (Ollama not running here, fallback path)")
print("=" * 60)
print("\n-- Feedback for Case A (correct) --")
print(generate_feedback(results_correct, "very"))
print("\n-- Feedback for Case B (v->w error) --")
print(generate_feedback(results_error, "very"))

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
correct_flagged = any(r["flagged"] for r in results_correct)
error_flagged = any(r["flagged"] and r["expected"] == "v" for r in results_error)
print(f"Correct pronunciation wrongly flagged? {'YES -- BUG' if correct_flagged else 'No (good)'}")
print(f"v->w error correctly flagged?          {'Yes (good)' if error_flagged else 'NO -- BUG'}")
