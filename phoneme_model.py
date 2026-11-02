"""
phoneme_model.py
Loads the phoneme-level speech recognizer (wav2vec2 fine-tuned on espeak
phonemes). No language-model correction layer -- it reports the raw
phones it heard, which is what makes mispronunciation detection possible.
First run downloads the model weights from Hugging Face (~1.2GB), then
caches locally.
"""
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
print(f"Loading {MODEL_NAME} (first run downloads ~1.2GB, please wait)...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.")

VOCAB = processor.tokenizer.get_vocab()
ID_TO_PHONE = {v: k for k, v in VOCAB.items()}


def get_logits(audio_path: str) -> torch.Tensor:
    """Returns raw model logits, shape [num_frames, vocab_size]."""
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits[0]
    return logits


def get_phones(audio_path: str) -> str:
    """Returns the decoded phone string, e.g. 'v ɛ r i'."""
    logits = get_logits(audio_path)
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids.unsqueeze(0))[0]