"""tts.py -- reference audio via Edge TTS, espeak fallback."""
import io
import os
import shutil
import subprocess
import tempfile


def synthesize_wav(text, voice="en-US-JennyNeural"):
    try:
        return _edge_tts(text, voice)
    except Exception:
        return _espeak_tts(text)


def _edge_tts(text, voice):
    import asyncio
    import edge_tts
    mp3_path = tempfile.mktemp(suffix=".mp3")
    asyncio.run(edge_tts.Communicate(text, voice).save(mp3_path))
    import librosa
    import soundfile as sf
    speech, _ = librosa.load(mp3_path, sr=22050)
    buf = io.BytesIO()
    sf.write(buf, speech, 22050, format="WAV")
    if os.path.exists(mp3_path):
        os.remove(mp3_path)
    return buf.getvalue()


def _espeak_tts(text):
    if shutil.which("espeak-ng") is None:
        raise RuntimeError("No TTS backend. Run: pip install edge-tts")
    tmp_path = tempfile.mktemp(suffix=".wav")
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150",
                    "-w", tmp_path, text],
                   check=True, capture_output=True, timeout=10)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data
