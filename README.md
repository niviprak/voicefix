# VoiceFix

VoiceFix is a privacy-first, phoneme-level pronunciation coach for second-language
speakers. You say one phrase, and it scores every individual sound in that phrase
against the target, shows you what the speech model actually heard, and gives you one
beginner-friendly coaching tip. Everything runs on the host machine: no audio is ever
stored or uploaded, and no cloud speech API hears your voice.

It was built as an MSc Interactive Media dissertation at University College Cork
(2025-2026), and it grew out of a simple, frustrating experience: being fluent on
paper but constantly asked to repeat yourself at shop counters and bus stops, with no
way to learn which sound actually failed.

## The process: how a recording becomes a score

1. **Phonemise the target.** The typed or chosen phrase is converted into its list of
   expected sounds (IPA) with eSpeak-ng through the Phonemizer toolkit (US English,
   no stress markers).
2. **Normalise the symbols.** eSpeak's symbol inventory and the model's vocabulary do
   not match exactly, so a normalisation layer maps between them: a curated variant
   table first, then greedy longest-chunk matching, then per-character substitution as
   a last resort.
3. **Extract raw acoustic evidence.** The recording is resampled to 16 kHz and pushed
   through a wav2vec2 CTC model fine-tuned on eSpeak phonemes across sixty languages.
   The raw frame-level logits are kept, with deliberately no language-model decoding,
   so a mispronounced sound is reported instead of being auto-corrected away.
4. **Align in time.** A Viterbi search over a 2n+1 blank/phone lattice locates when
   each expected sound was most likely spoken. Skipping the blank between two labels
   is only legal when the labels differ, which stops repeated sounds collapsing into
   one segment.
5. **Filter and score.** Silent frames are discarded (a segment that looks fully
   silent is widened once by 35 percent of its length before being declared
   speechless). For each sound, Goodness of Pronunciation is computed: the mean
   log-probability of the target on its speech frames minus the mean log-probability
   of the strongest competing sound, clamped at zero.
6. **Calibrate and explain.** A sigmoid maps raw GOP onto a 0-100 confidence, with the
   raw value -4.0 landing exactly on 50, the flag threshold. The result renders as a
   two-row chip timeline (target versus what we heard), and the three weakest sounds
   go to a locally hosted Qwen2.5 model through Ollama, which writes one coaching tip
   of exactly three sentences. If Ollama is down, a deterministic plain-language
   fallback guide takes over, so the feedback loop never breaks.

## Tools

- **Python 3.10+** - the whole stack
- **PyTorch and Hugging Face Transformers** - tensor math and the wav2vec2 CTC
  checkpoint (facebook/wav2vec2-lv-60-espeak-cv-ft)
- **Gradio** - browser interface, microphone capture and audio players
- **Phonemizer with eSpeak-ng** - text-to-IPA conversion and the offline TTS fallback
- **Ollama with Qwen2.5 (3B)** - local coaching-tip generation
- **edge-tts** - neural reference audio, with eSpeak-ng as the offline fallback
- **librosa and soundfile** - resampling and WAV encoding
- **ngrok** - HTTPS tunnel used for remote and mobile testing during the study
- **test_synthetic.py** - a weight-free regression test that verifies the alignment
  and GOP math on fabricated logits, so the scoring engine can be checked without
  downloading the model
