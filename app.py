import base64
import os
import tempfile
import urllib.parse

import librosa
import gradio as gr
from phonemizer import phonemize
from phonemizer.separator import Separator

from feedback import generate_feedback
from gop import compute_gop
from phoneme_guide import CHOICE_TO_WORD, GUIDE_CHOICES, GUIDE_HTML
from phoneme_model import ID_TO_PHONE, VOCAB, get_logits
from phoneme_timeline import render_timeline
from tts import synthesize_wav

SEP = Separator(phone=" ", word="", syllable="")
IDLE_TIMELINE = render_timeline([])
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])

# ---------------- landing hero config ----------------
HERO_DARK = True
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Your hosted video URL:
HERO_VIDEO_URL = "https://sparkly-sherbet-ea96d3.netlify.app/hero.mp4"

_VIDEO_TAG = ('<video class="vf-hero-video" autoplay muted loop playsinline '
              'preload="auto" src="' + HERO_VIDEO_URL + '"></video>')

MEDIA_HTML = (_VIDEO_TAG +
              '<div class="vf-tint"></div>' +
              '<div class="vf-fade"></div>')

COPY_HTML = (
    "<span class='vf-badge'><span class='dot'></span>PRONUNCIATION COACH</span>"
    "<div class='vf-title'>Speak clearly.<br>"
    "Own <span class='accent'>every sound</span>.</div>"
    "<div class='vf-sub'>VoiceFix listens to your recording, scores every "
    "phoneme, and tells you exactly what to practice next.</div>"
)

FOOT_HTML = (
    "<div class='vf-foot'>No installs needed. Your browser is the only tool. "
    "Recordings are processed on the host machine.</div>"
)

SCENARIO_PACKS = {
    "Interview": [
        "I'm a strong fit for this role",
        "Let me walk you through my experience",
        "I led a team of four engineers",
        "Thank you for your time",
    ],
    "Small talk": [
        "Nice to meet you",
        "How have you been",
        "That sounds great",
    ],
    "Tricky words": [
        "water bottle",
        "world",
        "develop",
        "comfortable",
        "entrepreneur",
        "vegetable",
    ],
}

IPA_VARIANTS = {
    "ɑː": ["ɑ", "aː", "a"],
    "ɔː": ["ɔ", "oː", "o"],
    "ɜ": ["ɝ", "ɚ", "ə"],
    "ɝ": ["ː", "ɚ", "ə"],
    "ɚ": ["ɜː", "", "ə"],
    "ɹ": ["r"], "r": ["ɹ"],
    "ɡ": ["g"], "g": ["ɡ"],
    "j": ["y"], "y": ["j"],
    "ɾ": ["t"],
    "e": ["eɪ", ""], "o": ["oʊ", "ɔ"],
    "i": ["iː", "ɪ"], "u": ["uː", "ʊ"],
    "ɒ": ["ɑ", "ɔ"],
}


def _norm_token(tok):
    """Map one espeak token onto symbol(s) that exist in the model vocab."""
    for cand in [tok] + IPA_VARIANTS.get(tok, []):
        if cand in VOCAB:
            return [cand]
    out = []
    i = 0
    while i < len(tok):
        hit = None
        for length in (3, 2, 1):
            chunk = tok[i:i + length]
            if chunk in VOCAB:
                hit = chunk
                break
        if hit:
            out.append(hit)
            i += len(hit)
            continue
        repl = next((v for v in IPA_VARIANTS.get(tok[i], []) if v in VOCAB),
                    None)
        if repl:
            out.append(repl)
        i += 1
    return out


def get_expected_phones(text):
    raw = phonemize(text, language="en-us", backend="espeak",
                    separator=SEP, strip=True, preserve_punctuation=False,
                    with_stress=False)
    phones = []
    for tok in raw.split():
        phones.extend(_norm_token(tok))
    return phones


def verdict(overall):
    if overall >= 75:
        return "Good. Sounds clear."
    if overall >= 50:
        return "Average. A few sounds to polish."
    return "Needs practice. Try again slowly."


def analyze(audio_path, target_text):
    try:
        text = (target_text or "").strip().lower()
        if not text:
            return (IDLE_TIMELINE, "Pick or type a phrase first.",
                    gr.update(), gr.update())
        if audio_path is None:
            return (IDLE_TIMELINE, "Now record yourself saying it.",
                    gr.update(), gr.update())
        expected = get_expected_phones(text)
        if not expected:
            return (IDLE_TIMELINE, "Could not phonemize that text.",
                    gr.update(), gr.update())
        logits = get_logits(audio_path)
        duration = librosa.get_duration(path=audio_path)
        results = compute_gop(logits, expected, VOCAB, ID_TO_PHONE, duration)
        if not results:
            return (IDLE_TIMELINE, "Recording too short. Try again.",
                    gr.update(), gr.update())
        timeline = render_timeline(results)
        overall = round(sum(r["confidence"] for r in results) / len(results))
        worst = min(results, key=lambda r: r["gop_score"])
        tip = generate_feedback(results, text)
        summary = ("**" + verdict(overall) + "**  ·  Overall " + str(overall)
                   + "/100\n\nWeakest sound: /" + worst["expected"]
                   + "/ (we heard /" + worst["model_actually_heard"] + "/).\n\n"
                   + tip)
        return timeline, summary, \
            gr.update(visible=False), gr.update(visible=True)
    except Exception as e:
        return (IDLE_TIMELINE,
                "**Error:** " + str(e) + " (details in the terminal).",
                gr.update(), gr.update())


def play_reference(target_text):
    """One click = load AND play; the player appears only once it has audio."""
    text = (target_text or "").strip()
    if not text:
        return gr.update(), "Type or pick a phrase first."
    try:
        audio_bytes = synthesize_wav(text)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return gr.update(value=tmp.name, visible=True), ""
    except Exception as e:
        return gr.update(), "TTS failed: " + str(e)


def play_guide_example(choice):
    """Synthesize and play the example word for the picked phoneme."""
    word = CHOICE_TO_WORD.get(choice or "", "")
    if not word:
        return gr.update(), "Pick a sound first."
    try:
        audio_bytes = synthesize_wav(word)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return gr.update(value=tmp.name, visible=True), ""
    except Exception as e:
        return gr.update(), "TTS failed: " + str(e)


try:
    _WAVE = {"waveform_options": gr.WaveformOptions(
        waveform_height=56,
        waveform_color="#39415A",
        waveform_progress_color="#8B5CF6")}
except Exception:
    _WAVE = {}

PLAY_JS = """
() => {
  let done = false;
  const kick = () => {
    if (done) return;
    const root = document.getElementById('ref-audio') || document;
    const audio = root.querySelector('audio');
    if (audio) {
      if (!audio.paused) { done = true; return; }
      audio.play().then(() => { done = true; }).catch(() => {});
      return;
    }
    const btn = Array.from(root.querySelectorAll('button')).find(b => {
      const l = (b.getAttribute('aria-label') || '').toLowerCase();
      return l.includes('play') && !l.includes('pause');
    });
    if (btn) { btn.click(); done = true; }
  };
  [250, 700, 1400, 2200].forEach(t => setTimeout(kick, t));
}
"""

LOOP_JS = """
() => {
  if (!document.getElementById('vf-loop-style')) {
    const st = document.createElement('style');
    st.id = 'vf-loop-style';
    st.textContent = '.compact-audio .controls{flex-wrap:nowrap !important;}';
    document.head.appendChild(st);
  }

  const serverLoop = (wrapId, on) => {
    const map = {
      'ref-audio': on ? 'vf-loop-on-ref' : 'vf-loop-off-ref',
      'rec-audio': on ? 'vf-loop-on-rec' : 'vf-loop-off-rec'
    };
    const el = document.getElementById(map[wrapId]);
    if (el) el.click();
  };

  const sync = (wrap) => {
    const on = wrap.dataset.vfLoop === '1';
    const btn = wrap.querySelector('.vf-loop-btn');
    if (btn) btn.classList.toggle('on', on);
    const audio = wrap.querySelector('audio');
    if (audio) {
      audio.loop = on;
      if (!audio.dataset.vfEnded) {
        audio.dataset.vfEnded = '1';
        audio.addEventListener('ended', () => {
          if (wrap.dataset.vfLoop === '1') {
            audio.currentTime = 0;
            audio.play();
          }
        });
      }
    }
  };

  const install = (wrapId) => {
    const wrap = document.getElementById(wrapId);
    if (!wrap || wrap.querySelector('.vf-loop-btn')) return;
    const controls = wrap.querySelector('.controls') || wrap;
    const btns = controls.querySelectorAll('button');
    if (!btns.length) return;

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'vf-loop-btn';
    btn.title = 'Loop playback';
    btn.setAttribute('aria-label', 'Loop playback');
    btn.innerHTML = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>';
    btn.addEventListener('click', () => {
      const on = wrap.dataset.vfLoop !== '1';
      wrap.dataset.vfLoop = on ? '1' : '0';
      sync(wrap);
      serverLoop(wrapId, on);
    });

    if (wrapId === 'rec-audio') {
      const trim = Array.from(btns).find(b =>
        ((b.getAttribute('aria-label') || '') + (b.title || ''))
          .toLowerCase().includes('trim'));
      (trim || btns[btns.length - 1]).parentElement.appendChild(btn);
    } else {
      wrap.style.position = 'relative';
      btn.style.position = 'absolute';
      btn.style.right = '12px';
      btn.style.bottom = '8px';
      btn.style.margin = '0';
      wrap.appendChild(btn);
    }
    sync(wrap);
  };

  const hero = () => {
    const hv = document.querySelector('.vf-hero video');
    if (!hv) return;
    hv.muted = true;
    hv.loop = true;
    hv.controls = false;
    if (hv.paused) { hv.play().catch(() => {}); }
    if (!hv.dataset.vfErr) {
      hv.dataset.vfErr = '1';
      hv.addEventListener('error', () => { hv.style.display = 'none'; });
      setTimeout(() => { if (hv.readyState === 0) hv.style.display = 'none'; },
                 4000);
    }
  };

  setInterval(() => { install('ref-audio'); install('rec-audio'); }, 600);
  setInterval(hero, 800);
  setInterval(() => {
    ['ref-audio', 'rec-audio'].forEach(id => {
      const w = document.getElementById(id);
      if (w) sync(w);
    });
  }, 400);
}
"""

CSS = """
/* ============ page ============ */
html, body, gradio-app, .gradio-container { background:#0B0D12 !important; }
footer { display:none !important; }
html, body { overflow-x:hidden !important; }

.gradio-container,
.gradio-container > div,
.gradio-container > div > div,
.main, .wrap {
  max-width:none !important;
  width:100% !important;
  margin-left:0 !important; margin-right:0 !important;
  padding-left:0 !important; padding-right:0 !important;
}

body::before {
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(700px 420px at 88% -8%, rgba(99,102,241,.16), transparent 62%),
    radial-gradient(760px 520px at -12% 34%, rgba(236,72,153,.10), transparent 62%),
    radial-gradient(900px 640px at 50% 112%, rgba(139,92,246,.12), transparent 62%);
}
.gradio-container { position:relative; z-index:1; }

/* ============ HARD DARK-MODE LOCK ============ */
:root {
  --body-background-fill:#0B0D12 !important;
  --body-text-color:#E8EAF0 !important;
  --background-fill-primary:rgba(255,255,255,.03) !important;
  --background-fill-secondary:rgba(255,255,255,.02) !important;
  --border-color-primary:rgba(255,255,255,.10) !important;
  --border-color-accent:#8B5CF6 !important;
  --color-accent:#8B5CF6 !important;
  --color-accent-soft:rgba(139,92,246,.16) !important;
  --block-background-fill:rgba(255,255,255,.03) !important;
  --block-border-color:rgba(255,255,255,.10) !important;
  --block-label-background-fill:rgba(255,255,255,.03) !important;
  --block-label-border-color:rgba(255,255,255,.10) !important;
  --block-label-text-color:#9AA3B2 !important;
  --block-title-text-color:#E8EAF0 !important;
  --block-info-text-color:#6E7684 !important;
  --panel-background-fill:rgba(255,255,255,.02) !important;
  --input-background-fill:rgba(255,255,255,.05) !important;
  --input-border-color:rgba(255,255,255,.12) !important;
  --input-text-color:#E8EAF0 !important;
  --input-placeholder-color:#6E7684 !important;
  --button-primary-background-fill:#4F46E5 !important;
  --button-primary-text-color:#FFFFFF !important;
  --button-primary-border-color:#4F46E5 !important;
  --button-secondary-background-fill:rgba(255,255,255,.05) !important;
  --button-secondary-text-color:#C9CFDA !important;
  --button-secondary-border-color:rgba(255,255,255,.12) !important;
  --checkbox-background-fill:rgba(255,255,255,.05) !important;
  --checkbox-border-color:rgba(255,255,255,.12) !important;
  --checkbox-label-background-fill:rgba(255,255,255,.05) !important;
  --checkbox-label-text-color:#E8EAF0 !important;
  --radio-dot-color:#8B5CF6 !important;
  --table-border-color:rgba(255,255,255,.10) !important;
  --table-even-background-fill:rgba(255,255,255,.03) !important;
  --table-odd-background-fill:rgba(255,255,255,.02) !important;
  --loader-color:#8B5CF6 !important;
  --slider-color:#8B5CF6 !important;
  --prose-text-color:#E8EAF0 !important;
}

:root {
  --button-large-padding: 9px 18px;
  --button-small-padding: 6px 12px;
  --button-large-text-size: 14px;
  --button-small-text-size: 12px;
}

@keyframes vf-grad {
  0%   { background-position:0% 50%; }
  50%  { background-position:100% 50%; }
  100% { background-position:0% 50%; }
}

/* ============ full-bleed split hero ============ */
.vf-hero {
  --hero-bg:#000000;
  --hero-ink:#FFFFFF;
  --hero-sub:#A9B0BE;
  --hero-mid:rgba(0,0,0,.88);
  --hero-end:rgba(0,0,0,0);
  position:relative; overflow:hidden;
  width:100%; min-height:100vh;
  background:var(--hero-bg);
  border:none !important; box-shadow:none !important;
  padding:0 !important; margin:0 !important;
}
.vf-hero--light {
  --hero-bg:#F7F6F2;
  --hero-ink:#101322;
  --hero-sub:#5D6470;
  --hero-mid:rgba(247,246,242,.9);
  --hero-end:rgba(247,246,242,0);
}
.vf-media-video {
  position:absolute !important; inset:0 !important;
  z-index:1 !important;
  padding:0 !important; margin:0 !important;
  border:none !important; box-shadow:none !important;
  background:transparent !important;
  overflow:hidden !important;
  pointer-events:none;
}
.vf-hero-video {
  position:absolute; top:0; right:0; height:100%; width:56%;
  object-fit:cover; object-position:center;
  filter:saturate(1.15) contrast(1.05);
  border:none !important; outline:none !important;
  box-shadow:none !important; background:#000 !important;
  border-radius:0 !important;
}
.vf-hero-video-missing {
  background:radial-gradient(60% 80% at 78% 50%,
              rgba(99,102,241,.28), rgba(0,0,0,0) 70%);
}
.vf-hero .vf-tint {
  position:absolute; inset:0;
  background:linear-gradient(160deg,
    rgba(99,102,241,.16), rgba(139,92,246,.08) 45%, rgba(236,72,153,.12));
  mix-blend-mode:screen;
}
.vf-fade {
  position:absolute; inset:0;
  background:linear-gradient(90deg,
    var(--hero-bg) 0%, var(--hero-bg) 34%,
    var(--hero-mid) 52%, var(--hero-end) 82%);
}
.vf-hero .vf-media { position:absolute; inset:0; z-index:2; pointer-events:none; }
.vf-hero > *:not(.vf-media) { position:relative; z-index:3; }
.vf-hero-copy { padding:16vh 6% 0 7%; max-width:760px; text-align:left; }
.vf-hero .vf-title { color:var(--hero-ink); font-size:52px; margin-top:18px; }
.vf-hero .vf-sub { color:var(--hero-sub); }
.vf-cta { display:block !important; width:100% !important;
          max-width:300px !important; margin:32px 0 0 7% !important; }
.vf-foot { padding:18px 6% 0 7%; color:var(--hero-sub);
           font-size:12.5px; letter-spacing:.4px; opacity:.85; }
@media (prefers-reduced-motion: reduce) {
  .vf-hero-video { display:none; }
}
@media (max-width: 900px) {
  .vf-hero-video { width:100%; opacity:.45; }
  .vf-fade { background:linear-gradient(180deg,
             var(--hero-bg) 18%, var(--hero-mid) 55%, var(--hero-end) 100%); }
  .vf-hero-copy { padding-top:12vh; }
}

/* ============ centered content on inner pages ============ */
.vf-appbar, .card, .nav-row {
  width:min(1160px, calc(100% - 32px));
  margin-left:auto !important; margin-right:auto !important;
}
.nav-row { margin-top:10px !important; }
.nav-btn { padding:6px 16px !important; font-size:12.5px !important; }

/* ============ hero type ============ */
.vf-badge {
  display:inline-flex; align-items:center; gap:8px;
  background:rgba(99,102,241,.16); color:#C7CBF8;
  border:1px solid rgba(129,140,248,.4);
  border-radius:999px; padding:6px 14px;
  font-size:11px; font-weight:700; letter-spacing:1.2px;
}
.vf-hero--light .vf-badge { background:#EEF0FF; color:#4338CA;
                            border-color:#DCDFFB; }
.vf-badge .dot { width:7px; height:7px; border-radius:50%; background:#6366F1;
                 box-shadow:0 0 0 4px rgba(99,102,241,.18); }
.vf-title { line-height:1.05; font-weight:800; letter-spacing:-1.6px; }
.vf-title .accent, .vf-word .accent {
  background-image:linear-gradient(100deg,#6366F1,#8B5CF6,#EC4899,#6366F1);
  background-size:300% 300%; animation:vf-grad 8s ease infinite;
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.vf-sub { margin-top:14px; font-size:16px; }

/* ============ app bar ============ */
.vf-appbar { display:flex; align-items:baseline; gap:12px;
             justify-content:center; padding:26px 0 6px; }
.vf-word { font-weight:800; font-size:21px; letter-spacing:-.5px;
           color:#FFFFFF; }
.vf-appbar-sub { color:#6E7684; font-size:11px; letter-spacing:1.6px;
                 text-transform:uppercase; font-weight:700; }

/* ============ frosted-glass cards ============ */
.card {
  background:rgba(255,255,255,.035) !important;
  border:1px solid rgba(255,255,255,.08) !important;
  border-radius:20px;
  padding:24px 26px; margin-top:16px; margin-bottom:16px;
  box-shadow:0 24px 60px -35px rgba(0,0,0,.85);
  position:relative;
}
.card::before {
  content:""; position:absolute; top:0; left:8%; right:8%; height:1px;
  background:linear-gradient(90deg, transparent,
             rgba(139,92,246,.65), rgba(236,72,153,.5), transparent);
}
.sec { display:flex; align-items:center; gap:10px; color:#E8EAF0;
       font-size:12px; font-weight:800; letter-spacing:1.8px;
       margin:0 0 18px; }
.sec .num {
  width:26px; height:26px; border-radius:9px;
  background:linear-gradient(135deg,#6366F1,#8B5CF6);
  color:#FFFFFF; border:none;
  box-shadow:0 6px 14px -6px rgba(99,102,241,.8);
  font-size:12px; font-weight:800;
  display:inline-flex; align-items:center; justify-content:center;
}

/* ============ buttons ============ */
.pack-btn {
  background:rgba(255,255,255,.04) !important;
  border:1px solid rgba(255,255,255,.10) !important;
  color:#C9CFDA !important; border-radius:999px !important; font-weight:600;
  transition:all .15s ease;
}
.pack-btn:hover {
  border-color:rgba(139,92,246,.6) !important; color:#FFFFFF !important;
  background:rgba(139,92,246,.12) !important;
  transform:translateY(-1px);
  box-shadow:0 8px 20px -10px rgba(99,102,241,.6);
}
#ref-btn, #guide-btn {
  white-space:nowrap;
  background:rgba(99,102,241,.08) !important;
  border:1px solid rgba(129,140,248,.45) !important;
  color:#C7CBF8 !important; border-radius:999px !important; font-weight:700;
  transition:all .15s ease;
}
#ref-btn:hover, #guide-btn:hover {
  background:rgba(99,102,241,.16) !important; color:#FFFFFF !important;
  transform:translateY(-1px);
  box-shadow:0 8px 20px -10px rgba(99,102,241,.7);
}
.analyze-btn {
  border:none !important; color:#FFFFFF !important;
  border-radius:999px !important; font-weight:800; letter-spacing:.3px;
  text-shadow:0 1px 4px rgba(0,0,0,.35);
  display:block !important; width:100% !important;
  max-width:460px !important; margin:16px auto 0 !important;
  transition:all .15s ease;
}
.vf-grad {
  background-image:linear-gradient(100deg,#4F46E5,#8B5CF6,#EC4899,#4F46E5) !important;
  background-size:300% 300% !important;
  animation:vf-grad 8s ease infinite;
  box-shadow:0 8px 20px -10px rgba(99,102,241,.6) !important;
}
.analyze-btn:hover { filter:brightness(1.08); transform:translateY(-1px); }

/* ============ players ============ */
.compact-audio .component-wrapper {
  padding:6px 14px !important;
  background:rgba(255,255,255,.02) !important;
  border:1px solid rgba(255,255,255,.08) !important;
  border-radius:14px !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.05);
  transition:border-color .15s ease;
}
.compact-audio .component-wrapper:hover {
  border-color:rgba(139,92,246,.35) !important;
}
.compact-audio .timestamps { padding:0 !important; margin:0 !important;
                             font-size:11px !important; color:#6E7684 !important; }
.compact-audio .controls { margin-top:0 !important; }
.compact-audio div[class*="waveform"] { height:56px !important;
                                        max-height:56px !important; }
.head-buttons button, [class*="icon-btn"] {
  background:rgba(255,255,255,.05) !important; color:#9AA3B2 !important;
  border:1px solid rgba(255,255,255,.10) !important; border-radius:8px !important;
}

/* ============ loop button ============ */
.vf-loop-btn {
  background:transparent; border:none; color:#6E7684;
  padding:2px 6px; border-radius:8px; cursor:pointer;
  display:inline-flex; align-items:center;
}
.vf-loop-btn:hover { color:#E8EAF0; background:rgba(255,255,255,.06); }
.vf-loop-btn.on { color:#A78BFA; background:rgba(139,92,246,.15);
                  box-shadow:inset 0 0 0 1px rgba(139,92,246,.4); }

/* ============ text ============ */
.prose, .prose p, .prose strong, .prose em, .prose li,
.prose h1, .prose h2, .prose h3 { color:#E8EAF0 !important; }
.ref-status { color:#F0A35E !important; font-size:12px; min-height:14px; }
.vf-js-hidden { display:none !important; }

/* ============ left-aligned bigger CTA + larger type ============ */
.vf-hero .vf-cta {
  display:block !important;
  width:100% !important;
  max-width:380px !important;
  margin:36px 0 0 7% !important;
  padding:14px 30px !important;
  font-size:16px !important;
  letter-spacing:.4px !important;
}
:root {
  --body-text-size: 15px;
  --input-text-size: 15px;
  --button-large-text-size: 15px;
  --button-small-text-size: 13px;
  --block-label-text-size: 14px;
  --block-title-text-size: 15px;
  --prose-text-size: 16px;
}
.sec { font-size:13px !important; letter-spacing:2px !important; }
.nav-btn { font-size:13.5px !important; padding:8px 18px !important; }
.vf-word { font-size:22px !important; }
.vf-appbar-sub { font-size:12px !important; }
.prose, .prose p, .prose li, .prose strong { font-size:16px !important;
                                             line-height:1.65 !important; }
.compact-audio .timestamps { font-size:12px !important; }
.phone-chip { font-size:16px !important; }
span[style*="width:140px"] { font-size:16px !important; }
.ref-status { font-size:13px !important; }
.vf-foot { font-size:13px !important; }

/* ============ hero left edge + dropdown fixes ============ */
.vf-hero { padding-left:7% !important; padding-right:6% !important; }
.vf-hero-copy { padding:16vh 0 0 0 !important; }
.vf-hero .vf-cta { margin:36px 0 0 0 !important; }
.vf-foot { padding:18px 0 0 0 !important; }
@media (max-width: 900px) {
  .vf-hero { padding-left:8% !important; padding-right:8% !important; }
  .vf-hero-copy { padding-top:12vh !important; }
}
.card { overflow:visible !important; }
.card { z-index:1; }
.card:hover { z-index:40; }
.card:focus-within { z-index:80; }
.card {
  backdrop-filter:none !important;
  -webkit-backdrop-filter:none !important;
}
[role="listbox"], .options, [class*="listbox"] { z-index:9999 !important; }

/* ============ sound guide ============ */
.vf-guide-intro { color:#C9CFDA; font-size:15px; line-height:1.65;
                  margin:0 0 18px; }
.vf-guide-h { color:#E8EAF0; font-size:13px; font-weight:800;
              letter-spacing:2px; margin:20px 0 10px; }
.vf-guide-grid { display:grid;
                 grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
                 gap:12px; }
.vf-guide-card { background:rgba(255,255,255,.04);
                 border:1px solid rgba(255,255,255,.09);
                 border-radius:14px; padding:14px 16px; }
.vf-guide-sym { display:inline-flex; align-items:center; justify-content:center;
                min-width:44px; height:44px; border-radius:10px;
                font-family:ui-monospace,monospace; font-weight:700;
                font-size:18px; }
.vf-guide-sym.v { background:rgba(99,102,241,.16); color:#C7CBF8;
                  border:1px solid rgba(99,102,241,.35); }
.vf-guide-sym.d { background:rgba(236,72,153,.14); color:#F9A8D4;
                  border:1px solid rgba(236,72,153,.35); }
.vf-guide-sym.c { background:rgba(45,212,191,.12); color:#5EEAD4;
                  border:1px solid rgba(45,212,191,.3); }
.vf-guide-cue { margin-top:9px; color:#E8EAF0; font-size:14px; font-weight:600; }
.vf-guide-ex { margin-top:2px; color:#9AA3B2; font-size:13px; }
"""

try:
    THEME = gr.themes.Soft(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )
except Exception:
    THEME = None

APPBAR_HTML = (
    "<div class='vf-appbar'><span class='vf-word'>Voice<span class='accent'>"
    "Fix</span></span>"
    "<span class='vf-appbar-sub'>Pronunciation coach</span></div>"
)

_BLOCKS_KW = {"title": "VoiceFix"}
if _GRADIO_MAJOR < 6 and THEME is not None:
    _BLOCKS_KW["theme"] = THEME

with gr.Blocks(**_BLOCKS_KW) as demo:
    vf_on_ref = gr.Button("", elem_id="vf-loop-on-ref",
                          elem_classes=["vf-js-hidden"])
    vf_off_ref = gr.Button("", elem_id="vf-loop-off-ref",
                           elem_classes=["vf-js-hidden"])
    vf_on_rec = gr.Button("", elem_id="vf-loop-on-rec",
                          elem_classes=["vf-js-hidden"])
    vf_off_rec = gr.Button("", elem_id="vf-loop-off-rec",
                           elem_classes=["vf-js-hidden"])

    # ================= PAGE 1: LANDING =================
    with gr.Column(visible=True,
                   elem_classes=["vf-hero",
                                 "vf-hero--dark" if HERO_DARK
                                 else "vf-hero--light"]) as landing_view:
        gr.HTML(MEDIA_HTML, elem_classes=["vf-media"])
        gr.HTML(COPY_HTML, elem_classes=["vf-hero-copy"])
        start_btn = gr.Button("Start practicing",
                              elem_classes=["analyze-btn", "vf-grad",
                                            "vf-cta"])
        gr.HTML(FOOT_HTML)

    # ================= PAGE 2: PRACTICE =================
    with gr.Column(visible=False) as practice_view:
        gr.HTML(APPBAR_HTML)
        with gr.Row(elem_classes=["nav-row"]):
            home_btn_2 = gr.Button("Home", scale=0, min_width=90,
                                   elem_classes=["pack-btn", "nav-btn"])
            guide_btn_2 = gr.Button("Sound guide", scale=0, min_width=130,
                                    elem_classes=["pack-btn", "nav-btn"])
        with gr.Column(elem_classes=["card"]):
            gr.HTML("<div class='sec'><span class='num'>1</span>"
                    "CHOOSE A PHRASE</div>")
            with gr.Row():
                pack_buttons = [gr.Button(name, scale=1,
                                          elem_classes=["pack-btn"])
                                for name in SCENARIO_PACKS]
            with gr.Row(equal_height=True):
                word_input = gr.Dropdown(show_label=False,
                                         choices=SCENARIO_PACKS["Tricky words"],
                                         value="water bottle",
                                         allow_custom_value=True, scale=3)
                ref_btn = gr.Button("Hear the correct pronunciation",
                                    scale=2, elem_id="ref-btn")
            ref_audio = gr.Audio(show_label=False, interactive=False,
                                 autoplay=True, loop=False, visible=False,
                                 buttons=["download"],
                                 elem_id="ref-audio",
                                 elem_classes=["compact-audio"], **_WAVE)
            ref_status = gr.Markdown("", elem_classes=["ref-status"])

        with gr.Column(elem_classes=["card"]):
            gr.HTML("<div class='sec'><span class='num'>2</span>"
                    "RECORD YOURSELF</div>")
            audio_input = gr.Audio(sources=["microphone"], type="filepath",
                                   loop=False, show_label=False,
                                   elem_id="rec-audio",
                                   elem_classes=["compact-audio"], **_WAVE)
            analyze_btn = gr.Button("Analyze Pronunciation", variant="primary",
                                    elem_classes=["analyze-btn", "vf-grad"])

    # ================= PAGE 3: RESULT =================
    with gr.Column(visible=False) as result_view:
        gr.HTML(APPBAR_HTML)
        with gr.Row(elem_classes=["nav-row"]):
            back_btn = gr.Button("Back to recording", scale=0, min_width=180,
                                 elem_classes=["pack-btn", "nav-btn"])
            home_btn_3 = gr.Button("Home", scale=0, min_width=90,
                                   elem_classes=["pack-btn", "nav-btn"])
            guide_btn_3 = gr.Button("Sound guide", scale=0, min_width=130,
                                    elem_classes=["pack-btn", "nav-btn"])
        with gr.Column(elem_classes=["card"]):
            gr.HTML("<div class='sec'><span class='num'>3</span>"
                    "YOUR RESULT</div>")
            with gr.Accordion("How the scoring works", open=False):
                gr.Markdown(
                    "**New here?** Every spoken word is a sequence of individual "
                    "sounds (phonemes). The **Target** row shows the sounds your "
                    "phrase is made of; the **What we heard** row shows what the "
                    "speech model actually recognised in your recording, sound by "
                    "sound. A '?' chip means that slice of your recording contained "
                    "no clear speech (too quiet, too fast, or cut off). Speak a "
                    "little louder and slower and it will fill in.\n\n"
                    "**Every sound gets a GOP score** (Goodness of Pronunciation): "
                    "how closely what the model heard matched the target sound. "
                    "We compare the target phone's probability against the best "
                    "alternative on speech frames, then convert to a 0-100 "
                    "confidence.\n\n"
                    "**Chip colors:** Green = 75+ (good) | Yellow = 50-74 (average) | "
                    "Red = below 50 (needs work).\n\n"
                    "New to these symbols? Open the **Sound guide** for a plain "
                    "explanation and an audio example of every sound."
                )
            timeline_output = gr.HTML(IDLE_TIMELINE)
            output = gr.Markdown("")

    # ================= PAGE 4: SOUND GUIDE =================
    with gr.Column(visible=False) as guide_view:
        gr.HTML(APPBAR_HTML)
        with gr.Row(elem_classes=["nav-row"]):
            guide_back = gr.Button("Back to practice", scale=0, min_width=180,
                                   elem_classes=["pack-btn", "nav-btn"])
            guide_home = gr.Button("Home", scale=0, min_width=90,
                                   elem_classes=["pack-btn", "nav-btn"])
        with gr.Column(elem_classes=["card"]):
            gr.HTML("<div class='sec'><span class='num'>?</span>"
                    "SOUND GUIDE</div>")
            with gr.Row(equal_height=True):
                guide_dropdown = gr.Dropdown(show_label=False,
                                             choices=GUIDE_CHOICES,
                                             value=GUIDE_CHOICES[0], scale=3)
                guide_play = gr.Button("Hear example", scale=1,
                                       elem_id="guide-btn")
            guide_audio = gr.Audio(show_label=False, interactive=False,
                                   autoplay=True, loop=False, visible=False,
                                   buttons=["download"],
                                   elem_id="guide-audio",
                                   elem_classes=["compact-audio"], **_WAVE)
            guide_status = gr.Markdown("", elem_classes=["ref-status"])
            gr.HTML(GUIDE_HTML)

    # ---------------- events ----------------
    def goto(target):
        order = ["landing", "practice", "result", "guide"]
        return tuple(gr.update(visible=(v == target)) for v in order)

    def pick(name):
        return gr.update(choices=SCENARIO_PACKS[name],
                         value=SCENARIO_PACKS[name][0])

    for btn, name in zip(pack_buttons, SCENARIO_PACKS):
        btn.click(fn=pick, inputs=gr.State(name), outputs=word_input)

    start_btn.click(fn=lambda: goto("practice"),
                    outputs=[landing_view, practice_view, result_view,
                             guide_view])
    home_btn_2.click(fn=lambda: goto("landing"),
                     outputs=[landing_view, practice_view, result_view,
                              guide_view])
    home_btn_3.click(fn=lambda: goto("landing"),
                     outputs=[landing_view, practice_view, result_view,
                              guide_view])
    guide_home.click(fn=lambda: goto("landing"),
                     outputs=[landing_view, practice_view, result_view,
                              guide_view])
    guide_btn_2.click(fn=lambda: goto("guide"),
                      outputs=[landing_view, practice_view, result_view,
                               guide_view])
    guide_btn_3.click(fn=lambda: goto("guide"),
                      outputs=[landing_view, practice_view, result_view,
                               guide_view])
    guide_back.click(fn=lambda: goto("practice"),
                     outputs=[landing_view, practice_view, result_view,
                              guide_view])
    back_btn.click(fn=lambda: goto("practice"),
                   outputs=[landing_view, practice_view, result_view,
                            guide_view])

    ref_btn.click(fn=play_reference, inputs=word_input,
                  outputs=[ref_audio, ref_status]).then(fn=None, js=PLAY_JS)

    guide_play.click(fn=play_guide_example, inputs=guide_dropdown,
                     outputs=[guide_audio, guide_status])

    analyze_btn.click(fn=analyze, inputs=[audio_input, word_input],
                      outputs=[timeline_output, output,
                               practice_view, result_view])

    vf_on_ref.click(lambda: gr.update(loop=True), outputs=ref_audio)
    vf_off_ref.click(lambda: gr.update(loop=False), outputs=ref_audio)
    vf_on_rec.click(lambda: gr.update(loop=True), outputs=audio_input)
    vf_off_rec.click(lambda: gr.update(loop=False), outputs=audio_input)

    demo.load(fn=None, js=LOOP_JS)

if __name__ == "__main__":
    _LAUNCH_KW = {"css": CSS, "allowed_paths": [ASSETS_DIR]}
    if _GRADIO_MAJOR >= 6 and THEME is not None:
        _LAUNCH_KW["theme"] = THEME
    demo.launch(server_name="0.0.0.0", **_LAUNCH_KW)