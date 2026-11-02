"""
feedback.py
One beginner-friendly coaching tip per attempt, via a local Ollama model.
Written for users who have never seen phonetics: names the missed sound
with an example word, explains how to make it in plain language, and
gives one concrete practice step. Falls back to a deterministic tip
(with a built-in plain-language sound guide) if Ollama is down.
All copy is dash-free: no em dashes, no en dashes, no double hyphens.
"""
import re

import ollama

OLLAMA_MODEL = "qwen2.5:3b"  # must match `ollama list` output

SYSTEM_PROMPT = (
    "You are a warm pronunciation coach teaching complete beginners who have "
    "never studied phonetics. The user just said an English phrase and a "
    "computer identified which individual sounds came through clearly and "
    "which did not. Write ONE encouraging coaching tip in exactly three short "
    "sentences: (1) name the missed sound using a familiar example word, like "
    "'the TH in \"think\"'. Never rely on phonetic symbols alone. "
    "(2) explain in plain everyday words how to place the tongue, lips or "
    "teeth to produce that sound. (3) give one small practice step, such as "
    "repeating one word slowly three times. Maximum 60 words, simple "
    "vocabulary, no jargon, no bullet points, never repeat the prompt. "
    "Never use em dashes, en dashes or double hyphens; join ideas with "
    "commas, periods or brackets instead."
)

# Plain-language mini guide used by the offline fallback:
# sound -> (how to name it to a beginner, how to physically make it)
PHONE_GUIDE = {
    "ð": ("the voiced TH in 'this' and 'the'",
          "rest your tongue tip lightly between your teeth and let your voice buzz"),
    "θ": ("the soft TH in 'think' and 'thank'",
          "place your tongue tip between your teeth and blow air gently, without voice"),
    "ʒ": ("the ZH in 'vision' and 'measure'",
          "shape your mouth like 'sh' but switch your voice on"),
    "dʒ": ("the J in 'jump'",
           "touch the ridge behind your top teeth with your tongue, then release with voice"),
    "tʃ": ("the CH in 'chair'",
           "touch the ridge behind your top teeth, then release with a puff of air"),
    "ŋ": ("the NG in 'sing'",
          "hum while raising the back of your tongue so the air exits through your nose"),
    "ɹ": ("the R in 'red'",
          "curl your tongue back slightly without letting it touch anything"),
    "l": ("the L in 'light'",
          "tap your tongue tip on the ridge just behind your top teeth"),
    "v": ("the V in 'very'",
          "touch your top teeth to your bottom lip and buzz"),
    "w": ("the W in 'water'",
          "round your lips tightly, then glide straight into the vowel"),
    "iː": ("the EE in 'see'",
           "spread your lips wide and keep the tongue high and forward"),
    "ɪ": ("the I in 'sit'",
          "keep the tongue relaxed and slightly lower than in 'see'"),
    "æ": ("the A in 'cat'",
          "open your mouth wide and keep the tongue low and forward"),
    "ʌ": ("the U in 'cup'", "keep your mouth relaxed and slightly open"),
    "ə": ("the 'uh' in 'about'", "keep your whole mouth completely relaxed"),
    "ɜ": ("the IR in 'bird'",
          "keep the tongue mid and relaxed while your voice stays on"),
    "eɪ": ("the AY in 'say'", "glide smoothly from 'eh' to 'ee' in one move"),
    "aɪ": ("the I in 'my'", "glide from 'ah' to 'ee' in one move"),
    "ɔɪ": ("the OY in 'boy'", "glide from 'aw' to 'ee' in one move"),
    "oʊ": ("the O in 'go'", "glide from 'aw' to 'oo' with rounded lips"),
    "aʊ": ("the OW in 'now'", "glide from 'ah' to 'oo' in one move"),
    "ʊ": ("the OO in 'book'", "keep your lips rounded but relaxed"),
    "uː": ("the OO in 'food'", "push your lips forward into a tight circle"),
    "ɔ": ("the AW in 'saw'", "round your lips and keep the tongue low and back"),
    "ɑ": ("the AH in 'father'", "drop your jaw wide with relaxed lips"),
}


def _clean_tip(tip: str) -> str:
    """Safety net: strip em/en dashes and double hyphens from any tip,
    even if the language model ignores the instruction."""
    tip = tip.replace("—", ", ").replace("–", ", ").replace("--", ", ")
    tip = tip.replace(", ,", ", ").replace(" ,", ",")
    tip = re.sub(r"\s+", " ", tip)
    return tip.strip()


def build_prompt(results: list, target_text: str) -> str:
    worst = sorted(results, key=lambda r: r["gop_score"])[:3]
    lines = [f"Target phrase: '{target_text}'",
             "Sounds the listener struggled with (sound, then what it sounded like):"]
    for w in worst:
        if w["flagged"] or w["confidence"] < 75:
            lines.append(f"- /{w['expected']}/ sounded like /{w['model_actually_heard']}/")
    return "\n".join(lines)


def _fallback_tip(pool: list) -> str:
    w = min(pool, key=lambda r: r["gop_score"])
    name, how = PHONE_GUIDE.get(
        w["expected"],
        (f"the '{w['expected']}' sound",
         "listen to the model once and copy it slowly"))
    return (f"Don't worry, this one is very common. Your recording missed "
            f"{name}; it came out closer to /{w['model_actually_heard']}/. "
            f"To make it, {how}. Try the phrase once more very slowly, "
            f"stretching that sound, then say it again at normal speed.")


def generate_feedback(results: list, target_text: str) -> str:
    if not results:
        return ""
    weak = [r for r in results if r["flagged"] or r["confidence"] < 75]
    if not weak:
        return ("Great job! Every sound came through clearly. To build "
                "muscle memory, say the same phrase once more at a slightly "
                "faster, natural conversational pace.")
    prompt = build_prompt(results, target_text)
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.4, "num_predict": 120},
        )
        tip = _clean_tip(response["message"]["content"])
        if tip:
            return tip
    except Exception:
        pass
    return _fallback_tip(weak)