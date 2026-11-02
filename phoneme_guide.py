"""phoneme_guide.py -- general English phoneme guide.

Provides:
  PHONE_CUES      : symbol -> plain-language cue, reused by result tooltips
  GUIDE_CHOICES   : picker options for the hear-example player
  CHOICE_TO_WORD  : picker option -> example word to synthesize
  GUIDE_HTML      : the full guide page content (intro + grouped cards)
"""

# (group name, group key, [(symbol, plain cue, example words)])
GUIDE = [
    ("VOWELS", "v", [
        ("iː", "the EE in see", "see, sea"),
        ("ɪ", "the I in sit", "sit, ship"),
        ("ɛ", "the E in bed", "bed, head"),
        ("æ", "the A in cat", "cat, hand"),
        ("ɑː", "the AH in father", "father, calm"),
        ("ɔː", "the AW in saw", "saw, water"),
        ("ʊ", "the OO in book", "book, put"),
        ("uː", "the OO in food", "food, blue"),
        ("ʌ", "the U in cup", "cup, love"),
        ("ə", "the uh in about", "about, comma"),
        ("ɜː", "the IR in bird", "bird, learn"),
    ]),
    ("DIPHTHONGS (TWO VOWELS GLIDED TOGETHER)", "d", [
        ("eɪ", "the AY in say", "say, day"),
        ("aɪ", "the I in my", "my, time"),
        ("ɔɪ", "the OY in boy", "boy, coin"),
        ("oʊ", "the O in go", "go, home"),
        ("aʊ", "the OW in now", "now, mouth"),
        ("ɪə", "the EAR in here", "here, ear"),
        ("ɛə", "the AIR in care", "care, hair"),
    ]),
    ("CONSONANTS", "c", [
        ("p", "the P in pen", "pen, spin"),
        ("b", "the B in big", "big, job"),
        ("t", "the T in top", "top, stop"),
        ("d", "the D in dog", "dog, did"),
        ("k", "the K in key", "key, school"),
        ("ɡ", "the G in go", "go, give"),
        ("f", "the F in fun", "fun, phone"),
        ("v", "the V in very", "very, love"),
        ("θ", "the TH in think", "think, math"),
        ("ð", "the TH in this", "this, mother"),
        ("s", "the S in sun", "sun, city"),
        ("z", "the Z in zoo", "zoo, has"),
        ("ʃ", "the SH in ship", "ship, nation"),
        ("ʒ", "the ZH in vision", "vision, measure"),
        ("h", "the H in hat", "hat, hello"),
        ("tʃ", "the CH in chair", "chair, catch"),
        ("dʒ", "the J in jump", "jump, age"),
        ("m", "the M in man", "man, summer"),
        ("n", "the N in no", "no, know"),
        ("ŋ", "the NG in sing", "sing, finger"),
        ("l", "the L in light", "light, ball"),
        ("ɹ", "the R in red", "red, around"),
        ("w", "the W in wet", "wet, one"),
        ("j", "the Y in yes", "yes, you"),
    ]),
]

PHONE_CUES = {sym: cue for _name, _key, items in GUIDE for sym, cue, _ex in items}

GUIDE_CHOICES = []
CHOICE_TO_WORD = {}
for _name, _key, _items in GUIDE:
    for _sym, _cue, _ex in _items:
        _word = _ex.split(",")[0].strip()
        _choice = _sym + "  ·  " + _cue + "  ·  e.g. " + _word
        GUIDE_CHOICES.append(_choice)
        CHOICE_TO_WORD[_choice] = _word

INTRO_HTML = (
    "<p class='vf-guide-intro'>A phoneme is one single sound, the smallest "
    "building block of a spoken word. English spelling hides this: 'walk' is "
    "four letters but three sounds (w ɔː k), and 'experience' is ten letters "
    "but eight sounds. VoiceFix scores your speech sound by sound, which is "
    "why your results use these symbols. Every card below shows one sound, a "
    "plain way to remember it, and two example words. Use the picker above to "
    "hear any sound inside a real word.</p>"
)


def render_guide():
    parts = [INTRO_HTML]
    for name, key, items in GUIDE:
        parts.append("<div class='vf-guide-h'>" + name + " (" + str(len(items))
                     + ")</div><div class='vf-guide-grid'>")
        for sym, cue, examples in items:
            parts.append(
                "<div class='vf-guide-card'>"
                "<span class='vf-guide-sym " + key + "'>" + sym + "</span>"
                "<div class='vf-guide-cue'>" + cue + "</div>"
                "<div class='vf-guide-ex'>" + examples + "</div></div>"
            )
        parts.append("</div>")
    return "".join(parts)


GUIDE_HTML = render_guide()