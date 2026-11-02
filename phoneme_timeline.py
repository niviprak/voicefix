"""phoneme_timeline.py -- two-row phoneme chips with GOP tooltips.
Chip colors: the pastel palette. Tooltips lead with a plain-language cue
(from phoneme_guide) so beginners can read every symbol on sight."""

try:
    from phoneme_guide import PHONE_CUES
except Exception:
    PHONE_CUES = {}


def _target_style(confidence):
    if confidence < 50:
        return "#FEE2E2", "#991B1B", "#FBC9C9"   # rose tint
    elif confidence < 75:
        return "#FEF3C7", "#92400E", "#FDE68A"   # amber tint
    else:
        return "#DCFCE7", "#166534", "#BBF7D0"   # mint tint


def _cue(sym):
    c = PHONE_CUES.get(sym, "")
    return (" · " + c) if c else ""


def render_timeline(results):
    if not results:
        return ("<div style='color:#6E7684;font-size:13px;text-align:center;"
                "padding:18px 0;'>Record to see the phoneme breakdown.</div>")

    def chip(r, row):
        if row == 0:
            bg, fg, border = _target_style(r["confidence"])
            content = r["expected"]
            title = (content + _cue(content) + " · GOP: " + str(r["gop_score"])
                     + " -> confidence " + str(r["confidence"]) + "/100")
        else:
            if r["model_actually_heard"] == "?":
                bg, fg, border = "#F3F4F6", "#6B7280", "#E5E7EB"
                title = ("No clear speech detected in this slice of your "
                         "recording (too quiet, too fast, or cut off).")
            elif r["model_actually_heard"] == r["expected"]:
                bg, fg, border = "#EEF2FF", "#4338CA", "#DDE3FC"
                title = ("The model heard /" + r["model_actually_heard"]
                         + "/" + _cue(r["model_actually_heard"]) + " here.")
            else:
                bg, fg, border = "#FFF7ED", "#9A3412", "#FED7AA"
                title = ("The model heard /" + r["model_actually_heard"]
                         + "/" + _cue(r["model_actually_heard"]) + " here.")
            content = r["model_actually_heard"]
        return ("<span class='phone-chip' title='" + title + "'"
                + " data-start-frac='" + str(round(r["start_frac"], 4))
                + "' data-end-frac='" + str(round(r["end_frac"], 4))
                + "' style='display:inline-block;min-width:46px;"
                  "padding:10px 12px;margin:0 4px;border-radius:10px;"
                  "background:" + bg + ";border:1px solid " + border + ";color:"
                + fg + ";font-family:ui-monospace,monospace;font-weight:700;"
                  "font-size:15px;text-align:center;"
                  "transition:transform .1s ease;'>"
                + content + "</span>")

    target_row = "".join(chip(r, 0) for r in results)
    heard_row = "".join(chip(r, 1) for r in results)
    return ("<div style='background:#12151D;border:1px solid #242A38;"
            "border-radius:16px;padding:22px 24px;"
            "box-shadow:0 1px 2px rgba(0,0,0,.4);'>"
            "<div style='display:flex;align-items:center;flex-wrap:wrap;'>"
            "<span style='width:140px;color:#E8EAF0;font-size:15px;"
            "font-weight:700;'>Target:</span>"
            "<div style='display:flex;flex-wrap:wrap;'>" + target_row + "</div></div>"
            "<div style='display:flex;align-items:center;flex-wrap:wrap;"
            "margin-top:14px;'>"
            "<span style='width:140px;color:#9AA3B2;font-size:15px;'>"
            "What we heard:</span>"
            "<div style='display:flex;flex-wrap:wrap;'>" + heard_row
            + "</div></div></div>")