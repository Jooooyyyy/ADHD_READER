import streamlit as st
import re, time, tempfile, string
from collections import Counter

def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+(?=[A-Z0-9‘“\(\[\u4e00-\u9fff])", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        return [text]
    return parts

def chunk(sentences, size):
    return [sentences[i:i+size] for i in range(0, len(sentences), size)]

def bionicize(s: str, ratio: float = 0.4):
    out = []
    for tok in re.split(r"(\s+)", s):
        if tok.isspace() or tok == "":
            out.append(tok); continue
        pure = re.sub(r"[^\w\u4e00-\u9fff]", "", tok, flags=re.UNICODE)
        cut = max(1, round(len(pure)*ratio))
        out.append(f"**{tok[:cut]}**{tok[cut:]}")
    return "".join(out)

def extract_outline(md_text: str):
    outline = []
    for line in md_text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            outline.append((len(m.group(1)), m.group(2).strip()))
    return outline

STOPWORDS = set("""a an the and or but if while for from into during of on in to at by with without within over under after before about as is are was were be been being do does did doing have has had having not no nor too very can could should would may might must will just than then so such it its
i me my we our you your he she they them his her their this that these those which who whom whose where when why how also more most less least many much few each per via etc""".split())

def tokenize_words(text: str):
    text = text.lower()
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and not t.isdigit()]
    return tokens

def sentence_scores(sentences):
    word_freq = Counter()
    sent_tokens = []
    for s in sentences:
        toks = tokenize_words(s)
        sent_tokens.append(toks)
        word_freq.update(toks)
    if not word_freq:
        return [0.0]*len(sentences)
    maxf = max(word_freq.values())
    for w in list(word_freq.keys()):
        word_freq[w] = word_freq[w] / maxf
    scores = []
    for toks in sent_tokens:
        score = sum(word_freq.get(t, 0.0) for t in toks) / (len(toks) + 1e-9)
        scores.append(score)
    return scores

def summarize_text(text: str, max_sentences: int = 5):
    sents = split_sentences(text)
    if len(sents) <= max_sentences:
        return sents
    scores = sentence_scores(sents)
    idxs = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)[:max_sentences]
    idxs = sorted(idxs)
    return [sents[i] for i in idxs]

def one_sentence_gist(text: str):
    sents = split_sentences(text)
    if not sents:
        return ""
    scores = sentence_scores(sents)
    top = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)[:2]
    gist = " ".join([sents[i] for i in sorted(top)])
    gist = re.sub(r"\([^)]*\)", "", gist)
    gist = re.sub(r"\s{2,}", " ", gist).strip()
    return gist

def eli5_simplify(text: str):
    replacements = {
        "utilize": "use", "facilitate": "help", "prioritize": "focus on",
        "methodology": "method", "demonstrate": "show", "approximately": "about",
        "individuals": "people", "participants": "people", "children": "kids",
        "assistance": "help", "purchase": "buy", "consume": "use",
        "implement": "do", "optimal": "best", "significant": "big",
        "beneficial": "good", "advantageous": "helpful", "consequently": "so",
        "nevertheless": "but", "moreover": "also", "furthermore": "also",
        "therefore": "so", "however": "but"
    }
    sents = split_sentences(text)
    out = []
    for s in sents:
        s2 = s
        for k, v in replacements.items():
            s2 = re.sub(rf"\b{k}\b", v, s2, flags=re.IGNORECASE)
        if len(s2) > 200 and "," in s2:
            parts = [p.strip() for p in s2.split(",") if p.strip()]
            for p in parts:
                out.append(p.lstrip())
        else:
            out.append(s2.strip())
    return " ".join(out[:6])

def key_terms_and_glossary(text: str, topk: int = 10):
    tokens = tokenize_words(text)
    freq = Counter(tokens)
    common = [w for w, c in freq.most_common(topk)]
    sents = split_sentences(text)
    gloss = {}
    for term in common:
        for s in sents:
            if re.search(rf"\b{re.escape(term)}\b", s, flags=re.IGNORECASE):
                gloss[term] = s.strip()
                break
    return common, gloss

def quick_questions(text: str):
    sents = split_sentences(text)
    if not sents:
        return []
    qs = []
    qs.append("What is the main idea of this section? (Answer in one sentence.)")
    if len(sents) >= 2:
        qs.append("Name one key detail or example mentioned.")
    qs.append("After reading, what is one action you could take or one thing you still wonder about?")
    return qs

def tts_synthesize_to_wav(text: str, rate: int = 180, voice: str = None):
    try:
        import pyttsx3
    except Exception as e:
        return None, f"pyttsx3 not installed. Please run: pip install pyttsx3  ({e})"
    if not text.strip():
        return None, "No text to read."
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", rate)
        if voice:
            engine.setProperty("voice", voice)
    except Exception:
        pass
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        return tmp_path, None
    except Exception as e:
        return None, f"TTS error: {e}"

st.set_page_config(page_title="ADHD-Friendly Reader (Python)", layout="wide")
st.title("ADHD-Friendly Reader — Python / Streamlit (+ TTS & Understanding)")

with st.sidebar:
    st.header("Controls")
    chunk_size = st.slider("Chunk size (sentences)", 1, 6, 3)
    use_bionic = st.checkbox("Bionic reading", True)
    dyslexia = st.checkbox("Dyslexia-friendly (wider spacing/leading)", False)
    high_contrast = st.checkbox("High contrast", False)
    sprint_mins = st.slider("Focus sprint (minutes)", 5, 20, 7)

    if "timer_end" not in st.session_state:
        st.session_state.timer_end = None
    col_a, col_b = st.columns(2)
    if col_a.button("Start sprint"):
        st.session_state.timer_end = time.time() + sprint_mins*60
    if col_b.button("Stop sprint"):
        st.session_state.timer_end = None

    st.markdown("---")
    st.subheader("Checklist")
    for key in ["I set a goal", "I previewed the outline", "I took one note", "I finished my sprint"]:
        st.checkbox(key, key=key)

default_text = """# Try me with any article

This ADHD-friendly reader breaks text into small, swipeable chunks. Use the keyboard or buttons to move. Start with two or three sentences per chunk and adjust as you get comfortable.

## Tips
• Preview the outline first, then set a tiny goal for your next sprint.
• Reduce visual noise with High Contrast.
• If reading is tough today, listen first elsewhere, then skim the chunk.
"""
raw = st.text_area("Paste your text", value=default_text, height=220)

sentences = split_sentences(raw)
chunks = chunk(sentences, chunk_size)

if st.session_state.timer_end:
    left = max(0, int(st.session_state.timer_end - time.time()))
    mm, ss = divmod(left, 60)
    st.info(f"⏱ Focus sprint — {mm:02d}:{ss:02d}")

if "idx" not in st.session_state:
    st.session_state.idx = 0

left, right = st.columns([1,1])

with left:
    st.subheader("Reader")
    st.write(f"Chunk {min(st.session_state.idx+1, len(chunks))} / {len(chunks)} · {len(sentences)} sentences")

    style = []
    if high_contrast:
        style.append("body{background:#000;color:#fff;} .stTextArea textarea{background:#111;color:#eee;}")
    if dyslexia:
        style.append("p{line-height:1.9; letter-spacing:0.02em; word-spacing:0.2em;}")
    if style:
        st.markdown(f"<style>{' '.join(style)}</style>", unsafe_allow_html=True)

    if chunks:
        display_chunk = chunks[st.session_state.idx]
        for s in display_chunk:
            st.markdown(bionicize(s) if use_bionic else s)
    else:
        st.write("…")

    prog = 0 if not chunks else (st.session_state.idx+1)/len(chunks)
    st.progress(prog)

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("⟲ Restart"):
        st.session_state.idx = 0
    if c2.button("← Prev"):
        st.session_state.idx = max(0, st.session_state.idx-1)
    if c3.button("Next →"):
        st.session_state.idx = min(len(chunks)-1, st.session_state.idx+1)

    if c4.button("↓ Export notes"):
        notes = ["Notes (one line per chunk)\n"] + [f"{i+1}. {' '.join(ch)}" for i, ch in enumerate(chunks)]
        st.download_button("Download .txt", "\n".join(notes), file_name="adhd_reader_notes.txt")

    st.markdown("### 🔊 Text-to-Speech (offline)")
    tcol1, tcol2, tcol3 = st.columns([1,1,2])
    tts_rate = tcol1.slider("Speech rate", 100, 250, 180, 10)
    scope = tcol2.selectbox("Read scope", ["Current chunk", "Whole document"])
    if tcol3.button("Synthesize & Play"):
        text_to_read = " ".join(chunks[st.session_state.idx]) if (scope=="Current chunk" and chunks) else raw
        path, err = tts_synthesize_to_wav(text_to_read, rate=tts_rate)
        if err:
            st.error(err)
        else:
            with open(path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
            st.success("TTS ready. You can play it above or download it.")
            with open(path, "rb") as f:
                st.download_button("Download WAV", data=f.read(), file_name="adhd_tts.wav")

with right:
    st.subheader("AI Understanding (local)")
    scope2 = st.radio("Analyze", ["Current chunk", "Whole document"], horizontal=True)
    text_scope = " ".join(chunks[st.session_state.idx]) if (scope2=="Current chunk" and chunks) else raw

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Key bullets", "One-sentence gist", "ELI5", "Glossary", "Quick Qs"])

    with tab1:
        bullets = summarize_text(text_scope, max_sentences=5)
        if bullets:
            for s in bullets:
                st.markdown(f"- {s}")
        else:
            st.caption("No content to summarize.")

    with tab2:
        st.write(one_sentence_gist(text_scope) or "…")

    with tab3:
        st.write(eli5_simplify(text_scope) or "…")

    with tab4:
        terms, gloss = key_terms_and_glossary(text_scope, topk=10)
        if terms:
            st.markdown("**Key terms**: " + ", ".join(terms))
            st.markdown("---")
            st.markdown("**Glossary (context snippets)**")
            for k, v in gloss.items():
                st.markdown(f"- **{k}** — _{v}_")
        else:
            st.caption("No salient terms found.")

    with tab5:
        for q in quick_questions(text_scope):
            st.markdown(f"- {q}")

    st.markdown("---")
    st.subheader("Outline (Markdown headers)")
    outline = extract_outline(raw)
    if outline:
        for lvl, title in outline:
            st.write(("— " if lvl>=3 else "") + title)
    else:
        st.caption("No headers found. Add lines starting with # / ## / ### …")
