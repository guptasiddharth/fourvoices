"""FourVoices — live demo (Streamlit).

Upload a short clip → Gemma's multimodal vision grounds the scene → four captions
in distinct voices (formal, sarcastic, humorous-tech, humorous-non-tech).

Deploy free on Streamlit Community Cloud from the public repo. Set the model
endpoint via Secrets (Settings → Secrets):
    LLM_BASE_URL = "https://api.fireworks.ai/inference/v1"
    LLM_API_KEY  = "fw-..."
    LLM_MODEL    = "accounts/fireworks/models/gemma-4-26b-a4b-it"
With no secrets it runs in stub mode (shows the four-voice structure offline).
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

# Push Streamlit secrets into the environment BEFORE importing app config.
# Guarded: accessing st.secrets with no secrets file raises — treat as stub mode.
def _secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        return None


for _k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "VC_LLM_MODE", "VC_N_FRAMES"):
    _v = _secret(_k)
    if _v is not None:
        os.environ[_k] = str(_v)

from app.caption import VideoCaptioner          # noqa: E402
from app.config import SETTINGS                  # noqa: E402
from app.styles import STYLES                    # noqa: E402

st.set_page_config(page_title="FourVoices", page_icon="🎬", layout="centered")
st.title("🎬 FourVoices")
st.caption("Grounded four-style video captioning with Gemma — one video, four voices, faithful to every frame.")

with st.sidebar:
    st.subheader("Model")
    st.write(f"**mode:** `{SETTINGS.mode}`")
    st.write(f"**model:** `{SETTINGS.model}`")
    st.write(f"**endpoint:** {SETTINGS.base_url or '(stub — set Secrets to go live)'}")
    st.markdown("---")
    st.markdown("**How it works**\n\nGround once, style four times: Gemma vision "
                "describes what's on screen, then all four voices are written from "
                "that same grounded description — accurate *and* on-tone.")

_LABELS = {"formal": "Formal", "sarcastic": "Sarcastic",
           "humorous_tech": "Humorous · Tech", "humorous_non_tech": "Humorous · Non-tech"}


def _render(result: dict) -> None:
    st.markdown("#### 👁 Gemma vision — grounded facts")
    st.info(result["grounded_facts"])
    st.markdown("#### 🎙 The four voices")
    for style in STYLES:
        cap = result["captions"][style.key]
        chk = result["checks"].get(style.key, {})
        badge = "✅" if chk.get("accuracy") and chk.get("tone") else "⚠️"
        with st.container(border=True):
            st.markdown(f"**{_LABELS[style.key]}**  {badge}")
            st.write(cap)
    st.caption(f"{result['n_frames']} frames sampled · distinct: {result['distinct']}")


# --- Static demo: the sample clip + its REAL Gemma captions (no model/keys needed) ---
_here = os.path.dirname(__file__)
_sample = os.path.join(_here, "eval", "sample_output", "dancing_cats.json")
_clip = os.path.join(_here, "eval", "sample_output", "dancing_cats.mp4")
if os.path.exists(_sample):
    import json as _json
    _d = _json.load(open(_sample))
    st.subheader("Sample — real Gemma output")
    if os.path.exists(_clip):
        st.video(_clip)
    st.caption(f"Gemma multimodal vision · {_d.get('n_frames', '?')} frames sampled across the clip")
    st.markdown("**What Gemma saw** (grounded facts)")
    st.info(_d.get("grounded_facts", ""))
    st.markdown("**The four voices**")
    for _s in STYLES:
        with st.container(border=True):
            st.markdown(f"**{_LABELS[_s.key]}**")
            st.write(_d["captions"].get(_s.key, ""))
    st.divider()

if SETTINGS.mode == "stub":
    # No model backend → don't show a live uploader (it would fabricate captions).
    st.info("👆 This is a **static demo** of the sample clip above (real Gemma output). "
            "To caption **your own** clips live, run the app with a model endpoint — "
            "either a **Fireworks API key** (`LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`) "
            "or a **local Gemma endpoint** via Ollama. See the repo README.")
else:
    tab_video, tab_text = st.tabs(["Caption a video", "Caption from a description"])

    with tab_video:
        up = st.file_uploader(
            "Upload a short clip (30s–2min)",
            type=["mp4", "m4v", "mov", "qt", "mkv", "webm", "avi", "flv", "f4v", "wmv",
                  "asf", "mpeg", "mpg", "m2v", "mts", "m2ts", "ts", "3gp", "3g2", "ogv",
                  "vob", "divx", "mxf", "gif"])
        if up and st.button("Caption it", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as f:
                f.write(up.read())
                path = f.name
            st.video(path)
            with st.spinner("Gemma is watching the clip and writing four voices…"):
                try:
                    _render(VideoCaptioner().caption(clip_path=path))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Inference failed: {exc}")

    with tab_text:
        facts = st.text_area("Describe what's in the video (grounded facts)",
                             "Five cartoon cats in pink sunglasses do a synchronized dance on a cyan background.")
        if st.button("Generate four voices"):
            with st.spinner("Writing four voices…"):
                try:
                    _render(VideoCaptioner().caption(facts=facts))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Inference failed: {exc}")
