"""
Shush — Misophonia Trigger Reducer (Streamlit)
Two tabs only: Upload, and Live (mic).

NOTE: naming this file python.py normally breaks things (TensorFlow re-imports
it and the app runs twice). The __name__ guard below prevents that, so it works
either way — but shush_app.py is still the clean name.

Run:
    python -m pip install streamlit streamlit-webrtc av imageio-ffmpeg soundfile scipy numpy torch tensorflow tensorflow-hub deepfilternet
    streamlit run shush_app.py
"""

import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
from scipy.signal import resample_poly

# ==========================================================================
# FIXED CONFIG (no user controls)
# ==========================================================================
TRIGGERS = {
    # core misophonia triggers
    "Chewing, mastication", "Biting", "Crunch", "Cough", "Throat clearing",
    "Sneeze", "Sniff", "Breathing", "Gasp", "Burping, eructation",
    "Computer keyboard", "Typing", "Typewriter", "Writing",
    "Finger snapping", "Tap", "Clicking", "Slap, smack",
    # precautionary extras
    "Wheeze", "Snoring", "Pant", "Snort", "Hiccup", "Whispering", "Grunt",
    "Cutlery, silverware", "Dishes, pots, and pans", "Chink, clink",
    "Tick", "Tick-tock", "Clickety-clack", "Scratch", "Crack",
}
THRESH_LO, THRESH_HI = 0.10, 0.40
MAX_REDUCTION_DB = 20
PAD_FRAMES = 1
USE_DEEPFILTERNET = True
SR_Y, SR_D = 16000, 48000
HOP, WIN = 0.48, 0.96
# ==========================================================================


@st.cache_resource(show_spinner="Loading YAMNet (one time)...")
def load_yamnet():
    import tensorflow_hub as hub
    m = hub.load("https://tfhub.dev/google/yamnet/1")
    with open(m.class_map_path().numpy().decode()) as f:
        names = [r["display_name"] for r in csv.DictReader(f)]
    return m, names


@st.cache_resource(show_spinner="Loading DeepFilterNet (one time)...")
def load_df():
    from df import init_df
    model, state, _ = init_df()
    return model, state


def extract(path, sr):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = Path(tempfile.mkdtemp()) / f"a_{sr}.wav"
    subprocess.run([ff, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                    "-c:a", "pcm_s16le", str(out)], check=True, capture_output=True)
    return out


def dilate(x, k):
    p = np.pad(x, k, mode="edge")
    return np.max([p[i:i + len(x)] for i in range(2 * k + 1)], axis=0)


def process_file(path):
    import tensorflow as tf
    yamnet, class_names = load_yamnet()
    trig_idx = [i for i, n in enumerate(class_names) if n in TRIGGERS]

    wav16, _ = sf.read(str(extract(path, SR_Y)), dtype="float32")
    scores, _, _ = yamnet(tf.constant(wav16))
    scores = scores.numpy()
    trig = scores[:, trig_idx].max(axis=1)

    centers = np.arange(len(trig)) * HOP + WIN / 2
    trig = dilate(trig, PAD_FRAMES)
    audio, sr = sf.read(str(extract(path, SR_D)), dtype="float32")
    t = np.arange(len(audio)) / sr
    env = np.interp(t, centers, trig)
    amount = np.clip((env - THRESH_LO) / (THRESH_HI - THRESH_LO + 1e-9), 0.0, 1.0)
    gain = 10 ** (-MAX_REDUCTION_DB * amount / 20)

    if USE_DEEPFILTERNET:
        import torch
        from df import enhance
        model, state = load_df()
        audio = enhance(model, state, torch.from_numpy(audio).unsqueeze(0)
                        ).squeeze(0).detach().cpu().numpy()

    n = min(len(audio), len(gain))
    out = np.clip(audio[:n] * gain[:n], -1.0, 1.0)
    out_path = Path(tempfile.mkdtemp()) / "cleaned.wav"
    sf.write(str(out_path), out, sr)
    return out_path


def run_app():
    st.set_page_config(page_title="Shush", page_icon="🔇")
    st.title("🔇 Shush")

    # Load models ONCE (cached). Reused by both tabs and every run after.
    load_yamnet()
    if USE_DEEPFILTERNET:
        load_df()

    tab_up, tab_rt = st.tabs(["📁 Upload", "🎙️ Live"])

    # ---- Upload ----------------------------------------------------------
    with tab_up:
        up = st.file_uploader("Upload audio / video",
                              type=["mp3", "wav", "m4a", "aac", "flac", "ogg",
                                    "mp4", "mkv", "mov", "avi", "webm"],
                              key="uploader")
        if up and st.button("Process", type="primary", key="process_btn"):
            tmp = Path(tempfile.mkdtemp()) / up.name
            tmp.write_bytes(up.getbuffer())
            with st.spinner("Processing..."):
                out_path = process_file(tmp)
            st.success("Done!")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Original")
                st.audio(str(tmp))
            with c2:
                st.caption("Cleaned")
                st.audio(str(out_path))
            st.download_button("⬇️ Download cleaned audio",
                               data=Path(out_path).read_bytes(),
                               file_name="cleaned.wav", mime="audio/wav",
                               key="download_btn")

    # ---- Live ------------------------------------------------------------
    with tab_rt:
        st.caption("Click START, allow mic. Speak / chew near the mic — "
                   "triggers get ducked and the audio comes back to you.")
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            import av
            import tensorflow as tf

            yamnet, class_names = load_yamnet()
            trig_idx = [i for i, n in enumerate(class_names) if n in TRIGGERS] or [0]

            class Reducer:
                def __init__(self):
                    self.buf = np.zeros(SR_Y, dtype=np.float32)
                    self.gain = 1.0

                def recv(self, frame):
                    pcm = frame.to_ndarray()
                    sr = frame.sample_rate
                    mono = pcm.astype(np.float32)
                    if mono.ndim > 1:
                        mono = mono.mean(axis=0)
                    if np.max(np.abs(mono)) > 1.5:
                        mono = mono / 32768.0
                    g = np.gcd(SR_Y, sr)
                    m16 = resample_poly(mono, SR_Y // g, sr // g).astype(np.float32)
                    k = len(m16)
                    if k >= len(self.buf):
                        self.buf = m16[-len(self.buf):]
                    else:
                        self.buf = np.concatenate([self.buf[k:], m16])
                    scores, _, _ = yamnet(tf.constant(self.buf))
                    prob = float(scores.numpy()[:, trig_idx].max())
                    target = 10 ** (-MAX_REDUCTION_DB / 20) if prob >= THRESH_LO else 1.0
                    self.gain += np.clip(target - self.gain, -0.3, 0.3)
                    out = (pcm.astype(np.float32) * self.gain).astype(pcm.dtype)
                    new = av.AudioFrame.from_ndarray(out, format=frame.format.name,
                                                     layout=frame.layout.name)
                    new.sample_rate = sr
                    return new

            webrtc_streamer(
                key="shush-rt",
                mode=WebRtcMode.SENDRECV,
                audio_frame_callback=Reducer().recv,
                media_stream_constraints={"audio": True, "video": False},
            )
        except ModuleNotFoundError:
            st.warning("Live needs extra packages:\n\n"
                       "`python -m pip install streamlit-webrtc av`")


# Guard: only run the UI when executed as the main Streamlit script.
# If this file is wrongly named python.py, TensorFlow may import it as module
# 'python'; that import has __name__ == 'python' (not '__main__'), so the UI
# is skipped and the app does NOT render twice.
if __name__ == "__main__":
    run_app()