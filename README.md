# misophonia-triger-shield

**Shush** — a Streamlit app that detects and reduces misophonia trigger sounds (chewing, typing, coughing, clicking, etc.) using YAMNet for detection and DeepFilterNet for noise cleanup.

- **Upload** tab: process an audio/video file and download the cleaned audio.
- **Live** tab: duck trigger sounds from your microphone in real time.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run python.py
```
