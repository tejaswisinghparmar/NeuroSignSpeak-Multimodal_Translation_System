# NeuroSignSpeak — Multimodal Translation System

A real-time multimodal assistive communication system that fuses **facial emotion recognition**, **EEG-based motor imagery classification**, and **ASL finger-spelling recognition** into translated speech output. Built with Python, the project ships two standalone GUIs: a lightweight Emotion + ASL app and a full NeuroSignSpeak dashboard with EEG integration and weighted fusion.

---

## Features

### Application 1 — Emotion & ASL Recognition (`python main.py`)
| Feature | Description |
|---|---|
| **Real-time Emotion Detection** | Webcam-based facial emotion recognition via DeepFace — detects Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral with per-emotion confidence bars |
| **ASL Finger-Spelling** | MediaPipe Hand Landmarker tracks 21 hand landmarks; a rule-based classifier maps finger positions to ASL letters (A, B, C, D, I, L, V, W, Y, 5) |
| **Stability Filter** | Letters are accepted only after 12 consecutive matching frames, reducing false positives |
| **Live ASL Buffer** | Recognised letters accumulate in an on-screen text buffer |
| **Ollama LLM Correction** | Sends the raw ASL buffer to a local Ollama model (default: `gemma3:1b`) for spelling/grammar correction |
| **Dark-Themed GUI** | CustomTkinter interface with purple accent palette, sidebar controls, and mirrored video feed |

### Application 2 — NeuroSignSpeak Dashboard (`python run_dashboard.py`)
| Feature | Description |
|---|---|
| **Thread A — Webcam + Emotion** | Real-time DeepFace emotion detection with bounding boxes and confidence overlays |
| **Thread B — EEG Stream Simulator** | Replays PhysioNet EEGBCI motor imagery data (64-channel, 160 Hz) through a queue as a simulated live EEG stream |
| **Thread C — EEG DSP + Classifier** | Applies a full DSP pipeline (FIR High-Pass 0.5 Hz → Notch 50 Hz → Band-Pass 8–30 Hz) then classifies motor imagery (Left Fist vs Right Fist) using CSP + SVM |
| **Weighted Fusion Engine** | Combines emotion context (tone/affect) with EEG motor intent (action) via adjustable slider weights to produce a translated speech sentence |
| **Real-time EEG Graph** | Scrolling Mu (8–12 Hz) and Beta (13–30 Hz) band power plot rendered on a pure Canvas (no matplotlib dependency) |
| **Thread Status Panel** | Live indicators for Webcam, EEG Simulator, EEG Processor, and Fusion threads |
| **Translated Speech Output** | Fused result displayed as a human-readable sentence (e.g., *"I'm feeling great! I want to go left, happily."*) |

---

## Performance Metrics

All scores were computed on the project's actual pipelines (see `benchmark.py` to reproduce).

### EEG Motor Imagery Classification — CSP + SVM

| Metric | Value |
|---|---|
| **Dataset** | PhysioNet EEGBCI — Subject 1, Runs 4/8/12 |
| **Task** | Left Fist vs Right Fist motor imagery (2-class) |
| **Channels** | 64 EEG channels |
| **Epochs** | 45 |
| **DSP Pipeline** | FIR HP 0.5 Hz → Notch 50 Hz → BP 8–30 Hz |
| **Classifier** | CSP (4 components, Ledoit-Wolf) → SVM (RBF, C=1.0) |
| **Accuracy (5-fold CV)** | **75.56% ± 8.31%** |
| **F1 Score (weighted)** | **75.10% ± 8.35%** |
| **Precision (weighted)** | **79.22%** |
| **Recall (weighted)** | **75.56%** |
| **Inference Latency** | **0.74 ms** avg (P95: 1.07 ms) |

### Facial Emotion Detection — DeepFace

| Metric | Value |
|---|---|
| **Model** | DeepFace (VGG-Face backbone) |
| **Face Detector** | OpenCV Haar Cascade |
| **Classes** | Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral (7) |
| **Published Accuracy (FER-2013)** | **57.44%** |
| **Inference Latency** | **60.33 ms** avg (P95: 64.09 ms) |
| **Detection Rate** | **100%** (on synthetic test frames) |

> *FER-2013 accuracy is the standard published benchmark for the default DeepFace emotion model. Real-world performance improves with good lighting and frontal face positioning.*

### ASL Finger-Spelling — MediaPipe + Rule-Based Classifier

| Metric | Value |
|---|---|
| **Hand Detection Model** | MediaPipe Hand Landmarker (float16) |
| **Classifier** | Rule-based heuristic (finger tip vs PIP joint geometry) |
| **Supported Letters** | A, B, C, D, I, L, V, W, Y, 5 (10 signs) |
| **Stability Threshold** | 12 consecutive matching frames |
| **Min Detection Confidence** | 0.70 |
| **Min Tracking Confidence** | 0.60 |

> *Rule-based accuracy is high for supported letters with clear hand orientation. Non-supported letters return no prediction (fail-safe).*

### System-Level Performance

| Metric | Value |
|---|---|
| **Video Feed** | ~30 FPS (threaded, non-blocking GUI) |
| **EEG Chunk Rate** | 1 chunk/sec (1 s window, 160 samples) |
| **Fusion Update Rate** | 5 Hz (200 ms tick) |
| **GUI Framework** | CustomTkinter (dark theme) |
| **Hardware** | CPU-only (GPU optional for TensorFlow) |

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Emotion Detection** | DeepFace, OpenCV (Haar Cascade), TensorFlow/Keras |
| **ASL Recognition** | MediaPipe Hand Landmarker (Tasks API) |
| **EEG Processing** | MNE-Python, SciPy (Welch PSD), scikit-learn (CSP + SVM) |
| **EEG Data** | PhysioNet EEGBCI Motor Imagery dataset (64-ch, 160 Hz) |
| **Fusion** | Custom weighted-decision engine |
| **LLM Integration** | Ollama (default model: `gemma3:1b`) |
| **GUI** | CustomTkinter |
| **Image Processing** | Pillow, NumPy |

---

## Requirements

- **Python 3.8 – 3.10** (required for TensorFlow compatibility)
- Webcam for real-time video input
- ~500 MB+ disk space for ML models + PhysioNet data
- Ollama installed and running (for LLM features in ASL mode)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/tejaswisinghparmar/facial-expression-recognition.git
cd facial-expression-recognition
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Hand Landmarker Model
The ASL module requires the MediaPipe Hand Landmarker model file:
```bash
wget -O hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
Place `hand_landmarker.task` in the project root.

> [Model download link](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) · [MediaPipe Hand Landmarker docs](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

### 5. Create MNE Data Directory
The EEG modules download PhysioNet data on first run:
```bash
mkdir -p ~/mne_data
```

### 6. Install & Start Ollama (optional)
For LLM-powered ASL text correction, install [Ollama](https://ollama.ai):
```bash
ollama serve            # start the service
ollama pull gemma3:1b   # download the default model
```

---

## Usage

### Emotion + ASL App
```bash
python main.py
```
- Toggle **Emotion Mode** or **ASL Mode** from the sidebar
- In ASL mode, spell letters with your hand; press **Send to Ollama** for grammar correction
- Press **Add Space** between words, **Clear Buffer** to reset

### NeuroSignSpeak Dashboard
```bash
python run_dashboard.py                # default — PhysioNet subject 1
python run_dashboard.py --subject 3    # use subject 3
python run_dashboard.py --log-level DEBUG
```
- Click **▶ Start All** to launch all threads (webcam, EEG simulator, EEG processor)
- Adjust **Emotion weight** / **EEG weight** sliders to tune the fusion
- Watch the translated speech update in real time

---

## Project Structure

```
facial-expression-recognition/
├── main.py                      # Entry point — Emotion + ASL GUI
├── app.py                       # CustomTkinter GUI (Emotion & ASL modes)
├── run_dashboard.py             # Entry point — NeuroSignSpeak Dashboard
├── dashboard.py                 # Multimodal dashboard (Emotion + EEG + Fusion)
├── benchmark.py                 # Performance benchmark script
├── requirements.txt             # Python dependencies
├── hand_landmarker.task         # MediaPipe hand model (download separately)
│
└── modules/
    ├── __init__.py
    ├── emotion_detector.py      # DeepFace emotion detection (7 classes)
    ├── asl_recognizer.py        # MediaPipe hand landmarks → ASL letters
    ├── eeg_stream_simulator.py  # PhysioNet EEGBCI replay (Thread B)
    ├── eeg_processor.py         # MNE DSP → CSP → SVM classifier (Thread C)
    ├── fusion.py                # Weighted decision engine (emotion + EEG)
    └── ollama_client.py         # Ollama LLM integration for text correction
```

---

## Module Details

### `emotion_detector.py`
- Analyses each webcam frame with `DeepFace.analyze()` (action: emotion)
- Draws colour-coded bounding boxes and per-emotion confidence bar charts
- Returns dominant emotion + full score dict for downstream fusion

### `asl_recognizer.py`
- Detects 21 hand landmarks via MediaPipe Hand Landmarker (IMAGE mode)
- Rule-based classifier checks finger tip-above-PIP geometry for each finger
- Stability filter (12 frames) prevents jittery predictions
- Draws hand skeleton and accumulated letter buffer on the video feed

### `eeg_stream_simulator.py`
- Downloads PhysioNet EEGBCI data (64-ch, 160 Hz, motor imagery runs)
- Streams 1-second chunks through a `queue.Queue` at real-time pace
- Attaches ground-truth event labels (rest / left_fist / right_fist) to each chunk
- Loops continuously; drop-in replaceable with OpenBCI / LSL hardware

### `eeg_processor.py`
- **Offline training**: loads PhysioNet data → DSP → epochs → CSP(4) + SVM(RBF)
- **Online inference**: consumes chunks from the queue, applies the same DSP, classifies
- Maintains rolling Mu/Beta band power history for the dashboard graph
- Reports `EEGResult` dataclass with label, confidence, and band powers

### `fusion.py`
- Maps emotions to tone/context (*"happily"*, *"anxiously"*) and EEG labels to intent phrases (*"I want to go left"*)
- Combines both via adjustable weights (default 50/50) into a composite translated speech string
- Normalises weights automatically; outputs `FusionResult` with full breakdown

### `ollama_client.py`
- Sends raw ASL letter buffer to a local Ollama model with a spelling/grammar correction prompt
- Runs inference in a background thread with a callback to avoid blocking the GUI
- Default model: `gemma3:1b` (configurable)

### `dashboard.py`
- Orchestrates three concurrent threads (Webcam, EEG Simulator, EEG Processor)
- 200 ms GUI tick updates EEG graph, fusion result, and thread status indicators
- Pure Canvas EEG plot with scrolling Mu/Beta traces and axis labels

---

## Troubleshooting

### EEG Data Download Fails
```bash
# Create the MNE data directory
mkdir -p ~/mne_data

# Or set a custom path
export MNE_DATA=/path/to/your/data
```

### Models Not Loading
```bash
pip install -r requirements.txt --upgrade
ls hand_landmarker.task   # must exist in project root
```

### Webcam Issues
- Check that no other application is using the camera
- Verify camera permissions on your OS
- The app retries camera open up to 5 times automatically

### Ollama Connection Error
```bash
ollama serve              # ensure the service is running
curl http://localhost:11434/api/tags   # verify connectivity
ollama pull gemma3:1b     # ensure the model is downloaded
```

### CUDA / GPU Warnings
TensorFlow may print CUDA warnings on CPU-only machines — these are **harmless**. All inference runs on CPU by default.

---

## Reproducing Benchmarks

```bash
source venv/bin/activate
python benchmark.py
# Results are saved to benchmark_results.json
```

---

## Future Enhancements

- [ ] Support for dynamic ASL signs (motion-based letters like J, Z)
- [ ] Expand rule-based classifier to all 26 ASL letters
- [ ] Replace EEG simulator with live OpenBCI / LSL hardware stream
- [ ] Add text-to-speech (TTS) output for translated speech
- [ ] Multi-subject EEG model generalisation
- [ ] Emotion-aware response generation via LLM
- [ ] Video recording with overlay annotations
- [ ] Export session data (emotions, EEG, fusion) to CSV

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue in the repository.

---

**Built with Python, OpenCV, MediaPipe, DeepFace, MNE-Python & scikit-learn**
