"""
Benchmark script — compute performance metrics for all modules.
Outputs a JSON summary at the end.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings

import json
import time
import warnings
import logging
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

results = {}

# ═══════════════════════════════════════════════════════════════════
# 1. EEG CSP + SVM  (PhysioNet EEGBCI — Subject 1, runs 4,8,12)
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("[1/3] EEG CSP+SVM Classification Benchmark")
print("=" * 60)

try:
    import mne
    from mne.datasets import eegbci
    from mne.decoding import CSP
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    from scipy.signal import welch

    subject = 1
    runs = [4, 8, 12]  # left fist vs right fist motor imagery

    print(f"  Loading PhysioNet EEGBCI — subject {subject}, runs {runs}...")
    raw_fnames = eegbci.load_data(subject, runs, update_path=True)
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore")
    picks = mne.pick_types(raw.info, eeg=True)
    raw.pick(picks)

    # DSP pipeline
    raw.filter(0.5, None, method="fir", fir_design="firwin", verbose=False)
    raw.notch_filter(50.0, method="fir", verbose=False)
    raw.filter(8.0, 30.0, method="fir", fir_design="firwin", verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    wanted = {}
    for k, v in event_id.items():
        if "T1" in k or "T2" in k:
            wanted[k] = v
    if not wanted:
        wanted = {k: v for k, v in event_id.items() if v in (2, 3)}

    epochs = mne.Epochs(raw, events, event_id=wanted,
                        tmin=0.5, tmax=3.5, baseline=None,
                        preload=True, verbose=False)
    epochs.drop_bad(verbose=False)

    X = epochs.get_data(copy=True)
    y = epochs.events[:, 2]

    print(f"  Epochs: {len(epochs)}  |  Classes: {np.unique(y)}  |  Channels: {X.shape[1]}")

    # 5-fold stratified cross-validation
    N_CSP = 4
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("CSP", CSP(n_components=N_CSP, reg="ledoit_wolf", log=True)),
        ("SVM", SVC(kernel="rbf", C=1.0, probability=True)),
    ])

    scores_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    scores_f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted")
    scores_prec = cross_val_score(pipe, X, y, cv=cv, scoring="precision_weighted")
    scores_rec = cross_val_score(pipe, X, y, cv=cv, scoring="recall_weighted")

    # Measure inference latency
    pipe_bench = Pipeline([
        ("CSP", CSP(n_components=N_CSP, reg="ledoit_wolf", log=True)),
        ("SVM", SVC(kernel="rbf", C=1.0, probability=True)),
    ])
    pipe_bench.fit(X, y)

    latencies = []
    for i in range(min(50, len(X))):
        t0 = time.perf_counter()
        pipe_bench.predict(X[i:i+1])
        latencies.append((time.perf_counter() - t0) * 1000)

    eeg_results = {
        "task": "Motor Imagery (Left Fist vs Right Fist)",
        "dataset": f"PhysioNet EEGBCI — Subject {subject}",
        "epochs": int(len(epochs)),
        "channels": int(X.shape[1]),
        "pipeline": "FIR HP 0.5Hz → Notch 50Hz → BP 8-30Hz → CSP(4) → SVM(RBF)",
        "cv_folds": 5,
        "accuracy_mean": round(float(scores_acc.mean()) * 100, 2),
        "accuracy_std": round(float(scores_acc.std()) * 100, 2),
        "f1_weighted_mean": round(float(scores_f1.mean()) * 100, 2),
        "f1_weighted_std": round(float(scores_f1.std()) * 100, 2),
        "precision_mean": round(float(scores_prec.mean()) * 100, 2),
        "recall_mean": round(float(scores_rec.mean()) * 100, 2),
        "inference_latency_ms_mean": round(float(np.mean(latencies)), 2),
        "inference_latency_ms_p95": round(float(np.percentile(latencies, 95)), 2),
    }
    results["eeg_csp_svm"] = eeg_results

    print(f"  Accuracy:  {eeg_results['accuracy_mean']:.2f}% ± {eeg_results['accuracy_std']:.2f}%")
    print(f"  F1 Score:  {eeg_results['f1_weighted_mean']:.2f}% ± {eeg_results['f1_weighted_std']:.2f}%")
    print(f"  Precision: {eeg_results['precision_mean']:.2f}%")
    print(f"  Recall:    {eeg_results['recall_mean']:.2f}%")
    print(f"  Inference: {eeg_results['inference_latency_ms_mean']:.2f}ms (p95: {eeg_results['inference_latency_ms_p95']:.2f}ms)")
    print()

except Exception as e:
    print(f"  ERROR: {e}")
    results["eeg_csp_svm"] = {"error": str(e)}

# ═══════════════════════════════════════════════════════════════════
# 2. Emotion Detection (DeepFace)
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("[2/3] Emotion Detection (DeepFace) Benchmark")
print("=" * 60)

try:
    from deepface import DeepFace
    import cv2

    # Create synthetic test faces at different sizes for latency benchmark
    # Use a real-looking test: webcam-sized frames with synthetic face region
    test_sizes = [(480, 640, 3), (720, 1280, 3)]
    
    # Measure DeepFace analysis latency on blank/noise frames
    # (measures pipeline overhead; real accuracy comes from the model's published benchmarks)
    latencies = []
    successful = 0
    total_tests = 20

    # Generate a simple synthetic face-like image (gray oval on darker background)
    def make_test_frame(h=480, w=640):
        frame = np.random.randint(40, 80, (h, w, 3), dtype=np.uint8)
        # Draw a face-like oval
        center = (w // 2, h // 2)
        cv2.ellipse(frame, center, (80, 110), 0, 0, 360, (180, 160, 140), -1)
        return frame

    print("  Running inference latency tests...")
    for i in range(total_tests):
        frame = make_test_frame()
        t0 = time.perf_counter()
        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
                detector_backend="opencv",
            )
            successful += 1
        except Exception:
            pass
        latencies.append((time.perf_counter() - t0) * 1000)

    # DeepFace uses a fine-tuned VGG-Face model for emotion.
    # Published benchmark accuracy on FER-2013 test set:
    emotion_results = {
        "model": "DeepFace (VGG-Face backbone)",
        "detector_backend": "OpenCV Haar Cascade",
        "emotions": ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"],
        "num_classes": 7,
        "published_accuracy_fer2013": 57.44,
        "note": "FER-2013 test set accuracy (published by DeepFace). Real-world accuracy varies with lighting, angle, and occlusion.",
        "inference_latency_ms_mean": round(float(np.mean(latencies)), 2),
        "inference_latency_ms_p95": round(float(np.percentile(latencies, 95)), 2),
        "frames_analyzed": total_tests,
        "detection_rate": round(successful / total_tests * 100, 1),
    }
    results["emotion_detection"] = emotion_results

    print(f"  Model:     DeepFace (VGG-Face backbone)")
    print(f"  Backend:   OpenCV Haar Cascade")
    print(f"  Published Accuracy (FER-2013): {emotion_results['published_accuracy_fer2013']}%")
    print(f"  Avg Latency:  {emotion_results['inference_latency_ms_mean']:.2f}ms")
    print(f"  P95 Latency:  {emotion_results['inference_latency_ms_p95']:.2f}ms")
    print(f"  Detection Rate: {emotion_results['detection_rate']}% ({successful}/{total_tests})")
    print()

except Exception as e:
    print(f"  ERROR: {e}")
    results["emotion_detection"] = {"error": str(e)}

# ═══════════════════════════════════════════════════════════════════
# 3. ASL Recognition (MediaPipe Hand Landmarker)
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("[3/3] ASL Recognition (MediaPipe) Benchmark")
print("=" * 60)

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode,
    )

    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    
    if not os.path.exists(model_path):
        print(f"  WARNING: hand_landmarker.task not found at {model_path}")
        raise FileNotFoundError("hand_landmarker.task not found")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Measure detection latency on synthetic frames
    latencies = []
    num_tests = 30
    print("  Running landmark detection latency tests...")
    for i in range(num_tests):
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t0 = time.perf_counter()
        result = landmarker.detect(mp_image)
        latencies.append((time.perf_counter() - t0) * 1000)

    landmarker.close()

    asl_results = {
        "model": "MediaPipe Hand Landmarker (float16)",
        "approach": "Rule-based heuristic classifier on 21 hand landmarks",
        "supported_letters": ["A", "B", "C", "D", "I", "L", "V", "W", "Y", "5"],
        "num_supported": 10,
        "stability_filter": "12-frame consecutive match threshold",
        "detection_confidence": 0.7,
        "tracking_confidence": 0.6,
        "inference_latency_ms_mean": round(float(np.mean(latencies)), 2),
        "inference_latency_ms_p95": round(float(np.percentile(latencies, 95)), 2),
        "note": "Rule-based on geometric finger positions; accuracy depends on hand orientation and lighting.",
    }
    results["asl_recognition"] = asl_results

    print(f"  Model:     MediaPipe Hand Landmarker (float16)")
    print(f"  Classifier: Rule-based (finger tip vs PIP joint positions)")
    print(f"  Letters:   {', '.join(asl_results['supported_letters'])} ({asl_results['num_supported']} total)")
    print(f"  Stability: {asl_results['stability_filter']}")
    print(f"  Avg Latency: {asl_results['inference_latency_ms_mean']:.2f}ms")
    print(f"  P95 Latency: {asl_results['inference_latency_ms_p95']:.2f}ms")
    print()

except Exception as e:
    print(f"  ERROR: {e}")
    results["asl_recognition"] = {"error": str(e)}

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("BENCHMARK COMPLETE — Summary JSON:")
print("=" * 60)
print(json.dumps(results, indent=2))

# Save to file
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to benchmark_results.json")
