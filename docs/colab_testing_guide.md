# Testing the `ego2dex` Pipeline

This document provides a crisp, step-by-step summary of how to test the `ego2dex` pipeline on an egocentric video.

## 🛠 Software & Tools Used
- **Platform:** Google Colab (Free Tier, T4 GPU)
- **Codebase:** `ego2dex` Python repository
- **AI Models:**
  - **MediaPipe:** Hand tracking (used as a fully-permissive alternative to HaMeR).
  - **Grounding DINO:** Zero-shot open-vocabulary object detection.
  - **SAM 2:** Pixel-perfect object segmentation.

## 📥 Input & 📤 Output
- **Input:** Raw egocentric (first-person) video file (e.g., `.mp4` from the EPIC-Kitchens dataset).
- **Outputs:**
  - **JSON (`.json`):** Raw structured mathematical coordinates of hand joints, bounding boxes, and object IDs.
  - **COCO Dataset:** Standardized computer vision annotation format for downstream training.
  - **Visualizations (`.jpg` / `.png`):** Video frames with 3D hand skeletons, bounding boxes, and masks drawn directly onto the image.

---

## 🚀 Step-by-Step Testing Process

### 1. Environment Setup
We initialized a Colab notebook with a GPU and installed the required software.
- Cloned the `ego2dex` repository.
- Installed the core dependencies via `pip install -e ".[dev]"`.
- Explicitly downloaded the heavy model weights (`hand_landmarker.task` for MediaPipe and `sam2.1_hiera_large.pt` for SAM 2) to a local `_DATA` directory to prevent silent loading errors.

### 2. Configuration (`colab.yaml`)
We created a custom YAML configuration file to dictate how the AI should process the video. 
- **Data Limits:** Capped the run to 150 frames (30 seconds) at 5 FPS to fit within Colab's memory constraints.
- **Bypassing Bugs:** To bypass parameter-swallowing bugs in the combo stage, we split the object detection into two explicit stages (`grounding_dino` followed by `sam2`).
- **Prompting:** Set explicit text prompts (`["plate", "pan", "knife", "cup", "bowl", "hand"]`) for Grounding DINO to look for.

### 3. Execution
We launched the processing job via the Command Line Interface (CLI):
```bash
ego2dex run --config configs/pipeline/colab.yaml --input my_videos/test_video.mp4 --output outputs/full_run
```

---

## 📊 Results & Accuracy Analysis

**Summary of 30-Second Clip (150 frames at 5 FPS):**
- **Hands Tracked:** 49 frames.
- **Objects Detected:** 357 instances.

**Issue: Misclassifications (e.g., "Sink" Hallucinations)**
Upon reviewing the visualization frames, some objects were incorrectly labeled (e.g., a shadow or counter edge marked as a "sink"). 
- **Why it happens:** Grounding DINO is a zero-shot detector. It attempts to mathematically match the text prompts to any visual features. If the `box_threshold` (confidence score) is too low, it becomes overly aggressive and guesses incorrectly.
- **The Fix:**
  1. **Algorithmic:** Increase `box_threshold` (e.g., `0.3` → `0.5`) in `colab.yaml` to force strict confidence. Remove irrelevant words from the `classes` list.
  2. **Manual (Human-in-the-Loop):** Import the exported `coco.json` into annotation software (like CVAT or Label Studio) to manually delete or correct the bounding boxes. *(Highly recommended to include screenshots of these tools here for documentation).*

**Calculating Efficiency (Metric Proposal):**
To measure the pipeline's performance before training, efficiency is calculated as:
`Efficiency = (True Positives) / (True Positives + False Positives + False Negatives)`
*Goal:* Achieve >95% precision via threshold tuning, bridging the final gap to 100% through manual cleanup in CVAT.
