# 🧪 Ego2Dex Colab Testing Guide (Florence-2 Pipeline)

This branch contains the precise code modifications and configurations we developed to run the **Florence-2 + Grounding DINO + SAM 2** object detection pipeline seamlessly on Google Colab.

## 📁 Files Modified in this Branch
You are pushing exactly 3 files to your repository in this branch:
1. **`src/ego2dex/stages/caption/florence2.py`**
   - **Why:** We completely rewrote this file to fix the PyTorch `c10::Half` crash on Colab T4 GPUs. We also programmed it to run the `<OD>` (Object Detection) task so it automatically extracts tags and feeds them directly into Grounding DINO.
2. **`configs/pipeline/colab.yaml`**
   - **Why:** This is our master configuration file. It is explicitly configured to use **Florence-2** for auto-tagging, removes the `max_frames` limit so you can process full-length videos, and sets `sample_fps: 5` to ensure the Colab server doesn't crash during long 10-hour runs.
3. **`TESTING_README.md`**
   - **Why:** This very document, which acts as a quick-start guide for your company.

---

## 🚀 How to Run the Pipeline in Colab
If anyone at your company wants to test the pipeline with our successful Florence-2 configuration, all they have to do is open a blank Google Colab Notebook and run this exact cell:

```python
import os

# ==========================================
# STEP 1: WIPE AND CLONE THE CORRECT BRANCH
# ==========================================
%cd /content
!rm -rf /content/ego2dex

# Notice we are cloning your new 'colab-testing' branch!
!git clone -b colab-testing https://github.com/adityagarg7/ego2dex.git
%cd /content/ego2dex

# ==========================================
# STEP 2: INSTALL DEPENDENCIES
# ==========================================
!pip install .[dev] -q
# Crucial: Downgrade transformers to 4.49.0 to prevent Florence-2 errors
!pip install transformers==4.49.0 accelerate qwen-vl-utils einops -q
!pip install mediapipe -q
!pip install git+https://github.com/facebookresearch/sam2.git -q

# ==========================================
# STEP 3: DOWNLOAD CORE WEIGHTS
# ==========================================
!mkdir -p /content/_DATA
!wget -q -nc -O /content/_DATA/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
!wget -q -nc -O /content/_DATA/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# ==========================================
# STEP 4: GET YOUR VIDEO FROM DRIVE
# ==========================================
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p my_videos
# Assuming your video is named testing.mp4
!cp "/content/drive/MyDrive/testing.mp4" my_videos/testing.mp4

# ==========================================
# STEP 5: RUN THE PIPELINE!
# ==========================================
print("⏳ Starting the Florence-2 pipeline. You should see a progress bar below...")
# Because the colab.yaml is already saved in this branch, we just point to it!
!ego2dex run --config configs/pipeline/colab.yaml --input my_videos/testing.mp4 --output /content/outputs/florence_run
```

## ⚙️ How to Change the Configuration Later
If you want to test **Manual Classes** instead of Florence-2 to maximize speed on your local RTX 5090 rig, simply open `configs/pipeline/colab.yaml` and make these two changes:
1. **Delete** the Florence-2 block:
   ```yaml
   - family: caption
     name: florence2
     enabled: true
   ```
2. **Add your manual classes** to Grounding DINO:
   ```yaml
   - family: detection
     name: grounding_dino
     enabled: true
     params:
       classes: ["plate", "knife", "pan", "bowl"]
   ```
