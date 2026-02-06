# Virtual Try-On System v7.7

## Final Project Report & Technical Documentation

---

## 1. Project Overview

**Project Title:** Virtual Try-On System v7.7  
**Type:** Inference-only AI system  
**Domain:** Computer Vision, Generative AI, Virtual Try-On  
**Execution Environment:** CUDA-enabled GPU  
**Training:** ❌ None  
**Fine-tuning:** ❌ None  
**Datasets:** ❌ None

This project implements a **production-grade virtual try-on system** that allows a user to virtually try **shirts or pants independently** on a person image using **only pretrained models at inference time**.

The system is designed to preserve:
- Person identity (especially face)
- Body pose and geometry
- Arms and occlusion correctness
- Background integrity

While accurately transferring:
- Garment color
- Texture
- Fabric patterns

---

## 2. Design Constraints & Motivation

### 2.1 Constraints

- Zero budget
- No custom training
- No LoRA / fine-tuning
- Only pretrained models
- Deterministic, debuggable pipeline

### 2.2 Motivation

Most virtual try-on systems fail due to:
- Face distortion
- Garment bleeding into arms
- Incorrect pose alignment
- Full-body replacement instead of partial editing

This project solves these issues **purely through architecture and masking logic**, not training.

---

## 3. High-Level Architecture

### 3.1 Input

1. **Person Image**
   - Full-body image
   - Source of identity, pose, and geometry

2. **Garment Image**
   - Shirt OR pants image
   - Appearance reference only

3. **Garment Type Selector**
   - `shirt` or `pants`
   - Enforces mutual exclusivity

---

### 3.2 Core Components

| Component | Purpose |
|--------|--------|
| Human Parsing | Pixel-accurate garment segmentation |
| ControlNet | Pose & body geometry preservation |
| IP-Adapter | Garment appearance transfer |
| Diffusion Inpainting | Localized garment replacement |
| Face Hard Compositing | Identity preservation |

---

## 4. Human Parsing Module

### 4.1 Model Used

- SegFormer (clothes-parsing checkpoint)
- Pretrained on human-clothing datasets

### 4.2 Why SegFormer

- Lightweight
- Fast inference
- Produces semantic segmentation maps
- No training required

### 4.3 Verified Label Mapping

| Label ID | Meaning |
|-------|--------|
| 4 | Upper clothes (shirt) |
| 6 | Pants |
| 9, 10 | Arms |
| 11 | Face |
| 2 | Hair |

### 4.4 Garment Masks

- Shirt mask → label `4`
- Pants mask → label `6`
- Masks resized to input resolution using nearest-neighbor

---

## 5. ControlNet (Pose & Geometry)

### 5.1 Purpose

- Preserve:
  - Arm position
  - Shoulder alignment
  - Leg pose
  - Torso structure

### 5.2 Why ControlNet

- Prevents pose hallucination
- Keeps garment aligned to body
- No geometry learned from garment image

### 5.3 Input

- Pose extracted from person image
- Full-body conditioning

---

## 6. IP-Adapter (Appearance Transfer)

### 6.1 Purpose

- Inject garment appearance only
- Transfer:
  - Color
  - Texture
  - Fabric pattern

### 6.2 What IP-Adapter Does NOT Do

- Does NOT change pose
- Does NOT change body shape
- Does NOT override ControlNet

### 6.3 Configuration

- Single IP-Adapter instance
- Used for both shirt and pants
- Scale tuned for appearance dominance without structure override

---

## 7. Diffusion Inpainting Pipeline

### 7.1 Model

- Stable Diffusion Inpainting (SD 1.5)

### 7.2 Why Inpainting

- Allows **partial image modification**
- Non-masked regions remain unchanged
- Ideal for garment replacement

### 7.3 Inputs

- Original person image
- Garment-specific mask
- ControlNet pose conditioning
- IP-Adapter garment image

### 7.4 Key Parameters

- `strength` < 0.7 to avoid latent drift
- Guidance scale tuned for realism

---

## 8. v7.6: Dual-Garment Support

### 8.1 Motivation

Users should be able to:
- Change shirt only
- OR change pants only

But never both in the same run.

### 8.2 Solution

- Garment type selector (`shirt` / `pants`)
- Mutually exclusive masks
- Single diffusion pass

### 8.3 Benefits

- Predictable output
- No garment interference
- Simplified logic

---

## 9. v7.7: Sleeve–Arm Occlusion

### 9.1 Problem

Sleeves often overwrite arms due to incorrect mask ordering.

### 9.2 Solution

- Extract arm masks (labels 9, 10)
- Dilate arm regions
- Subtract arms from shirt mask

### 9.3 Result

- Arms always appear above sleeves
- Natural occlusion
- No repainting of limbs

---

## 10. Face Distortion Problem

### 10.1 Root Cause

Even when face is not masked, diffusion can:
- Alter face latents
- Cause identity drift

Masking alone is insufficient.

---

## 11. v7.7.1: Face Hard Compositing (Final Fix)

### 11.1 Core Idea

> Never trust diffusion with identity-critical regions.

### 11.2 Steps

1. Extract face from original image using segmentation
2. Create a soft face mask
3. Run diffusion normally
4. Paste original face back onto generated image

### 11.3 Why This Works

- Deterministic
- No training
- No face-swap model
- Zero identity drift

---

## 12. Code Architecture

### 12.1 File Structure

```
vton_v7_7/
├── app.py                 # Flask backend
├── pipeline.py            # Core try-on pipeline
├── human_parsing.py       # Segmentation & masks
├── face_utils.py          # Face extraction & blending
├── controlnet_utils.py    # Pose extraction
├── config.py              # Configuration
├── templates/index.html   # Frontend UI
├── static/results/        # Output images
├── uploads/               # Uploaded inputs
└── requirements.txt
```

---

## 13. Code File Responsibilities

### 13.1 `pipeline.py`

- Loads all pretrained models
- Selects correct garment mask
- Runs diffusion
- Merges original face back
- Saves debug outputs

### 13.2 `human_parsing.py`

- Performs semantic segmentation
- Generates shirt / pants masks
- Handles occlusion logic

### 13.3 `face_utils.py`

- Extracts face mask
- Creates soft blending mask
- Used for final face merge

### 13.4 `controlnet_utils.py`

- Extracts pose keypoints
- Provides ControlNet conditioning image

### 13.5 `app.py`

- Flask backend
- Handles file uploads
- Calls pipeline
- Returns result URL to frontend

---

## 14. Frontend Overview

- HTML/CSS/JavaScript UI
- Image upload with preview
- Garment type selector
- POST request to `/tryon`
- Displays generated image

No AI logic in frontend.

---

## 15. Key Achievements

- ✅ No training
- ✅ No datasets
- ✅ Face preserved
- ✅ Accurate garment transfer
- ✅ Correct occlusion
- ✅ Modular, debuggable design

---

## 16. Limitations

- Lighting mismatch may cause slight color variance
- Only one garment per run
- SD 1.5 resolution limits

---

## 17. Conclusion

This project demonstrates that **high-quality virtual try-on is achievable without training** by carefully combining:

- Semantic segmentation
- Pose conditioning
- Appearance adapters
- Inpainting
- Deterministic compositing

The final v7.7 system is **robust, explainable, and production-ready**, making it suitable for academic submission, demos, and real-world experimentation.

---

## 18. Future Scope

- SDXL upgrade
- Multi-garment layering
- Video try-on
- Performance optimization

---

**End of Report**

