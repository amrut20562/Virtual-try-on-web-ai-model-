# Virtual Try-On System v7.7 👕👖

An inference-only AI-based virtual try-on system that realistically replaces upper-body (shirts) or lower-body (pants) garments on a person image using pretrained diffusion models.

This project works with **zero training**, **no custom datasets**, and **no fine-tuning**, using only pretrained models at inference time.

---

## ✨ Key Features

- ✅ Shirt-only or pants-only virtual try-on
- ✅ Person identity and face preservation
- ✅ ControlNet-based pose and structure control
- ✅ IP-Adapter-based garment appearance transfer
- ✅ Sleeve–arm occlusion handling (v7.7)
- ✅ Hard face identity preservation via face compositing
- ✅ No training, no fine-tuning, no datasets
- ✅ Flask backend + frontend integration

---

## 🧠 System Architecture

**Inputs**
- Person image
- Garment image (shirt OR pants)
- Garment type selector

**Core Components**
- Human Parsing (SegFormer – clothes-aware)
- Stable Diffusion Inpainting
- ControlNet (pose & geometry)
- IP-Adapter (appearance transfer)
- Face Extraction & Hard Merge (identity preservation)

**Output**
- Realistic try-on image with preserved face and body geometry

---

## 🧩 Project Structure

vton_v7_7/
│
├── app.py # Flask backend
├── pipeline.py # Core try-on pipeline
├── human_parsing.py # Garment segmentation
├── controlnet_utils.py # Pose extraction
├── face_utils.py # Face extraction & merging
├── config.py # Model & device config
│
├── templates/
│ └── index.html # Frontend UI
│
├── static/
│ └── results/ # Generated outputs
│
├── uploads/ # Temporary uploads (ignored in git)
├── requirements.txt
├── LICENSE
└── README.md


---

## 🚀 How It Works

1. User uploads a person image and a garment image
2. Human parsing isolates the selected garment region
3. ControlNet preserves body pose and structure
4. IP-Adapter transfers garment color and texture
5. Diffusion replaces only the selected garment
6. Original face is merged back for identity preservation
7. Final try-on image is returned

---

## ⚙️ Installation


pip install -r requirements.txt

Make sure you have:

- Python 3.9+
- CUDA-enabled GPU
- PyTorch with CUDA support

python app.py

http://localhost:5000


##⚠️ Known Limitations

Sleeve length cannot be fully changed without relaxing arm constraints.
Lighting differences between person and garment images may cause minor color variance.
Designed for single-person images only.
These limitations are intentional to preserve identity and realism.

##📜 License

This project is licensed under the Apache License 2.0.
You are free to use, modify, and distribute this project with proper attribution.
