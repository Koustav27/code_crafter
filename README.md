# 🌱 Crop Leaf Disease Detection using CNN

## 📌 Overview
This project was developed for **Smart India Hackathon 2026 (Problem Statement ID: 26193)** under the theme *Agriculture, Food & Rural Development*.  
Our solution helps farmers and agricultural workers detect crop leaf diseases early using **AI-powered image classification**.

By simply uploading a leaf image through a **Streamlit web app**, the system classifies it into one of **38 categories** (healthy or diseased) within seconds.  
This acts as a **low-cost, rapid screening tool** for farmers, enabling timely action before diseases spread.

---

## 🚀 Features
- Upload crop-leaf images via a simple web interface.
- CNN model trained on **70,000+ images** across 38 classes.
- Achieves **97.07% training accuracy** and **95.08% validation accuracy**.
- Farmer-friendly results with clear disease labels.
- Lightweight workflow designed for field-level screening.

---

## 🛠️ Tech Stack
- **Python**
- **TensorFlow/Keras** – CNN model training & inference
- **Streamlit** – Web interface
- **NumPy, Pandas** – Preprocessing & data handling
- **Matplotlib, Seaborn** – Visualization
- **Librosa** – (optional, for audio preprocessing if extended)

---

## 🧠 Model Architecture
- Input: 128×128 RGB leaf image
- Layers:
  - Multiple **Conv2D** + **MaxPooling2D** layers for feature extraction
  - **Dropout** layers to reduce overfitting
  - **Dense** layers for classification
- Output: **Softmax layer with 38 classes**

---

## 📊 Training Details
- **Dataset:** [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Optimizer:** Adam (learning rate = 0.0001)
- **Loss Function:** Categorical Crossentropy
- **Epochs:** 10
- **Results:**
  - Training Accuracy: **97.07%**
  - Validation Accuracy: **95.08%**

---

## ⚡ How to Run
- **1.** Clone this repository: git clone https://github.com/Koustav27/code_crafter.git
- **2.** pip install -r requirement.txt
- **3.** streamlit run main.py
- **4.** Upload a leaf image → Get instant disease classification.
   
---
