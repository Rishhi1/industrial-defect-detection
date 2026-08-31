# Industrial Defect Detection

An AI-powered industrial visual inspection system for detecting manufacturing defects using deep-learning-based anomaly detection.

The system uses a pretrained **ResNet18** backbone to extract visual features from industrial images. Normal samples are represented using category-specific **memory banks**, and anomaly scores are calculated by comparing an input image against the learned normal reference. A selected threshold is then used to classify the image as **NORMAL** or **ANOMALOUS**.

The project also provides visual anomaly localization through heatmaps and an interactive **Streamlit** web application.

---

## 🚀 Live Demo

**Streamlit App:**  
[Launch Industrial Defect Detection]([YOUR_STREAMLIT_APP_URL](https://industrial-defect-detection-s38f4vjmmver3tsrbkifjb.streamlit.app))

> Replace `YOUR_STREAMLIT_APP_URL` with your deployed Streamlit URL.

---

## 📌 Project Overview

Quality inspection is an important part of modern manufacturing. Traditional manual inspection can be time-consuming, inconsistent, and difficult to scale.

This project demonstrates an automated computer-vision-based inspection pipeline capable of identifying anomalous industrial products from images.

The system is designed around **anomaly detection**, meaning it learns the appearance of normal products and identifies images that deviate significantly from the learned normal reference.

---

## 🎯 Objectives

- Detect defects in industrial product images.
- Learn normal visual patterns from defect-free training images.
- Extract meaningful visual features using a pretrained CNN.
- Build category-specific feature memory banks.
- Calculate anomaly scores using feature-space distances.
- Automatically determine suitable classification thresholds.
- Classify images as NORMAL or ANOMALOUS.
- Localize suspected defect regions using anomaly heatmaps.
- Provide an interactive Streamlit dashboard.
- Support multiple MVTec AD categories.
- Provide a reproducible CPU-compatible workflow.

---

## 🧠 Methodology

The overall pipeline is:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Pretrained ResNet18
     ↓
Feature Extraction
     ↓
Feature Representation
     ↓
Normal Feature Memory Bank
     ↓
Distance / Anomaly Score
     ↓
Thresholding
     ↓
NORMAL / ANOMALOUS
     ↓
Anomaly Localization
     ↓
Heatmap
     ↓
Final Inspection Result
