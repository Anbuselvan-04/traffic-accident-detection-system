# 🚦 IntelliCrash AI
### AI-Powered Traffic Accident Detection, Risk Assessment & Explainable Monitoring System

> An intelligent computer vision system that detects road traffic accidents from surveillance videos using a custom-trained YOLOv8 model and provides real-time accident analytics, heatmap visualization, fuzzy severity assessment, explainable AI (XAI), and automated emergency alerts.

---

## 🌟 Project Overview

Road accidents require immediate detection for faster emergency response. Traditional CCTV systems depend on human operators, making accident detection slow and error-prone.

**IntelliCrash AI** automatically identifies accident scenes from traffic surveillance footage and transforms raw video into actionable intelligence through AI-driven accident localization, risk analysis, heatmap generation, explainable predictions, and event reporting.

---

## ✨ Key Features

✔ Custom YOLOv8 Accident Detection Model

✔ Accurate Accident Bounding Box Localization

✔ Accident Heatmap Visualization

✔ AI-powered Risk Score Calculation

✔ Fuzzy Logic Severity Classification

✔ Explainable AI (XAI)

✔ Automated Event Logging

✔ Accident Frame Gallery

✔ Email Alert System

✔ Location Information Integration

✔ Interactive Streamlit Dashboard

✔ Video Upload & Offline Analysis

---

## 🧠 System Architecture

```
Video Input
      │
      ▼
Frame Extraction
      │
      ▼
YOLOv8 Accident Detection
      │
      ▼
Bounding Box Localization
      │
      ▼
Accident Heatmap
      │
      ▼
Risk Assessment
      │
      ▼
Fuzzy Severity Engine
      │
      ▼
Explainable AI (XAI)
      │
      ▼
Analytics Dashboard
      │
      ▼
Event Log & Email Alert
```

---

## ⚙️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Deep Learning | YOLOv8 |
| Computer Vision | OpenCV |
| Dashboard | Streamlit |
| Explainability | SHAP |
| Decision System | Fuzzy Logic |
| Visualization | Matplotlib |
| Data Processing | NumPy, Pandas |

---

## 📂 Project Structure

```
traffic-accident-detection-system
│
├── models/
│   └── best.pt
│
├── dashboard.py
├── detect_and_analyze.py
├── heatmap_analysis.py
├── fuzzy_severity.py
├── xai_explainer.py
├── email_alert.py
├── location_service.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Anbuselvan-04/traffic-accident-detection-system.git

cd traffic-accident-detection-system
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Application

```bash
streamlit run dashboard.py
```

Open

```
http://localhost:8501
```

---

## 📊 Dashboard Modules

- 🚨 Accident Detection
- 📦 Bounding Box Visualization
- 🔥 Accident Heatmap
- 📈 Analytics Dashboard
- 🧠 XAI Explanation Panel
- 📸 Accident Gallery
- 📋 Event Log
- 📧 Email Alerts
- 📍 Location Details

---

## 🎯 Applications

- Smart City Surveillance
- Highway Monitoring
- Intelligent Transportation Systems
- Traffic Control Centers
- Emergency Response Systems
- AI-based Public Safety

---


## 👨‍💻 Developed By

**Anbuselvan R**

B.E. Computer Science and Engineering

AI & Machine Learning Enthusiast

GitHub:
https://github.com/Anbuselvan-04

---

## ⭐ If you found this project useful, consider giving it a Star.
