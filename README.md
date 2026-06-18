# 🫘 Dry Bean Type Classifier

A machine learning web app that classifies dry bean varieties from physical measurements using a Support Vector Machine (SVM) model — built as part of the Applied Data Science, ML & AI program at **E&ICT Academy, IIT Guwahati**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bean-classifier-app-byvineeth.streamlit.app/)

---

## 🌱 Project Overview

An agriculture company wanted to automate the manual classification of dry beans — a process that was labour-intensive, error-prone and inefficient at scale. This project builds a supervised ML pipeline to classify beans based on 16 physical characteristics (area, perimeter, shape, compactness etc.) captured via camera-based computer vision systems.

**7 Bean Classes:**

| Class | Description |
|---|---|
| 🟡 SEKER | Small, round, yellowish bean |
| 🔴 BARBUNYA | Medium, speckled pinkish-red bean |
| ⚫ BOMBAY | Large, dark brown bean |
| 🟤 CALI | Medium-large, light brown bean |
| 🟢 DERMASON | Small, oval-shaped green bean |
| 🟠 HOROZ | Large, elongated orange-brown bean |
| 🔵 SIRA | Medium, oval-shaped pale bean |

---

## 📊 Dataset

- **Source:** Dry Bean Dataset (UCI Machine Learning Repository)
- **Samples:** 13,611
- **Features:** 16 physical measurements (Area, Perimeter, ShapeFactors etc.)
- **Target:** 7 bean classes
- **Collection Method:** Camera-based computer vision systems

---

## 🔬 ML Pipeline

```
Data Loading → EDA → Outlier Treatment → Feature Engineering
→ Model Building → Class Imbalance Handling → Evaluation
→ Hyperparameter Tuning → Model Comparison → Deployment
```

### Models Trained & Compared

| Model | Test Accuracy | F1 Score | Overfitting |
|---|---|---|---|
| **SVM (Best)** | **92.58%** | **92.57%** | **No** |
| Logistic Regression | 92.47% | 92.48% | No |
| Random Forest | 92.25% | 92.25% | No |
| Gradient Boosting | 92.21% | 92.21% | No |
| KNN | 91.99% | 92.00% | No |
| Naive Bayes | 89.90% | 89.92% | No |
| Decision Tree | 89.31% | 89.28% | Yes |
| AdaBoost | 82.56% | 81.14% | No |

### Key Findings

- **Best Model:** SVM with RBF kernel (C=10, gamma=scale) → 92.58% accuracy
- **Minority Class:** BOMBAY bean (~522 samples) — handled via SMOTE
- **High Multicollinearity:** Area, Perimeter, ConvexArea, EquivDiameter → addressed with StandardScaler
- **Overfitting:** Decision Tree showed clear overfitting (Train=100%, Test=89%)
- **Best Features:** ShapeFactor1, ShapeFactor2 & Compactness for class separation

---

## 🚀 Streamlit App

Live demo → **[Bean Classifier App](https://bean-classifier-app-byvineeth.streamlit.app/)**

**App Features:**
- Input 16 bean measurements via interactive sliders
- ShapeFactors via Low / Medium / High dropdowns
- Predicts bean type with confidence percentage
- Displays class probabilities for all 7 classes
- Mobile-friendly dark green UI

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Model | SVM (RBF Kernel) — Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Class Imbalance | Imbalanced-learn (SMOTE) |
| Model Persistence | Joblib |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📁 Repository Structure

```
bean-classifier/
│
├── app.py                ← Streamlit app
├── requirements.txt      ← Python dependencies
├── bean_model.pkl        ← Trained SVM model
├── scaler.pkl            ← StandardScaler
├── label_encoder.pkl     ← LabelEncoder
└── README.md             ← This file
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/Vineeth-Muraleedharan/Bean-classifier.git
cd Bean-classifier

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
numpy
pandas
scikit-learn
joblib
openpyxl
```

---

## 👤 Author

**Vineeth Muraleedharan**  
Senior Radiation Therapist | Clinical AI Developer  
Applied Data Science, ML & AI - E&ICT Academy, IIT Guwahati

[![GitHub](https://img.shields.io/badge/GitHub-Vineeth--Muraleedharan-black?logo=github)](https://github.com/Vineeth-Muraleedharan)

---

## 📄 License

This project is part of an academic mini project submission for IIT Guwahati supervised ML coursework.
