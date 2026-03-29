# Heart Disease Detection 🫀

A machine learning project that predicts whether a patient has heart disease based on clinical measurements — built to help doctors prioritize high-risk patients early.

---

## Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection significantly improves patient outcomes. This project builds a binary classification model using real clinical data, with a focus on **maximizing recall** — because missing a sick patient is far more dangerous than a false alarm.

---

## Dataset

- **Source:** [Cleveland Heart Disease Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Samples:** 302 unique patients (after removing 723 duplicate rows)
- **Features:** 13 clinical measurements
- **Target:** 1 = Heart Disease, 0 = No Disease
- **Class balance:** ~54% / 46% — balanced dataset

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy (0–3) |
| thal | Thalassemia type (0–3) |

---

## Project Workflow

```
1. Problem Framing       → Define metric priority (Recall > Accuracy)
2. EDA                   → Distributions, correlations, class balance
3. Data Cleaning         → Found and removed 723 duplicate rows (data leakage)
4. Pipeline              → StandardScaler + model in one clean object
5. Model Selection       → Compared Logistic Regression, SVM, Random Forest
6. Hyperparameter Tuning → GridSearchCV optimizing for Recall
7. Final Evaluation      → Confusion matrix, ROC curve, AUC
8. Model Saving          → joblib export for production use
```

---

## Key Finding — Data Leakage Discovery

The raw dataset contained **723 duplicate rows** out of 1025 total. Before cleaning, the model achieved a suspiciously perfect score:

```
Recall: 100% | Precision: 100% | F1: 100%  ← Red flag
```

After removing duplicates and retaining 302 unique patients, results became honest and realistic:

```
Recall: 78.8% | Precision: 78.8% | F1: 78.8% | AUC: 0.858
```

This is a critical lesson: **perfect scores are a red flag, not a celebration.**

---

## Results

### Model Comparison (cross-validation on training set)

| Model | Recall | Precision | F1 |
|---|---|---|---|
| Logistic Regression | 89.3% | 81.9% | 85.5% |
| SVM | ~87% | ~84% | ~85% |
| **Random Forest** | **98.3%** | **98.1%** | **98.2%** |

### Final Test Set Performance (Random Forest, tuned)

| Metric | Score |
|---|---|
| Recall | 78.8% |
| Precision | 78.8% |
| F1 Score | 78.8% |
| AUC | **0.858** |

### Confusion Matrix

```
               Predicted No Disease   Predicted Disease
Actual No Disease       21                  7
Actual Disease           7                 26
```

- **True Positives:** 26 sick patients correctly identified
- **False Negatives:** 7 sick patients missed (main area for improvement)

### ROC Curve

AUC = 0.858 — the model has an **85.8% chance** of correctly ranking a sick patient above a healthy one. A random classifier scores 0.5.

---

## Best Hyperparameters (GridSearchCV)

```python
{
    'max_depth': 5,
    'max_features': 'sqrt',
    'min_samples_split': 2,
    'n_estimators': 100
}
```

GridSearchCV tested **54 combinations × 5 folds = 270 model fits**, optimizing for recall.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas + NumPy | Data manipulation |
| Matplotlib + Seaborn | Visualization |
| Scikit-learn | ML pipeline, models, evaluation |
| Joblib | Model serialization |

---

## Project Structure

```
heart-disease-detection/
│
├── heart_disease_detection.ipynb   ← Full notebook with analysis
├── heart_disease_detection.py      ← Clean Python script
├── heart_disease_model.pkl         ← Saved trained model
├── README.md                       ← This file
└── data/
    └── heart.csv                   ← Dataset
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/amrmohammedali/heart-disease-detection.git
cd heart-disease-detection

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib

# Run the notebook
jupyter notebook heart_disease_detection.ipynb

# Or run the script
python heart_disease_detection.py
```

---

## Load the Saved Model

```python
import joblib
import pandas as pd

# Load model
model = joblib.load("heart_disease_model.pkl")

# Predict on a new patient
patient = pd.DataFrame([{
    'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 130,
    'chol': 250, 'fbs': 0, 'restecg': 1, 'thalach': 165,
    'exang': 0, 'oldpeak': 1.2, 'slope': 2, 'ca': 0, 'thal': 2
}])

prediction = model.predict(patient)
probability = model.predict_proba(patient)[:, 1]

print(f"Prediction: {'Heart Disease' if prediction[0]==1 else 'No Disease'}")
print(f"Confidence: {probability[0]:.1%}")
```

---

## Key Learnings

1. **Perfect scores = red flag** — always investigate before celebrating
2. **Check duplicates before splitting** — data leakage ruins real-world performance  
3. **Choose metrics based on domain** — recall matters more than accuracy in medicine
4. **predict_proba over predict** — confidence scores are more useful in production
5. **Pipeline over manual steps** — prevents data leakage and simplifies deployment

---

## Author

**Amr Mohammed Ali**
- 🎓 CS & AI Graduate — Beni Suef University (2024)
- 💼 ML Engineer | Data Scientist
- 🔗 [Kaggle Notebook](https://www.kaggle.com/code/amrmohammedali/heart-disease-detection)
- 📧 Connect on LinkedIn

---

## License

MIT License — free to use and modify with attribution.
