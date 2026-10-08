Live App Link Render: https://fair-loan-predictor.onrender.com

# 🏦 Fair Loan Predictor — SDG 10: Reduced Inequalities

An AI-powered home loan eligibility predictor built with Machine Learning and Flask.  
Predicts loan approval based on **real Indian income ranges** and **audits itself for gender bias** — directly addressing [UN SDG 10: Reduced Inequalities](https://sdgs.un.org/goals/goal10).

---

## 🎯 Project Overview

Most loan approval systems are black boxes — and many perpetuate historical biases against women, lower-income groups, or less-educated applicants. This project:

- **Predicts** whether a home loan will be approved using a trained ML model (99%+ accuracy)
- **Explains** the key factors driving each decision
- **Audits** itself for fairness across gender groups
- **Enforces** strict real-world validation (no invalid or impractical inputs accepted)

---

## ✨ Features

### 🔮 Predict Page
- Toggle buttons, segment pills, dual sliders for all inputs
- **Live Financial Summary** — updates EMI, loan-to-income ratio, and EMI burden in real time as you type
- Instant validation with clear error messages (income range, EMI rule, etc.)
- Result panel with approval probability bar + key factor breakdown

### 📊 Dashboard Page
- Model comparison (Random Forest vs Gradient Boosting vs Logistic Regression)
- Feature importance bar chart
- Dataset stats overview

### ⚖️ Fairness Audit Page
- Gender-group metrics table (Approval Rate, TPR, FPR)
- Visual approval rate comparison
- Auto-generated bias interpretation with pass/fail flags

### ℹ️ About Page
- SDG 10 context
- Full tech stack and ML pipeline details
- Fairness metrics explained

---

## 🤖 ML Pipeline

| Step | Detail |
|---|---|
| Dataset | 2,000 realistic Indian loan applications |
| Income range | ₹10,000 – ₹5,00,000 / month |
| Loan range | ₹5 Lakh – ₹2 Crore |
| Features | 14 (incl. derived: EMI, FOIR, Loan-to-Income) |
| Models trained | Random Forest, Gradient Boosting, Logistic Regression |
| Selection | Best AUC-ROC via 5-fold cross-validation |
| Best model | Logistic Regression (AUC = 99.94%, Acc = 98.75%) |
| Fairness audit | Approval rate parity, TPR equality, FPR monitoring by gender |

---

## 🛡️ Validation Rules

All inputs are validated on both **client (browser)** and **server (Flask)**:

| Field | Rule |
|---|---|
| Applicant Income | ₹10,000 – ₹5,00,000 / month |
| Co-applicant Income | ₹0 – ₹3,00,000 / month |
| Loan Amount | ₹5L – ₹2Cr (500–20,000 in thousands) |
| Loan Tenure | 10 / 15 / 20 / 25 / 30 years only |
| Credit History | 0 (bad/none) or 1 (good) |
| EMI Rule | Monthly EMI cannot exceed **90% of applicant income** |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/harmohitsingh/fair-loan-predictor.git
cd fair-loan-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```
This generates the `model/` folder with trained model artifacts.  
*(Takes ~30–60 seconds. Only needs to be done once.)*

### 4. Start the web app
```bash
python run_server.py
```

### 5. Open your browser
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
fair-loan-predictor/
│
├── app.py                        # Flask REST API backend
├── run_server.py                 # Clean server launcher
├── train_model.py                # ML model training + fairness audit
├── generate_dataset.py           # Realistic Indian dataset generator
│
├── realistic_loan_dataset.csv    # Training dataset (2,000 records)
├── ibm_home_loan_approval_dataset.csv  # Original IBM dataset
│
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html                # Full single-page frontend
│
└── model/                        # Generated after training
    ├── model.pkl                 # Trained classifier
    ├── scaler.pkl                # StandardScaler
    ├── encoders.pkl              # Label encoders
    └── metadata.json             # Model stats + fairness report
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web app home |
| `POST` | `/api/predict` | Predict loan approval |
| `GET` | `/api/stats` | Model metrics + feature importance |
| `GET` | `/api/fairness` | Fairness audit report by gender |
| `GET` | `/api/metadata` | Full model metadata |

### Example: POST `/api/predict`
```json
{
  "gender": "Male",
  "married": "Yes",
  "dependents": "0",
  "education": "Graduate",
  "self_employed": "No",
  "property_area": "Urban",
  "credit_history": "1",
  "applicant_income": "60000",
  "coapplicant_income": "30000",
  "loan_amount": "3000",
  "loan_term": "240"
}
```

### Response
```json
{
  "approved": true,
  "prediction": "Approved",
  "approval_prob": 100.0,
  "confidence": 100.0,
  "factors": [...],
  "derived": {
    "total_income": 90000,
    "emi_monthly": 12500,
    "emi_ratio": 13.9,
    "loan_to_income": 0.4
  }
}
```

---

## 🧰 Tech Stack

- **Python 3.11+**
- **scikit-learn** — ML models, preprocessing, cross-validation
- **Flask** — REST API and web server
- **NumPy** — numerical feature engineering
- **Vanilla JS + CSS** — frontend (zero frameworks, zero dependencies)

---

## 🌍 SDG 10 Connection

> *"Reduce inequality within and among countries"*  
> — United Nations Sustainable Development Goal 10

Financial lending is one of the primary mechanisms through which inequality is either reinforced or challenged. By building a transparent, explainable, and fairness-audited loan predictor, this project:

- Surfaces **hidden bias** in approval patterns across gender groups
- Provides **equal opportunity scoring** — the same algorithm for everyone
- Makes AI decision-making **explainable** to applicants
- Demonstrates how ML can actively **measure and reduce** systemic discrimination

---

## 📜 License

MIT License — free to use, modify, and distribute.
