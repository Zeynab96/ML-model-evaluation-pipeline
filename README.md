# 🤖 SmartML Optimizer

## 📘 Overview
**SmartML Optimizer** is an intelligent Machine Learning automation system that automatically selects, tunes, and evaluates the **best-performing model** for both **classification** and **regression** tasks.  
It supports a wide range of models — from classic algorithms like **Random Forest** and **SVM** to **Spline-based models** and advanced ensemble methods — and uses **Grid Search**, **Cross-Validation**, and **Automatic Balancing** to deliver the most optimized results.

---

## 🚀 Key Features
- ⚙️ **Automatic Model Selection** — Chooses the best model (Classification or Regression) based on dataset type.  
- 🧠 **Hyperparameter Optimization** — Uses `GridSearchCV` and `Optuna` for parameter tuning.  
- 📊 **Comprehensive Evaluation** — Calculates all major metrics (R², MAE, MSE, F1, Precision, Recall, AUC, Kappa).  
- 🔄 **Imbalance Handling** — Automatically balances classification data using `RandomUnderSampler`.  
- 📈 **Feature Importance Extraction** — Supports both coefficients and feature importances.  
- 💾 **Excel Reporting** — Saves results, best parameters, and feature importance in structured Excel sheets.  
- 🧩 **Spline Integration** — Optionally includes B-spline feature transformation for both regression and classification models.  

---

## 🧠 Supported Models

### 🔹 Classification
- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- Support Vector Machine (SVC)  
- (Optionally) MLP, XGBoost, CatBoost, LightGBM  

### 🔹 Regression
- Linear Regression  
- Lasso & Ridge Regression  
- Random Forest Regressor  
- Decision Tree Regressor  
- (Optionally) Spline-based Regression Models  

---

## ⚙️ Tech Stack
| Component | Technology |
|------------|-------------|
| Language | Python 3.x |
| ML Libraries | scikit-learn, imbalanced-learn |
| Optimization | GridSearchCV, Optuna |
| Data Handling | pandas, numpy |
| Reporting | openpyxl, Excel |
| Visualization | matplotlib |
| Model Saving | joblib |

---

## 🧩 Project Structure
```
SmartML-Optimizer/
├── classification_model.py       # Classification pipeline (balancing + tuning)
├── regression_model.py            # Regression pipeline (CV + tuning)
├── utils/
│   ├── data_preprocessing.py      # Cleaning, encoding, scaling, and balancing
│   ├── spline_features.py         # B-spline transformation utilities
│   └── evaluation_metrics.py      # Metric calculations and visualizations
├── results/
│   ├── feature_info.xlsx          # Feature importances & coefficients
│   └── model_results.xlsx         # Performance summary
├── README.md
└── requirements.txt
```

---

## ⚡ How It Works
1. **Load Dataset** — Reads Excel or CSV input.  
2. **Detect Task Type** — Classification or Regression (based on target variable).  
3. **Preprocessing** — Handles missing data, scaling, encoding, and class balancing.  
4. **Model Training** — Runs multiple ML models with grid search cross-validation.  
5. **Evaluation** — Computes all metrics and selects the best model automatically.  
6. **Reporting** — Saves results, feature importances, and model artifacts to Excel.  

---

## 🧪 Example Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/SmartML-Optimizer.git
cd SmartML-Optimizer

# Install dependencies
pip install -r requirements.txt

# Run Classification Script
python classification_model.py

# Run Regression Script
python regression_model.py
```

---

## 📈 Output Example
After running the pipeline, you’ll get:

- `feature_info.xlsx` → sorted coefficients or feature importances per model  
- `model_results.xlsx` → accuracy, precision, recall, F1-score, R², MAE, MSE, etc.  
- `best_model.pkl` → saved serialized model for future use  

---

## 🔍 Evaluation Metrics

**Classification:**
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Cohen’s Kappa  

**Regression:**
- R² (Coefficient of Determination)  
- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)

---

## 💡 Future Enhancements
- 🔹 Integration with **Optuna** for automatic hyperparameter search  
- 🔹 Support for **Neural Network models** (MLP, CNN, LSTM)  
- 🔹 Advanced **Spline Regression & Classification** modules  
- 🔹 Auto-detection of categorical features with hybrid encoding  

---

## 👩‍💻 Author
**Zeynab Tabatabaei**  
AI & Data Science Engineer  
📍 Hakim Toos University — Parsian Project  
📧 ztabatabaei974@gmail.com
💼 [LinkedIn](https://www.linkedin.com/in/zeynab-tabatabaei-950419233/)

---

## 🪪 License
This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.
