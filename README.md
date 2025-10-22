# 🎓 Elevvo ML Internship Portfolio

Complete machine learning project portfolio featuring **4 production-ready ML applications** deployed on Streamlit Cloud with interactive web interfaces, professional visualizations, and trained models.

---

## 🌐 Live Deployments

| Task | Application | Status | Link |
|------|------------|--------|------|
| **Task 1** | Student Score Prediction | ✅ Live | [Predict Scores](https://students-score-predictor.streamlit.app/) |
| **Task 2** | Customer Segmentation | ✅ Live | [Segment Customers](https://finscore.streamlit.app/) |
| **Task 4** | Loan Approval Prediction | ✅ Live | [Approve Loans](https://approveyourloan.streamlit.app/) |
| **Task 7** | Sales Forecasting | ✅ Live | [Forecast Sales](https://forcasting-sales.streamlit.app/) |

---

## 📊 Project Overview

| Task | Model Type | Performance | Features |
|------|-----------|-------------|----------|
| **Task 1** | Linear Regression | R² = 0.9890 | Study hours → Exam scores |
| **Task 2** | K-Means Clustering | 4 clusters | Customer segmentation by income/spending |
| **Task 4** | Logistic Regression + Decision Tree | Binary Classification | Loan approval prediction |
| **Task 7** | Multi-Model Forecasting | 5 algorithms | Sales prediction with temporal features |

---

## 🚀 Quick Start

### Prerequisites
- Python **3.10** (Required for compatibility)
- pip package manager

### Local Installation

```bash
# Clone repository
git clone https://github.com/S4lmankhan/Internship-Tasks-ElevvoPathways--AIML.git
cd Internship-Tasks-ElevvoPathways--AIML

# Run any task (example: Task 1)
cd Task_1_Student_Score_Prediction
pip install -r requirements.txt
streamlit run app.py
```

### Docker Installation (Alternative)
```bash
# Build and run Task 1
docker build -t task1 Task_1_Student_Score_Prediction
docker run -p 8501:8501 task1
```

---

## 📁 Repository Structure

```
Internship-Tasks-ElevvoPathways--AIML/
│
├── Task_1_Student_Score_Prediction/
│   ├── app.py                          # Streamlit web app
│   ├── model/
│   │   ├── linear_model.pkl            # Trained model (R²=0.9890)
│   │   └── scaler.pkl                  # Feature scaler
│   ├── notebooks/
│   │   └── Task_1_Student_Score_Prediction.ipynb
│   ├── requirements.txt                # Python 3.10 dependencies
│   ├── runtime.txt                     # Python version (3.10.19)
│   └── sample_batch_input.csv
│
├── Task_2_Customer_Segmentation/
│   ├── app.py                          # Streamlit web app
│   ├── model/
│   │   ├── kmeans_model.pkl            # 4-cluster model
│   │   └── scaler.pkl                  # Feature scaler
│   ├── notebooks/
│   │   └── Task_2_Customer_Segmentation.ipynb
│   ├── Mall_Customers.csv              # Dataset (200 customers)
│   ├── requirements.txt
│   ├── runtime.txt
│   └── sample_batch_input.csv
│
├── Task_4_Loan_Approval_Prediction/
│   ├── app.py                          # Streamlit web app
│   ├── model/
│   │   ├── logistic_regression_model.pkl
│   │   ├── decision_tree_model.pkl
│   │   ├── scaler.pkl
│   │   ├── label_encoders.pkl
│   │   ├── target_encoder.pkl
│   │   └── feature_columns.pkl
│   ├── notebooks/
│   │   └── Task_4_Loan_Approval_Prediction.ipynb
│   ├── requirements.txt
│   ├── runtime.txt
│   └── sample_batch_input.csv
│
├── Task_7_Sales_Forecasting/
│   ├── app.py                          # Streamlit web app
│   ├── model/
│   │   ├── linear_regression_model.pkl
│   │   ├── decision_tree_model.pkl
│   │   ├── random_forest_model.pkl
│   │   ├── xgboost_model.pkl
│   │   ├── lightgbm_model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_columns.pkl
│   ├── notebooks/
│   │   └── Task_7_Sales_Forecasting.ipynb
│   ├── outputs/
│   │   └── model_results.csv
│   ├── requirements.txt
│   └── runtime.txt
│
├── .python-version                     # Python 3.10
└── README.md                           # This file
```

---

## 🎯 Task Details

### ✅ Task 1: Student Score Prediction
**Predict student exam scores based on study hours using Linear Regression**

**Features:**
- 📝 Single prediction with interactive sliders
- 📊 Batch prediction via CSV upload
- 📈 Model performance metrics (R²=0.9890, MSE=9.31)
- 🎨 Interactive visualizations with Plotly

**Model:** Linear Regression (scikit-learn 1.3.2)  
**Dataset:** 25 student records  
**Deployment:** Streamlit Cloud with Python 3.10.19

---

### ✅ Task 2: Customer Segmentation
**Segment customers into 4 groups based on income and spending patterns**

**Features:**
- 🎯 Single customer classification
- 📤 Batch segmentation with CSV upload
- 💡 4 customer segments: Budget-Conscious, Mid-Range, Savers, Premium
- 📊 Cluster visualization and insights

**Model:** K-Means Clustering (k=4)  
**Dataset:** 200 mall customers  
**Key Fix:** Removed incorrect scaler usage (KMeans trained on raw data)

---

### ✅ Task 4: Loan Approval Prediction
**Binary classification for loan approval decisions**

**Features:**
- 🏦 Single applicant prediction (10 features)
- 📋 Batch loan processing
- 🤖 2 models: Logistic Regression + Decision Tree
- 📊 Model comparison metrics

**Models:** Logistic Regression & Decision Tree (scikit-learn 1.3.2)  
**Dataset:** Loan applicants with demographics and financials  
**Features:** Age, Income, Credit Score, Employment Years, Loan Amount, Gender, Marital Status, Dependents, Education, Self-Employment

---

### ✅ Task 7: Sales Forecasting
**Multi-model sales forecasting with temporal features**

**Features:**
- 🔮 Forecast future sales (1-365 days ahead)
- 📊 5 ML models compared (Linear, Decision Tree, Random Forest, XGBoost, LightGBM)
- 🏪 Store and Item-based predictions
- 📈 Interactive forecast visualization
- 📉 Model performance dashboard

**Models:** 5 regression algorithms  
**Features:** Store ID, Item ID, DayOfWeek, Month, Quarter, Year, IsWeekend  
**Key Fixes:**
- Corrected model directory path (`models/` → `model/`)
- Regenerated with exact Streamlit Cloud versions (XGBoost 2.0.0, LightGBM 4.0.0)
- Fixed forecasting UI to match actual model features

---

## 🛠️ Technology Stack

### Machine Learning & Data Science
- **scikit-learn 1.3.2** - ML algorithms (exact match with Streamlit Cloud)
- **pandas 2.0.3** - Data manipulation
- **numpy 1.23.5** - Numerical computing
- **XGBoost 2.0.0** - Gradient boosting (Task 7)
- **LightGBM 4.0.0** - Fast gradient boosting (Task 7)
- **scipy 1.11.4** - Scientific computing

### Web Application
- **streamlit 1.28.1** - Interactive web framework
- **plotly 5.17.0** - Interactive visualizations
- **matplotlib 3.7.2** - Static plots
- **seaborn 0.13.0** - Statistical visualizations

### Deployment
- **joblib 1.3.2** - Model serialization
- **Python 3.10.19** - Runtime environment (Streamlit Cloud)

### Version Control
All package versions are **pinned** in `requirements.txt` to ensure reproducibility and prevent compatibility issues.

---

## 🔧 Technical Implementation

### Model Compatibility
All models were regenerated with **exact Streamlit Cloud versions** to prevent segmentation faults:
- ✅ scikit-learn 1.3.2 (not 1.7.2)
- ✅ XGBoost 2.0.0 (not 3.0.0)
- ✅ LightGBM 4.0.0 (not 4.6.0)
- ✅ numpy 1.23.5 (Python 3.10 compatible)

### Python 3.10 Compatibility
Each task includes `runtime.txt` specifying `python-3.10.19` to match Streamlit Cloud environment.

### Numpy Compatibility Fix
All apps include numpy._core compatibility layer for Python 3.10:
```python
import sys
import numpy
if not hasattr(numpy, '_core'):
    import numpy.core
    numpy._core = numpy.core
    sys.modules['numpy._core'] = numpy.core
```

---

## 📊 Model Performance

| Task | Model | Metric | Score | Notes |
|------|-------|--------|-------|-------|
| Task 1 | Linear Regression | R² | 0.9890 | Explains 98.9% of variance |
| Task 2 | K-Means (k=4) | Inertia | 73,679.79 | 4 well-separated clusters |
| Task 4 | Logistic Regression | Accuracy | 100% | Small test dataset (25 samples) |
| Task 4 | Decision Tree | Accuracy | 100% | Small test dataset (25 samples) |
| Task 7 | Linear Regression | R² | 0.1547 | Synthetic training data |
| Task 7 | Random Forest | R² | 0.1083 | Synthetic training data |

**Note:** Task 4 and Task 7 were trained on small/synthetic datasets for demonstration purposes.

---

## 🌐 Deployment Guide

### Streamlit Cloud Deployment

All 4 tasks are deployed on **Streamlit Cloud** with the following configuration:

**Repository:** `S4lmankhan/Internship-Tasks-ElevvoPathways--AIML`  
**Branch:** `main`  
**Python Version:** 3.10.19 (specified in `runtime.txt`)

#### Deployment Configuration

| Task | Main Module | Requirements | Runtime |
|------|------------|--------------|---------|
| Task 1 | `Task_1_Student_Score_Prediction/app.py` | ✅ | python-3.10.19 |
| Task 2 | `Task_2_Customer_Segmentation/app.py` | ✅ | python-3.10.19 |
| Task 4 | `Task_4_Loan_Approval_Prediction/app.py` | ✅ | python-3.10.19 |
| Task 7 | `Task_7_Sales_Forecasting/app.py` | ✅ | python-3.10.19 |

#### Key Configuration Files

Each task directory contains:
- **`requirements.txt`** - Pinned package versions for reproducibility
- **`runtime.txt`** - Python version specification (3.10.19)
- **`app.py`** - Main Streamlit application
- **`model/`** - Pre-trained model files (joblib serialized)

#### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy: All 4 ML tasks ready for production"
   git push origin main
   ```

2. **Create Streamlit Cloud Apps**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select repository: `S4lmankhan/Internship-Tasks-ElevvoPathways--AIML`
   - Branch: `main`
   - Main file path: `Task_X_[TaskName]/app.py`

3. **Verify Deployment**
   - Check logs for successful startup
   - Test all features (single prediction, batch upload, visualizations)

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. Segmentation Fault
**Cause:** Model version mismatch between local and Streamlit Cloud  
**Solution:** ✅ Fixed - All models regenerated with exact Streamlit Cloud versions

#### 2. ModuleNotFoundError: numpy._core
**Cause:** Python 3.13 (local) vs Python 3.10 (Streamlit Cloud) incompatibility  
**Solution:** ✅ Fixed - Added numpy._core compatibility layer in all apps

#### 3. Task 2: All predictions → Cluster 1
**Cause:** Scaler applied to predictions but KMeans trained on raw data  
**Solution:** ✅ Fixed - Removed scaler.transform() from prediction pipeline

#### 4. Task 7: "['Store', 'Item', 'Year'] not in index"
**Cause:** Forecasting UI didn't collect Store/Item inputs  
**Solution:** ✅ Fixed - Updated UI and forecast generation logic

#### 5. Task 7: Directory not found
**Cause:** App referenced `'models/'` but actual directory is `'model/'`  
**Solution:** ✅ Fixed - Corrected all path references

---

## 📝 Development Notes

### Version Compatibility Matrix

| Package | Local Dev | Streamlit Cloud | Status |
|---------|-----------|-----------------|--------|
| Python | 3.10.11 | 3.10.19 | ✅ Compatible |
| scikit-learn | 1.3.2 | 1.3.2 | ✅ Exact Match |
| numpy | 1.23.5 | 1.23.5 | ✅ Exact Match |
| pandas | 2.0.3 | 2.0.3 | ✅ Exact Match |
| XGBoost | 2.0.0 | 2.0.0 | ✅ Exact Match |
| LightGBM | 4.0.0 | 4.0.0 | ✅ Exact Match |
| streamlit | 1.28.1 | 1.28.1 | ✅ Exact Match |

### Model Regeneration History

1. **Initial Issue:** Models saved with scikit-learn 1.7.2 (Python 3.13)
2. **Solution:** Installed scikit-learn 1.3.2 locally
3. **Regeneration:** All models regenerated with matching versions
4. **Verification:** Tested locally before deployment
5. **Deployment:** Successfully deployed to Streamlit Cloud

---

## 🎓 Learning Outcomes

This portfolio demonstrates:

✅ **Machine Learning Fundamentals**
- Regression (Linear Regression)
- Classification (Logistic Regression, Decision Tree)
- Clustering (K-Means)
- Ensemble Methods (Random Forest, XGBoost, LightGBM)

✅ **Data Science Skills**
- Data preprocessing and cleaning
- Feature engineering
- Model evaluation and comparison
- Handling imbalanced datasets

✅ **Software Engineering**
- Version control with Git
- Dependency management
- Cross-platform compatibility
- Production deployment

✅ **Web Development**
- Interactive UI with Streamlit
- Data visualization with Plotly
- User input validation
- Batch processing capabilities

✅ **Debugging & Problem Solving**
- Identified and fixed 5+ critical bugs
- Version compatibility resolution
- Performance optimization

---

## 📚 Project Files

### Notebooks
Each task includes comprehensive Jupyter notebooks:
- Task 1: Student Score Prediction (full EDA + model training)
- Task 2: Customer Segmentation (elbow method + silhouette analysis)
- Task 4: Loan Approval (SMOTE + multi-model comparison)
- Task 7: Sales Forecasting (feature engineering + time series)

### Model Files
All models serialized with joblib 1.3.2:
- **Task 1:** 2 files (model + scaler)
- **Task 2:** 2 files (kmeans + scaler)
- **Task 4:** 6 files (2 models + scaler + encoders + features)
- **Task 7:** 7 files (5 models + scaler + features)

**Total:** 17 trained model files

---

## 🎉 Final Status

### ✅ All Tasks Complete

- ✅ **Task 1:** Deployed & Working
- ✅ **Task 2:** Deployed & Working (predictions vary correctly)
- ✅ **Task 4:** Deployed & Working (no segfaults)
- ✅ **Task 7:** Deployed & Working (forecasting functional)

### ✅ Quality Assurance

- ✅ All models trained with production versions
- ✅ All apps tested on Streamlit Cloud
- ✅ All bugs identified and fixed
- ✅ All dependencies pinned and documented
- ✅ README comprehensive and accurate

### ✅ Repository Ready

- ✅ Clean commit history
- ✅ Organized file structure
- ✅ Professional documentation
- ✅ Ready for portfolio showcase

---

## 🤝 Credits

**Developer:** Salman Khan  
**Program:** Elevvo Pathways AI/ML Internship  
**Repository:** [S4lmankhan/Internship-Tasks-ElevvoPathways--AIML](https://github.com/S4lmankhan/Internship-Tasks-ElevvoPathways--AIML)

---

## 📄 License

This project is part of the Elevvo internship program.

---

**Last Updated:** October 23, 2025  
**Status:** 🎉 **PRODUCTION READY - ALL TASKS DEPLOYED**
