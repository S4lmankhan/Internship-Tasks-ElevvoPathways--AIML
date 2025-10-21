# Task 1: Student Score Prediction

## 📋 Project Overview
This project builds a **Linear Regression Model** to predict students' exam scores based on study habits and academic history. It includes a complete machine learning pipeline with data exploration, model training, evaluation, and **an interactive Streamlit web application** for easy testing and deployment.

**Features:**
- ✅ Machine learning model trained on 10,000 student records
- ✅ 95.57% accuracy (R² Score)
- ✅ Interactive Streamlit web app with 3 feature tabs
- ✅ Batch prediction capability (upload CSV)
- ✅ Complete analysis notebook with visualizations
- ✅ Production-ready code

**Status:** ✅ Complete & Ready for Deployment  
**Last Updated:** October 20, 2025

---

## 🎯 Project Achievements
- ✅ Trained Linear Regression model with 95.57% accuracy
- ✅ Built interactive Streamlit web application
- ✅ Created comprehensive analysis notebook
- ✅ Generated 7 visualization outputs
- ✅ Implemented batch prediction functionality
- ✅ Prepared model for cloud deployment
- ✅ Bonus: Polynomial regression & feature experiments

---

## 📊 Dataset Information

**Dataset:** `Student_Performance.csv` (10,000 records)

**Features (5 input variables):**
1. **Hours Studied** - Weekly study hours (0-10)
2. **Previous Scores** - Historical exam scores (0-100)
3. **Extracurricular Activities** - Participation (Yes/No)
4. **Sleep Hours** - Daily sleep hours (0-12)
5. **Sample Question Papers Practiced** - Papers attempted (0-20)

**Target Variable:**
- **Performance Index** - Final exam score to predict (0-100)

**Data Split:**
- Training set: 8,000 samples (80%)
- Testing set: 2,000 samples (20%)

---

## 📊 Dataset
**Dataset Name:** Student Performance Factors  
**Source:** [Kaggle - Student Performance Factors](https://www.kaggle.com/datasets/nikhil7280/student-performance-factors)

**Key Features:**
- `study_hours`: Number of hours spent studying
- `sleep_hours`: Number of hours of sleep
- `participation`: Classroom participation score
- `previous_scores`: Previous exam scores
- `target`: Final exam score (what we're predicting)

**Dataset Size:** Typically 100-1000+ student records

---

## 🛠️ Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Core programming language |
| **Pandas** | 2.0.3 | Data manipulation and analysis |
| **NumPy** | 1.24.3 | Numerical computations |
| **Scikit-learn** | 1.3.0 | Machine learning model |
| **Matplotlib** | 3.7.2 | Static data visualization |
| **Seaborn** | 0.12.2 | Statistical visualization |
| **Streamlit** | 1.28.1 | Web application framework |
| **Joblib** | 1.3.2 | Model serialization |
| **Jupyter** | 1.0.0 | Interactive notebook environment |

---

## 🛠️ Tools & Libraries
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Core programming language |
| Pandas | 2.0.3 | Data manipulation and analysis |
| NumPy | 1.24.3 | Numerical computations |
| Matplotlib | 3.7.2 | Data visualization |
| Seaborn | 0.12.2 | Statistical visualization |
| Scikit-learn | 1.3.0 | Machine learning models |
| Jupyter | 1.0.0 | Interactive notebooks |

---

## 📁 Project Structure (Clean & Minimal)

```
Task_1_Student_Score_Prediction/
│
├── 🎯 app.py                           Main Streamlit web application
├── 📚 README.md                        Project documentation (this file)
├── 📋 requirements.txt                 All Python dependencies
├── .gitignore                          Git configuration
│
├── 📊 Student_Performance.csv          Main dataset (10,000 records)
├── 📊 sample_batch_input.csv          Test data for batch predictions (20 students)
│
├── 🤖 model/
│   ├── linear_model.pkl               Trained Linear Regression model
│   └── scaler.pkl                     StandardScaler object
│
├── 📓 notebooks/
│   └── Task_1_Student_Score_Prediction.ipynb    Full analysis notebook
│
├── .streamlit/
│   └── config.toml                    Streamlit app configuration & theme
│
└── 📈 outputs/
    ├── distribution.png               Distribution and box plots
    ├── correlation.png                Feature correlation heatmap
    ├── actual_vs_predicted.png         Model predictions visualization
    ├── feature_importance.png          Feature coefficient analysis
    ├── residuals.png                   Model residuals analysis
    ├── polynomial_comparison.png       Bonus: Polynomial regression comparison
    ├── feature_combinations.png        Bonus: Feature experiments
    └── summary.txt                     Final summary report
```

**Total:** 20 files, 5 folders | Size: Optimized | Status: Production Ready

---

## 🚀 Quick Start Guide

### ⚡ Option 1: Run Streamlit App (Recommended - 30 seconds)

**Step 1:** Install dependencies
```bash
pip install -r requirements.txt
```

**Step 2:** Start the Streamlit app
```bash
streamlit run app.py
```

**Step 3:** Open your browser
```
Local URL: http://localhost:8501
Network URL: http://10.10.86.124:8501
```

**What you can do:**
- 🎯 Make single predictions with interactive sliders
- 📊 Batch upload CSV files for multiple predictions
- 📈 View model insights and feature importance
- 📥 Download prediction results as CSV

---

### 📓 Option 2: Jupyter Notebook

```bash
jupyter notebook notebooks/Task_1_Student_Score_Prediction.ipynb
```

Run cells to see:
- Data exploration and visualization
- Model training step-by-step
- Performance metrics and evaluation
- Bonus: Polynomial regression & feature experiments

---

### 🌐 Option 3: Deploy to Cloud (Streamlit Cloud)

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Click "New app" → Select repository → Deploy
4. Share your public URL!

See the notebook for detailed deployment instructions.

---

## 🚀 Quick Start Guide

### ⚡ Fast Start (2 minutes)

**Step 1:** Install dependencies
```bash
pip install -r requirements.txt
```

**Step 2:** Start the app
```bash
streamlit run app.py
```

**Step 3:** Open your browser
- App opens at: `http://localhost:8501`
- Make predictions, upload batch data, view model insights

👉 See `QUICKSTART.md` for more details

### Option 2: Jupyter Notebook

```bash
jupyter notebook notebooks/Task_1_Student_Score_Prediction.ipynb
```

### Option 3: Deploy to Streamlit Cloud

See `STREAMLIT_DEPLOYMENT.md` for step-by-step cloud deployment instructions.

---

## � Model Performance

### Test Results (on 2,000 unseen samples)

| Metric | Train | Test | Interpretation |
|--------|-------|------|-----------------|
| **R² Score** | 0.9558 | **0.9557** | 95.57% of variance explained ✅ |
| **RMSE** | 2.8137 | **2.8037** | Average prediction error: ±2.80 points |
| **MAE** | 2.0898 | **2.0865** | Average absolute error: ±2.09 points |

**Conclusion:** Excellent model fit! No overfitting detected. Model generalizes well to new data.

---

## 🎯 Streamlit App Features

### 🎯 Tab 1: Single Prediction
- Interactive sliders for all 5 input features
- Real-time predictions (<100ms)
- Performance level badges (Excellent/Good/Average/Below Average)
- Input summary table

### 📊 Tab 2: Batch Prediction
- Upload CSV file with multiple students
- Bulk predictions for all records
- Statistics (average, min, max, std deviation)
- Download results as CSV

### 📈 Tab 3: Model Insights
- Performance metrics display
- Feature importance visualization
- Feature information and ranges
- How the model works explanation

---

## �📈 Expected Workflow

### Step 1: Data Exploration
- Load and inspect dataset shape, data types, and missing values
- Calculate descriptive statistics
- Identify correlations between features and target

### Step 2: Data Cleaning & Preprocessing
- Handle missing values (drop, fill, or impute)
- Check for outliers
- Normalize/standardize features if necessary
- Create train-test split (80-20)

### Step 3: Model Development
- Train Linear Regression model on training data
- Make predictions on test data
- Review initial performance metrics

### Step 4: Evaluation & Visualization
- Calculate R² score, MAE, RMSE
- Plot actual vs predicted values
- Visualize residuals (prediction errors)
- Identify model strengths and weaknesses

### Step 5: Bonus Tasks (Optional)
- Implement **Polynomial Regression** with different degrees
- Compare performance between linear and polynomial models
- Experiment with **different feature combinations**:
  - Remove features one by one (feature importance)
  - Add interaction terms
  - Create new derived features

---

## 📊 Key Evaluation Metrics Explained

| Metric | Formula | Interpretation | Our Model |
|--------|---------|-----------------|-----------|
| **R² Score** | 1 - (SS_res / SS_tot) | Proportion of variance explained (0-1, higher is better) | **0.9557** ✅ |
| **MAE** | Mean(\|y_actual - y_pred\|) | Average absolute error (in same units as target) | **2.09** |
| **RMSE** | √(Mean((y_actual - y_pred)²)) | Square root of average squared error (penalizes large errors) | **2.80** |
| **MAPE** | Mean(\|(y_actual - y_pred) / y_actual\|) | Mean absolute percentage error | ~2.8% |

---

## 💡 Key Insights from Analysis

**Most Influential Features:**
1. **Previous Scores** - Strong positive correlation (historical performance predicts future)
2. **Hours Studied** - Strong positive correlation (more study → better scores)
3. **Sample Question Papers Practiced** - Positive correlation (practice improves performance)
4. **Sleep Hours** - Positive correlation (adequate sleep improves performance)
5. **Extracurricular Activities** - Slight positive correlation

**Model Characteristics:**
- Linear relationship between features and performance
- No significant outliers affecting predictions
- Residuals normally distributed (good model fit)
- Consistent performance on train and test sets

---

## 📊 Key Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **R² Score** | 1 - (SS_res / SS_tot) | Proportion of variance explained (0-1, higher is better) |
| **MAE** | Mean(∣y_actual - y_pred∣) | Average absolute error (in same units as target) |
| **RMSE** | √(Mean((y_actual - y_pred)²)) | Square root of average squared error (penalizes large errors) |
| **MAPE** | Mean(∣(y_actual - y_pred) / y_actual∣) | Mean absolute percentage error |

---

## 💡 Tips & Best Practices
- Always explore data before modeling (EDA)
- Check for multicollinearity using correlation matrix
- Validate assumptions of linear regression (linearity, homoscedasticity)
- Use cross-validation for more robust evaluation
- Document your findings and reasoning
- Create meaningful visualizations
- Compare models fairly (same train-test split)

---

## 📚 Learning Resources
- [Scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Pandas Data Cleaning Guide](https://pandas.pydata.org/docs/)
- [Matplotlib Visualization Guide](https://matplotlib.org/)

---

## ✅ Project Completion Status

### Core Requirements
- [x] Load and explore the Student Performance dataset
- [x] Perform data cleaning and preprocessing
- [x] Create visualizations (distribution, correlation, residuals)
- [x] Split data into training and testing sets (80-20)
- [x] Train Linear Regression model
- [x] Evaluate model (R², MAE, RMSE)
- [x] Visualize predictions vs actual values
- [x] Save model for future use

### Bonus Features
- [x] Implement Polynomial Regression comparison
- [x] Feature experimentation (removing features)
- [x] Generate comprehensive analysis notebook
- [x] Create interactive Streamlit web application
- [x] Batch prediction functionality
- [x] Cloud deployment ready

### Deliverables
- [x] Production-ready Python code
- [x] Trained ML model (95.57% accuracy)
- [x] Web application (3 interactive tabs)
- [x] Complete documentation
- [x] Sample test data included
- [x] Clean, organized project structure

---

## ✅ Completion Checklist
- [x] Dataset downloaded and loaded
- [x] Data exploration completed (visualizations, statistics)
- [x] Data cleaning and preprocessing done
- [x] Train-test split implemented
- [x] Linear regression model trained
- [x] Evaluation metrics calculated (R², MAE, RMSE)
- [x] Predictions visualized
- [x] Residual analysis performed
- [x] Bonus tasks attempted (polynomial regression, feature experimentation)
- [x] Results documented
- [x] Code well-commented and organized

---

## 🎓 Learning Outcomes

By working on this project, you will understand:

✅ **Machine Learning Fundamentals**
- Linear regression theory and practice
- Train-test split and model validation
- Evaluation metrics (R², RMSE, MAE)

✅ **Data Science Skills**
- Data exploration and visualization
- Data preprocessing and cleaning
- Feature analysis and importance

✅ **Python Development**
- Pandas and NumPy for data manipulation
- Scikit-learn for ML models
- Matplotlib and Seaborn for visualization
- Streamlit for web applications

✅ **Model Deployment**
- Saving and loading trained models
- Creating interactive web applications
- Batch processing and predictions
- Cloud deployment strategies

---

## 🎓 Learning Outcomes
By completing this task, you will have gained knowledge of:
✅ Data exploration and visualization  
✅ Data preprocessing and cleaning  
✅ Train-test data splitting  
✅ Linear regression fundamentals  
✅ Model evaluation metrics  
✅ Prediction visualization  
✅ Model comparison techniques  

---

## 📞 Support & Resources

**Need help?**
- Check the Jupyter notebook for step-by-step analysis
- Run the Streamlit app to see predictions in action
- Review the outputs folder for visualizations
- Check requirements.txt for all dependencies

**Model Files:**
- `model/linear_model.pkl` - Trained Linear Regression model
- `model/scaler.pkl` - StandardScaler for feature normalization

**Test Data:**
- `sample_batch_input.csv` - 20 sample students for testing batch predictions

---

**Status:** ✅ Complete and Production Ready  
**Accuracy:** 95.57% R² Score  
**Last Updated:** October 20, 2025  
**Author:** Elevvo Internship Program

---

**Status:** Not Started  
**Last Updated:** October 20, 2025  
**Author:** Elevvo Internship Program
