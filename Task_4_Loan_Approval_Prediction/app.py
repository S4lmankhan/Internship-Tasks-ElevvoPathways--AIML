import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Loan Approval", page_icon="💳", layout="wide")

st.markdown("""<style>.main {background-color: #f8f9fa;}</style>""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        lr = joblib.load('model/logistic_regression_model.pkl')
        dt = joblib.load('model/decision_tree_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        le_dict = joblib.load('model/label_encoders.pkl')
        le_target = joblib.load('model/target_encoder.pkl')
        features = joblib.load('model/feature_columns.pkl')
        return {'lr': lr, 'dt': dt, 'scaler': scaler, 'le_dict': le_dict, 'le_target': le_target, 'features': features}
    except:
        st.error("❌ Models not found! Run notebook first.")
        st.stop()

models = load_models()

st.title("💳 Loan Approval Prediction")
st.markdown("AI-powered binary classification to predict loan approval status")

# Sidebar
st.sidebar.header("📊 Model Info")
st.sidebar.info("""
**Best Model:** Decision Tree
**F1-Score:** 0.3243  
**Data:** 300 applications  
**SMOTE:** Applied for balance
""")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Single", "📊 Batch", "📈 Insights"])

# ========== TAB 1: SINGLE PREDICTION ==========
with tab1:
    st.header("🎯 Single Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 80, 35)
        income = st.slider("Income ($)", 20000, 150000, 60000, 5000)
        credit = st.slider("Credit Score", 300, 850, 700, 10)
        emp_yrs = st.slider("Employment Years", 0, 50, 5)
    with col2:
        loan_amt = st.slider("Loan Amount ($)", 5000, 500000, 100000, 10000)
        gender = st.selectbox("Gender", ["M", "F"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.slider("Dependents", 0, 4, 0)
    
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["Yes", "No"])
    
    if st.button("🔮 Predict", key="single"):
        try:
            input_data = pd.DataFrame({
                'Age': [age], 'Income': [income], 'Credit_Score': [credit],
                'Employment_Years': [emp_yrs], 'Loan_Amount': [loan_amt],
                'Gender': [gender], 'Married': [married], 'Dependents': [dependents],
                'Education': [education], 'Self_Employed': [self_emp]
            })
            
            input_enc = input_data.copy()
            for col in models['le_dict']:
                if col in input_enc.columns:
                    input_enc[col] = models['le_dict'][col].transform(input_enc[col].astype(str))
            
            input_enc = input_enc[models['features']]
            input_scaled = models['scaler'].transform(input_enc)
            
            lr_pred = models['lr'].predict(input_scaled)[0]
            lr_prob = models['lr'].predict_proba(input_scaled)[0]
            dt_pred = models['dt'].predict(input_scaled)[0]
            dt_prob = models['dt'].predict_proba(input_scaled)[0]
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Logistic Regression", models['le_target'].classes_[lr_pred], f"{lr_prob[lr_pred]:.1%}")
            with col2:
                st.metric("Decision Tree ⭐", models['le_target'].classes_[dt_pred], f"{dt_prob[dt_pred]:.1%}")
            with col3:
                ensemble = 1 if (lr_pred + dt_pred) >= 1 else 0
                st.metric("Ensemble", models['le_target'].classes_[ensemble], "Vote")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for idx, (prob, title) in enumerate([(lr_prob, 'LR'), (dt_prob, 'DT')]):
                axes[idx].bar(models['le_target'].classes_, prob, color=['#FF6B6B', '#4ECDC4'])
                axes[idx].set_ylim([0, 1])
                for i, v in enumerate(prob):
                    axes[idx].text(i, v+0.02, f'{v:.1%}', ha='center', fontweight='bold')
                axes[idx].set_title(title)
            plt.tight_layout()
            st.pyplot(fig)
            
            if models['le_target'].classes_[dt_pred] == 'Approved':
                st.success("✅ **APPROVED** - Applicant meets criteria")
            else:
                st.warning("❌ **REJECTED** - Need better credentials")
        except Exception as e:
            st.error(f"Error: {e}")

# ========== TAB 2: BATCH ==========
with tab2:
    st.header("📊 Batch Predictions")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_batch)} records")
        st.dataframe(df_batch.head(), use_container_width=True)
        
        if st.button("🔮 Predict All", key="batch"):
            try:
                df_pred = df_batch.copy()
                for col in models['le_dict']:
                    if col in df_pred.columns:
                        df_pred[col] = models['le_dict'][col].transform(df_pred[col].astype(str))
                
                df_pred = df_pred[models['features']]
                df_scaled = models['scaler'].transform(df_pred)
                
                lr_preds = models['lr'].predict(df_scaled)
                lr_probas = models['lr'].predict_proba(df_scaled)[:, 1]
                dt_preds = models['dt'].predict(df_scaled)
                dt_probas = models['dt'].predict_proba(df_scaled)[:, 1]
                
                results = pd.DataFrame({
                    'LR_Pred': [models['le_target'].classes_[p] for p in lr_preds],
                    'LR_Conf': lr_probas,
                    'DT_Pred': [models['le_target'].classes_[p] for p in dt_preds],
                    'DT_Conf': dt_probas,
                    'Ensemble': ['Approved' if (lr_preds[i] + dt_preds[i])/2 >= 0.5 else 'Rejected' for i in range(len(df_batch))]
                })
                
                results_final = pd.concat([df_batch, results], axis=1)
                st.dataframe(results_final, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Approved", (results['DT_Pred']=='Approved').sum())
                with col2:
                    st.metric("Rejected", (results['DT_Pred']=='Rejected').sum())
                with col3:
                    st.metric("Avg Confidence", f"{results['DT_Conf'].mean():.1%}")
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                vals = results['DT_Pred'].value_counts()
                axes[0].pie(vals.values, labels=vals.index, autopct='%1.1f%%', colors=['#4ECDC4', '#FF6B6B'])
                axes[0].set_title('Predictions')
                
                axes[1].hist(results['DT_Conf'], bins=20, color='#4ECDC4', edgecolor='black')
                axes[1].set_xlabel('Confidence')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Confidence Distribution')
                plt.tight_layout()
                st.pyplot(fig)
                
                csv = results_final.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Upload a CSV file to predict multiple loans")

# ========== TAB 3: INSIGHTS ==========
with tab3:
    st.header("📈 Model Insights")
    
    metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'LR': [0.2667, 0.1389, 0.2778, 0.1852, 0.2937],
        'DT': [0.5833, 0.3158, 0.3333, 0.3243, 0.4749]
    })
    st.dataframe(metrics, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, metrics['LR'], width, label='LR', color='#FF6B6B')
    ax.bar(x + width/2, metrics['DT'], width, label='DT', color='#4ECDC4')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics['Metric'])
    ax.legend()
    ax.set_ylim([0, 1])
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🔍 Feature Importance")
    
    importance = pd.DataFrame({
        'Feature': ['Income', 'Dependents', 'Age', 'Employment', 'Loan', 'Education', 'Credit', 'Married', 'Gender', 'Self-Emp'],
        'Score': [0.233, 0.217, 0.170, 0.088, 0.067, 0.063, 0.057, 0.054, 0.037, 0.014]
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance['Feature'], importance['Score'], color='#4ECDC4')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Decision Tree)')
    ax.invert_yaxis()
    st.pyplot(fig)
    
    st.success("**Most Important:** Income (23.3%)")
    st.info("**Dataset:** 300 samples, 70-30 imbalance, SMOTE applied")

st.markdown("---")
st.markdown("<div style='text-align:center'><p>💳 Loan Approval System | Oct 2025</p></div>", unsafe_allow_html=True)
