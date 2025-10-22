import sys

# AGGRESSIVE FIX: Pre-map numpy._core BEFORE ANY imports
import numpy
if not hasattr(numpy, '_core'):
    import numpy.core
    numpy._core = numpy.core
    sys.modules['numpy._core'] = numpy.core

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

st.set_page_config(page_title="Customer Segmentation", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #1f77b4;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    try:
        from pathlib import Path
        current_dir = Path(__file__).parent
        kmeans = joblib.load(str(current_dir / 'model' / 'kmeans_model.pkl'))
        scaler = joblib.load(str(current_dir / 'model' / 'scaler.pkl'))
        return kmeans, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the notebook first to generate kmeans_model.pkl and scaler.pkl")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

kmeans, scaler = load_model_and_scaler()

st.title("🎯 Customer Segmentation Dashboard")
st.markdown("AI-powered customer segmentation using K-Means clustering")

if kmeans is None or scaler is None:
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Information")
st.sidebar.info(f"""
**Clustering Algorithm:** K-Means
**Number of Clusters:** 5
**Features Used:**
- Annual Income
- Spending Score
""")

tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Upload", "📈 Model Insights"])

with tab1:
    st.header("Single Customer Segment Prediction")
    st.write("Enter customer details to assign to a segment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.slider("Annual Income (k$)", min_value=15, max_value=140, value=50, step=1)
    
    with col2:
        spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)
    
    if st.button("Predict Cluster", key="single_predict", use_container_width=True):
        customer_data = np.array([[income, spending_score]])
        cluster = kmeans.predict(customer_data)[0]
        
        st.success(f"✅ Customer assigned to **Cluster {cluster}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Income", f"${income}k")
        with col2:
            st.metric("Spending Score", spending_score)
        with col3:
            st.metric("Assigned Cluster", cluster)
        
        cluster_names = {
            0: "� Mid-Range Buyers (Medium Income & Spending)",
            1: "💼 Budget-Conscious (Low Income & Low Spending)",
            2: "🎯 High Income - Low Spenders (Savers)",
            3: "⭐ Premium Customers (High Income & High Spending)"
        }
        
        st.info(f"**Segment Type:** {cluster_names.get(cluster, 'Unknown Segment')}")

with tab2:
    st.header("Batch Customer Prediction")
    st.write("Upload a CSV file with multiple customers")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            required_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
            if not all(col in batch_df.columns for col in required_cols):
                st.error(f"❌ CSV must contain columns: {', '.join(required_cols)}")
            else:
                batch_data = batch_df[required_cols].copy()
                batch_scaled = scaler.transform(batch_data)
                clusters = kmeans.predict(batch_scaled)
                
                batch_df['Cluster'] = clusters
                
                st.success(f"✅ Processed {len(batch_df)} customers")
                st.dataframe(batch_df, use_container_width=True)
                
                csv_result = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results with Clusters",
                    data=csv_result,
                    file_name="customer_segments.csv",
                    mime="text/csv"
                )
                
                st.subheader("📊 Cluster Distribution")
                cluster_counts = batch_df['Cluster'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(cluster_counts.index, cluster_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
                ax.set_xlabel('Cluster', fontsize=12)
                ax.set_ylabel('Number of Customers', fontsize=12)
                ax.set_title('Customer Distribution Across Segments', fontsize=13)
                ax.grid(alpha=0.3, axis='y')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

with tab3:
    st.header("Model Insights & Cluster Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Clusters", "5")
    with col2:
        st.metric("Features Used", "2")
    with col3:
        st.metric("Algorithm", "K-Means")
    
    st.subheader("📍 Cluster Centers")
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(
        centers,
        columns=['Annual Income (k$)', 'Spending Score (1-100)'],
        index=[f'Cluster {i}' for i in range(kmeans.n_clusters)]
    )
    st.dataframe(centers_df.round(2), use_container_width=True)
    
    st.subheader("💡 Cluster Descriptions")
    
    cluster_profiles = {
        0: "💼 **Budget-Conscious**: Lower income, lower spending scores. Conservative shoppers.",
        1: "⭐ **Premium Spenders**: Higher income, higher spending scores. VIP customers.",
        2: "🎯 **Mid-Range Buyers**: Mid-to-high income, moderate spending. Balanced approach.",
        3: "🚀 **High-Value Prospects**: High income but lower spending. Untapped potential.",
        4: "💰 **Moderate Spenders**: Moderate income and spending. Regular customers."
    }
    
    for cluster, description in cluster_profiles.items():
        st.info(f"**Cluster {cluster}:** {description}")
    
    st.subheader("🔧 Model Specifications")
    st.text(f"""
Initialization Method: K-Means++ (smart center selection)
Distance Metric: Euclidean Distance
Feature Scaling: StandardScaler (normalized)
Optimization: Convergence criterion met
    """)

st.markdown("---")
st.footer = st.markdown(
    "<div style='text-align: center; padding: 20px; color: #888;'>"
    "<p>Task 2: Customer Segmentation • K-Means Clustering • AI-Powered Analysis</p>"
    "</div>",
    unsafe_allow_html=True
)
