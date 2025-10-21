import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📚 Student Performance Prediction")
st.markdown("Predict student performance index based on study habits and academic history")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        """
        This app uses a **Linear Regression Model** trained on student performance data.
        
        Features used:
        - Hours Studied
        - Previous Scores
        - Extracurricular Activities
        - Sleep Hours
        - Sample Question Papers Practiced
        """
    )
    st.divider()
    st.header("Model Information")
    st.text("Model Type: Linear Regression")
    st.text("Test R² Score: 0.9557")
    st.text("Test RMSE: 2.8037")

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler"""
    current_dir = Path(__file__).parent
    model_path = current_dir / "model" / "linear_model.pkl"
    scaler_path = current_dir / "model" / "scaler.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        st.error("⚠️ Model files not found. Please ensure model/ directory contains the trained model.")
        st.stop()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction", "📈 Model Insights"])

# Tab 1: Single Prediction
with tab1:
    st.header("Make a Single Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Student Information")
        hours_studied = st.slider(
            "Hours Studied (per week)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        )
        
        previous_scores = st.slider(
            "Previous Scores",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )
        
        extracurricular = st.selectbox(
            "Extracurricular Activities",
            ["Yes", "No"]
        )
    
    with col2:
        st.subheader("Additional Details")
        sleep_hours = st.slider(
            "Sleep Hours (per day)",
            min_value=0.0,
            max_value=12.0,
            value=7.0,
            step=0.5
        )
        
        practice_papers = st.slider(
            "Sample Question Papers Practiced",
            min_value=0,
            max_value=20,
            value=3,
            step=1
        )
    
    # Convert extracurricular to numeric
    extracurricular_numeric = 1 if extracurricular == "Yes" else 0
    
    # Create prediction DataFrame
    if st.button("🔮 Predict Performance", key="single_predict", use_container_width=True):
        input_data = pd.DataFrame({
            'Hours Studied': [hours_studied],
            'Previous Scores': [previous_scores],
            'Extracurricular Activities': [extracurricular_numeric],
            'Sleep Hours': [sleep_hours],
            'Sample Question Papers Practiced': [practice_papers]
        })
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display results
        st.divider()
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Predicted Performance Index",
                value=f"{prediction:.2f}",
                delta=None
            )
        
        with col2:
            # Determine performance level
            if prediction >= 80:
                level = "🌟 Excellent"
                color = "green"
            elif prediction >= 70:
                level = "✅ Good"
                color = "blue"
            elif prediction >= 60:
                level = "⚠️ Average"
                color = "orange"
            else:
                level = "❌ Below Average"
                color = "red"
            
            st.metric(label="Performance Level", value=level)
        
        with col3:
            percentage = (prediction / 100) * 100
            st.metric(label="Score Percentage", value=f"{percentage:.1f}%")
        
        # Display input summary
        st.subheader("Input Summary")
        input_display = pd.DataFrame({
            'Feature': input_data.columns,
            'Value': input_data.values[0]
        })
        st.dataframe(input_display, use_container_width=True, hide_index=True)

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch Prediction")
    st.write("Upload a CSV file with student data to make multiple predictions")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV should contain columns: Hours Studied, Previous Scores, Extracurricular Activities (Yes/No), Sleep Hours, Sample Question Papers Practiced"
    )
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            
            # Display uploaded data
            st.subheader("Uploaded Data")
            st.dataframe(batch_data, use_container_width=True)
            
            if st.button("🔮 Make Batch Predictions", use_container_width=True):
                # Prepare data
                batch_copy = batch_data.copy()
                
                # Convert Extracurricular Activities to numeric
                if 'Extracurricular Activities' in batch_copy.columns:
                    batch_copy['Extracurricular Activities'] = (
                        batch_copy['Extracurricular Activities'] == 'Yes'
                    ).astype(int)
                
                # Ensure column order matches training data
                required_cols = [
                    'Hours Studied', 'Previous Scores', 'Extracurricular Activities',
                    'Sleep Hours', 'Sample Question Papers Practiced'
                ]
                batch_copy = batch_copy[required_cols]
                
                # Scale and predict
                batch_scaled = scaler.transform(batch_copy)
                predictions = model.predict(batch_scaled)
                
                # Add predictions to dataframe
                batch_data['Predicted Performance'] = predictions
                batch_data['Performance Level'] = batch_data['Predicted Performance'].apply(
                    lambda x: "🌟 Excellent" if x >= 80 else ("✅ Good" if x >= 70 else ("⚠️ Average" if x >= 60 else "❌ Below Average"))
                )
                
                # Display results
                st.subheader("Predictions")
                st.dataframe(batch_data, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Prediction", f"{predictions.mean():.2f}")
                with col2:
                    st.metric("Highest Score", f"{predictions.max():.2f}")
                with col3:
                    st.metric("Lowest Score", f"{predictions.min():.2f}")
                with col4:
                    st.metric("Std Deviation", f"{predictions.std():.2f}")
                
                # Download results
                csv = batch_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the correct column names and format.")

# Tab 3: Model Insights
with tab3:
    st.header("Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Performance Metrics")
        metrics_data = {
            'Metric': ['Train R² Score', 'Test R² Score', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE'],
            'Value': ['0.9558', '0.9557', '2.8137', '2.8037', '2.0898', '2.0865']
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🔍 Feature Information")
        feature_info = {
            'Feature': [
                'Hours Studied',
                'Previous Scores',
                'Extracurricular Activities',
                'Sleep Hours',
                'Practice Papers'
            ],
            'Range': [
                '0 - 10 hours',
                '0 - 100',
                'Yes / No',
                '0 - 12 hours',
                '0 - 20'
            ],
            'Impact': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐']
        }
        feature_df = pd.DataFrame(feature_info)
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("💡 Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(
            """
            **Algorithm**: Linear Regression
            
            Linear regression models the relationship between input features and the target variable as a linear combination of features.
            """
        )
    
    with col2:
        st.info(
            """
            **Training Data**: 8,000 samples
            
            **Testing Data**: 2,000 samples
            
            **Test/Train Split**: 80-20
            """
        )
    
    with col3:
        st.info(
            """
            **Preprocessing**: StandardScaler
            
            All numerical features are scaled to have mean=0 and std=1 for better model performance.
            """
        )
    
    st.divider()
    
    st.subheader("📈 How the Model Works")
    st.write(
        """
        1. **Input**: Student provides 5 input features (study hours, previous scores, etc.)
        2. **Scaling**: Features are scaled using the StandardScaler to normalize values
        3. **Prediction**: Linear regression model computes weighted sum of features
        4. **Output**: Performance Index (typically 0-100 scale)
        
        The model achieves ~95.5% accuracy (R² score) on test data, meaning it explains 95.5% of the variance in student performance.
        """
    )

# Footer
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.8rem; margin-top: 2rem;">
        📚 Student Performance Prediction Model | Developed for Elevvo Internship Task 1
    </div>
    """,
    unsafe_allow_html=True
)
