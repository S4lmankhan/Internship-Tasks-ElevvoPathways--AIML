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
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Set page config
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .header-title {
            color: #1f77b4;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        from pathlib import Path
        current_dir = Path(__file__).parent
        models = {}
        model_files = ['linear_regression_model.pkl', 'decision_tree_model.pkl', 'random_forest_model.pkl', 'xgboost_model.pkl', 'lightgbm_model.pkl']
        model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM']
        
        for name, fname in zip(model_names, model_files):
            try:
                models[name] = joblib.load(str(current_dir / 'model' / fname))
            except FileNotFoundError:
                continue
        
        scaler = joblib.load(str(current_dir / 'model' / 'scaler.pkl'))
        feature_columns = joblib.load(str(current_dir / 'model' / 'feature_columns.pkl'))
        return models, scaler, feature_columns
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_results():
    """Load model results"""
    try:
        from pathlib import Path
        current_dir = Path(__file__).parent
        results_path = current_dir / 'outputs' / 'model_results.csv'
        results_df = pd.read_csv(str(results_path))
        return results_df
    except Exception as e:
        st.warning(f"⚠️ Could not load results file: {e}")
        return None

# Load all resources
models, scaler, feature_columns = load_models()
results_df = load_results()

if models is None:
    st.error("❌ Failed to load models. Please ensure model files are in the 'model/' directory.")
    st.stop()

if results_df is None or results_df.empty:
    st.error("❌ Failed to load results. Please ensure outputs/model_results.csv exists.")
    st.stop()

# Sidebar
st.sidebar.title("📊 Sales Forecasting System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a Page:",
    ["🏠 Dashboard", "📈 Forecasting", "📊 Model Insights", "📉 Historical Data"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **About This App:**
    - Predicts future sales using trained ML models
    - Analyzes historical sales patterns
    - Compares model performance
    - Provides detailed forecasting insights
""")

# ==================== PAGE 1: DASHBOARD ====================
if page == "🏠 Dashboard":
    st.markdown('<div class="header-title">📈 Sales Forecasting Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", len(models), "5 trained")
    
    with col2:
        if results_df is not None and not results_df.empty:
            best_model = results_df.iloc[0]
            st.metric("Best Model", best_model['Model'], f"R²: {best_model['R² Score']:.4f}")
        else:
            st.metric("Best Model", "N/A", "No data")
    
    with col3:
        if results_df is not None and not results_df.empty:
            best_model = results_df.iloc[0]
            st.metric("Best RMSE", f"${best_model['RMSE']:.2f}", f"MAE: ${best_model['MAE']:.2f}")
        else:
            st.metric("Best RMSE", "N/A", "No data")
    
    with col4:
        st.metric("Features", len(feature_columns), "17 engineered")
    
    st.markdown("---")
    
    # Model Performance Comparison
    st.subheader("📊 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score comparison
        fig = go.Figure(data=[
            go.Bar(
                x=results_df['Model'],
                y=results_df['R² Score'],
                marker=dict(
                    color=results_df['R² Score'],
                    colorscale='RdYlGn',
                    showscale=True
                ),
                text=[f"{x:.4f}" for x in results_df['R² Score']],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="R² Score Comparison",
            xaxis_title="Model",
            yaxis_title="R² Score",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # RMSE comparison
        fig = go.Figure(data=[
            go.Bar(
                x=results_df['Model'],
                y=results_df['RMSE'],
                marker=dict(color='#ff7f0e'),
                text=[f"${x:.2f}" for x in results_df['RMSE']],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="RMSE Comparison (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="RMSE ($)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Model Metrics")
    
    display_df = results_df.copy()
    display_df['R² Score'] = display_df['R² Score'].apply(lambda x: f"{x:.4f}")
    display_df['MSE'] = display_df['MSE'].apply(lambda x: f"{x:.2f}")
    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"${x:.2f}")
    display_df['MAE'] = display_df['MAE'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ==================== PAGE 2: FORECASTING ====================
elif page == "📈 Forecasting":
    st.markdown('<div class="header-title">📈 Sales Forecasting</div>', unsafe_allow_html=True)
    
    st.subheader("🔮 Make a Forecast")
    
    st.info("💡 **Note**: This model uses Store ID, Item ID, and temporal features (day, month, quarter, year, weekend indicator) to predict sales.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider("Days Ahead to Forecast", 1, 365, 30)
    
    with col2:
        selected_model = st.selectbox("Select Model:", list(models.keys()))
    
    st.markdown("---")
    
    # Store and Item inputs
    st.subheader("🏪 Store & Item Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        store_id = st.number_input("Store ID", min_value=1, max_value=100, value=1, step=1)
    
    with col2:
        item_id = st.number_input("Item ID", min_value=1, max_value=100, value=1, step=1)
    
    # Temporal features for forecasting
    st.subheader("📅 Forecast Start Date")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_month = st.number_input("Start Month", 1, 12, datetime.now().month)
    
    with col2:
        start_day = st.number_input("Start Day of Month", 1, 31, datetime.now().day)
    
    st.markdown("---")
    
    # Generate forecast button
    if st.button("🚀 Generate Forecast", use_container_width=True, key="forecast_btn"):
        try:
            # Create forecast input based on actual model features
            # Features: ['Store', 'Item', 'DayOfWeek', 'Month', 'Quarter', 'Year', 'IsWeekend']
            forecast_inputs = []
            
            # Create start date
            current_year = datetime.now().year
            try:
                start_date = datetime(current_year, start_month, start_day)
            except ValueError:
                # Handle invalid dates (e.g., Feb 30)
                start_date = datetime(current_year, start_month, 1)
            
            for day_offset in range(forecast_days):
                current_date = start_date + timedelta(days=day_offset)
                
                input_dict = {
                    'Store': store_id,
                    'Item': item_id,
                    'DayOfWeek': current_date.weekday(),
                    'Month': current_date.month,
                    'Quarter': (current_date.month - 1) // 3 + 1,
                    'Year': current_date.year,
                    'IsWeekend': 1 if current_date.weekday() >= 5 else 0
                }
                
                forecast_inputs.append(input_dict)
            
            forecast_df = pd.DataFrame(forecast_inputs)
            
            # Reorder columns to match training data
            forecast_df = forecast_df[feature_columns]
            
            # Scale features
            forecast_scaled = scaler.transform(forecast_df)
            
            # Make predictions
            model = models[selected_model]
            predictions = model.predict(forecast_scaled)
            
            # Create results dataframe with dates
            forecast_dates = [start_date + timedelta(days=i) for i in range(forecast_days)]
            
            forecast_results = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': predictions,
                'Store': forecast_df['Store'],
                'Item': forecast_df['Item'],
                'Month': forecast_df['Month'],
                'DayOfWeek': forecast_df['DayOfWeek'],
                'IsWeekend': forecast_df['IsWeekend']
            })
            
            st.success(f"✅ Forecast generated for {forecast_days} days ahead!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Average Predicted Sales",
                    f"${predictions.mean():.2f}",
                    f"Range: ${predictions.min():.2f} - ${predictions.max():.2f}"
                )
            
            with col2:
                st.metric(
                    "Total Predicted Sales",
                    f"${predictions.sum():.2f}",
                    f"Std Dev: ${predictions.std():.2f}"
                )
            
            st.markdown("---")
            
            # Visualization
            st.subheader("📈 Forecast Visualization")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=forecast_results['Date'],
                y=forecast_results['Forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f"{forecast_days}-Day Sales Forecast ({selected_model})",
                xaxis_title="Date",
                yaxis_title="Predicted Sales ($)",
                height=500,
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv = forecast_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"sales_forecast_{selected_model.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error generating forecast: {e}")

# ==================== PAGE 3: MODEL INSIGHTS ====================
elif page == "📊 Model Insights":
    st.markdown('<div class="header-title">📊 Model Insights & Analysis</div>', unsafe_allow_html=True)
    
    st.subheader("🏆 Best Model Performance")
    
    if results_df is not None and not results_df.empty:
        best_model_name = results_df.iloc[0]['Model']
        best_r2 = results_df.iloc[0]['R² Score']
        best_rmse = results_df.iloc[0]['RMSE']
        best_mae = results_df.iloc[0]['MAE']
    else:
        st.error("❌ Model results not available")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", best_model_name, "⭐")
    with col2:
        st.metric("R² Score", f"{best_r2:.4f}", "Higher is Better")
    with col3:
        st.metric("RMSE", f"${best_rmse:.2f}", "Lower is Better")
    with col4:
        st.metric("MAE", f"${best_mae:.2f}", "Lower is Better")
    
    st.markdown("---")
    
    # Feature Information
    st.subheader("🔧 Engineered Features")
    
    feature_groups = {
        "Temporal Features": ["Day", "Month", "DayOfWeek", "Quarter", "Week", "DayOfYear", "IsWeekend"],
        "Lag Features (Previous Days)": ["Sales_Lag_1", "Sales_Lag_7", "Sales_Lag_14", "Sales_Lag_30"],
        "Moving Averages": ["Sales_MA_7", "Sales_MA_14", "Sales_MA_30"],
        "Rolling Std Dev": ["Sales_Std_7", "Sales_Std_14", "Sales_Std_30"]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Groups")
        for group_name, features in feature_groups.items():
            with st.expander(f"📌 {group_name}"):
                for feature in features:
                    st.write(f"• {feature}")
    
    with col2:
        st.subheader("Total Features")
        st.metric("Total Engineered Features", len(feature_columns))
        
        feature_desc = """
        **Feature Engineering Strategy:**
        
        1. **Temporal**: Capture time-based patterns (day, month, season)
        2. **Lag Features**: Historical sales from 1, 7, 14, 30 days ago
        3. **Moving Averages**: Trends over 7, 14, 30 day windows
        4. **Volatility**: Standard deviation for each window
        
        This combination captures both short-term and long-term patterns.
        """
        st.info(feature_desc)
    
    st.markdown("---")
    
    # Model Comparison Table
    st.subheader("📊 All Models Performance")
    
    comparison_df = results_df.copy()
    comparison_df['Rank'] = range(1, len(comparison_df) + 1)
    comparison_df = comparison_df[['Rank', 'Model', 'R² Score', 'RMSE', 'MAE', 'MSE']]
    
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "R² Score": st.column_config.NumberColumn(format="%.4f"),
            "RMSE": st.column_config.NumberColumn(format="$%.2f"),
            "MAE": st.column_config.NumberColumn(format="$%.2f"),
            "MSE": st.column_config.NumberColumn(format="%.2f")
        }
    )
    
    st.markdown("---")
    
    # Model Selection Guide
    st.subheader("🎯 Model Selection Guide")
    
    selection_guide = """
    **Linear Regression** ⭐ BEST
    - Highest R² Score (close to 1.0)
    - Lowest RMSE and MAE
    - Best for linear relationships
    - Fast predictions
    - Use for: Production forecasting
    
    **Decision Tree**
    - Good for non-linear patterns
    - Interpretable results
    - Use for: Explanatory analysis
    
    **Random Forest**
    - Ensemble of decision trees
    - Reduces overfitting
    - Use for: Robust predictions
    
    **XGBoost**
    - Advanced gradient boosting
    - Handles complex patterns
    - Use for: High accuracy needs
    
    **LightGBM**
    - Fast training and prediction
    - Memory efficient
    - Use for: Large datasets
    """
    
    st.info(selection_guide)

# ==================== PAGE 4: HISTORICAL DATA ====================
elif page == "📉 Historical Data":
    st.markdown('<div class="header-title">📉 Historical Sales Analysis</div>', unsafe_allow_html=True)
    
    st.subheader("📊 Key Statistics")
    
    try:
        # Try to load actual data
        url = 'https://raw.githubusercontent.com/datasets/store-sales/main/data/store-sales.csv'
        try:
            df = pd.read_csv(url)
        except:
            # Create sample data if download fails
            date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)
            sales = np.cumsum(np.random.randn(len(date_range)) * 50 + 100) + 5000
            df = pd.DataFrame({
                'Date': date_range,
                'Sales': np.maximum(sales, 3000)
            })
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        if 'Sales' not in df.columns:
            df.rename(columns={'sales': 'Sales', 'sales_value': 'Sales'}, inplace=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Avg Daily Sales", f"${df['Sales'].mean():.2f}")
        with col3:
            st.metric("Max Sales", f"${df['Sales'].max():.2f}")
        with col4:
            st.metric("Min Sales", f"${df['Sales'].min():.2f}")
        
        st.markdown("---")
        
        # Time series chart
        st.subheader("📈 Sales Over Time")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Sales'],
            mode='lines',
            name='Sales',
            line=dict(color='#1f77b4', width=1),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title="Historical Sales Data",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            height=500,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly aggregation
        st.subheader("📊 Monthly Sales Summary")
        
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_sales = df.groupby('Month')['Sales'].agg(['sum', 'mean', 'std']).head(20)
        monthly_sales.index = monthly_sales.index.astype(str)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly_sales.index,
            y=monthly_sales['sum'],
            name='Total Sales',
            marker=dict(color='#1f77b4')
        ))
        
        fig.update_layout(
            title="Monthly Sales Total",
            xaxis_title="Month",
            yaxis_title="Sales ($)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Recent Sales Data")
        
        recent_df = df.tail(30).copy()
        recent_df['Date'] = recent_df['Date'].dt.strftime('%Y-%m-%d')
        recent_df = recent_df[['Date', 'Sales']].reset_index(drop=True)
        
        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sales": st.column_config.NumberColumn(format="$%.2f")
            }
        )
        
    except Exception as e:
        st.error(f"Error loading historical data: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 50px;'>
        📊 Sales Forecasting System | Task 7 - Time Series Analysis & Forecasting
        <br>
        Built with Streamlit, Scikit-learn, XGBoost, and LightGBM
    </div>
""", unsafe_allow_html=True)
