# Task 2: Customer Segmentation

## 📋 Project Overview
This project applies **K-Means Clustering** to segment customers into distinct groups based on their income and spending behaviors. This is a foundational unsupervised learning task that demonstrates how to identify customer personas without labeled data. **Status: ✅ Complete & Production Ready**

**Level:** Level 1 (Beginner)  
**Duration:** ~3-4 days  
**Last Updated:** October 20, 2025

---

## 🎯 Objectives
- ✅ Load and explore the Mall Customer dataset
- ✅ Perform exploratory data analysis (EDA)
- ✅ Standardize and scale features (crucial for clustering)
- ✅ Apply K-Means clustering algorithm
- ✅ Determine optimal number of clusters (Elbow Method, Silhouette Analysis)
- ✅ Visualize clusters in 2D scatter plots
- ✅ Analyze cluster characteristics and create customer personas
- ✅ Implement bonus features (different algorithms, cluster analysis)

---

## 📊 Dataset
**Dataset Name:** Mall Customer Segmentation Data  
**Source:** [Kaggle - Mall Customer Segmentation](https://www.kaggle.com/datasets/vjcde/customer-segmentation-tutorial-in-python)

**Key Features:**
- `CustomerID`: Unique identifier
- `Age`: Age of the customer
- `Annual_Income`: Annual income in thousands
- `Spending_Score`: Score assigned based on customer behavior (1-100)
- `Gender`: Male or Female

**Dataset Size:** ~200 customer records

---

## 🛠️ Tools & Libraries
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Core programming language |
| Pandas | 2.0.3 | Data manipulation and analysis |
| NumPy | 1.24.3 | Numerical computations |
| Matplotlib | 3.7.2 | Data visualization |
| Seaborn | 0.12.2 | Statistical visualization |
| Scikit-learn | 1.3.0 | Clustering algorithms |
| Jupyter | 1.0.0 | Interactive notebooks |
| Streamlit | 1.28.1 | Web application framework |
| Joblib | 1.3.2 | Model persistence |

---

## 📁 Project Structure
```
Task_2_Customer_Segmentation/
├── notebooks/
│   └── Task_2_Customer_Segmentation.ipynb  # Main analysis notebook (26 cells)
├── model/
│   ├── kmeans_model.pkl                    # Trained K-Means model
│   └── scaler.pkl                          # StandardScaler for features
├── outputs/
│   ├── distributions.png                   # Feature distributions
│   ├── correlation.png                     # Correlation heatmap
│   ├── elbow_method.png                    # Elbow curve analysis
│   ├── silhouette_analysis.png             # Silhouette scores
│   ├── cluster_visualization.png           # 2D cluster plot
│   ├── cluster_profiles.png                # Box plots by cluster
│   └── summary.txt                         # Analysis summary
├── app.py                                  # Streamlit web application
├── sample_batch_input.csv                  # Test data (25 customers)
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git exclusions
├── .streamlit/
│   └── config.toml                         # Streamlit configuration
└── README.md                               # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Dataset (Choose One)

**Option A: Load Online (Recommended)**
The notebook automatically attempts to load the dataset from GitHub. If online loading fails, it will prompt for fallback.

**Option B: Download from Kaggle**
1. Visit: https://www.kaggle.com/datasets/vjcde/customer-segmentation-tutorial-in-python
2. Download `Mall_Customers.csv`
3. Place in the project directory
4. Run the notebook again

### 3. Run the Analysis Notebook
```bash
jupyter notebook notebooks/Task_2_Customer_Segmentation.ipynb
```

### 4. Launch Streamlit Web App (After Notebook Completion)
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

---

## � Model Performance & Results

### Clustering Quality
- **Optimal Clusters:** 5 (determined by Elbow Method and Silhouette Analysis)
- **Algorithm:** K-Means with K-Means++ initialization
- **Feature Scaling:** StandardScaler (normalized to mean=0, std=1)
- **Distance Metric:** Euclidean distance

### Cluster Distribution
| Cluster | Size | Profile |
|---------|------|---------|
| Cluster 0 | ~20% | Budget-Conscious (Low income, low spending) |
| Cluster 1 | ~15% | Premium Spenders (High income, high spending) |
| Cluster 2 | ~18% | Mid-Range Buyers (Mid-high income, moderate spending) |
| Cluster 3 | ~22% | High-Value Prospects (High income, low spending) |
| Cluster 4 | ~25% | Moderate Spenders (Moderate income & spending) |

### Features Used
- **Annual Income (k$):** Range 15-140k
- **Spending Score (1-100):** Range 1-100
- **Both features scaled** for equal contribution to clustering

### Step 1: Data Exploration & Visualization
- Load and inspect dataset
- Check data types and missing values
- Visualize distributions of features
- Create correlation matrix

### Step 2: Feature Selection & Scaling
- Select relevant features (typically Income and Spending_Score)
- Apply **StandardScaler** or **MinMaxScaler**
- Why? K-Means is distance-based; scaling ensures fair feature contribution

### Step 3: Determine Optimal Clusters
- **Elbow Method**: Plot inertia vs number of clusters, find "elbow point"
- **Silhouette Analysis**: Calculate silhouette scores for different k values
- **Davies-Bouldin Index**: Lower values indicate better clustering
- Typical optimal k: 3-5 clusters

### Step 4: Apply K-Means Clustering
- Train K-Means with optimal k
- Assign clusters to each customer
- Store cluster labels

### Step 5: Cluster Analysis & Visualization
- Create 2D scatter plot colored by clusters
- Calculate cluster statistics (mean age, income, spending)
- Develop customer personas for each cluster
- Create business insights

### Step 6: Bonus Tasks (Optional)
- Try **DBSCAN** clustering (density-based)
- Compare DBSCAN vs K-Means performance
- Calculate **average spending per cluster**
- Create business recommendations

---

## 🔍 Key Concepts

### Elbow Method
- Plot sum of squared distances vs number of clusters
- Look for the "elbow" - point where reduction slows dramatically
- This typically indicates the optimal k

### Silhouette Score
- Ranges from -1 to 1
- Values closer to 1 indicate well-separated clusters
- Average silhouette score > 0.5 is generally good

### Feature Scaling Importance
```
Without scaling: 
  - Income (0-100,000) dominates distance calculation
  - Spending_Score (1-100) becomes negligible
  
With scaling:
  - All features equally contribute to clustering
  - Better, more meaningful clusters
```

---

## 📊 Expected Cluster Personas

Based on typical customer segmentation datasets, you might find clusters like:

| Cluster | Profile | Characteristics |
|---------|---------|-----------------|
| Cluster 1 | High-Value Customers | High income, high spending |
| Cluster 2 | Careful Shoppers | High income, low spending |
| Cluster 3 | Target Customers | Medium income, high spending |
| Cluster 4 | Low-Value Customers | Low income, low spending |
| Cluster 5 | Potential Customers | Low income, high spending |

---

## 💡 Tips & Best Practices
- **Always scale features** before clustering
- **Try multiple k values** to confirm optimal clusters
- **Use both elbow and silhouette methods** for validation
- **Interpret clusters** in business context (what do they mean?)
- **Visualize in multiple ways** (2D, 3D if possible)
- **Compare different algorithms** (K-Means, DBSCAN, Hierarchical)
- **Document cluster characteristics** and business implications

---

## 📚 Learning Resources
- [Scikit-learn K-Means Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Elbow Method Explanation](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
- [Silhouette Analysis Guide](https://scikit-learn.org/stable/auto_examples/cluster/plot_silhouette_analysis.html)

---

✅ Completion Checklist
- [x] Dataset loading implemented (online + fallback)
- [x] Data exploration completed (visualizations, statistics)
- [x] Features selected and scaled (StandardScaler)
- [x] Elbow method performed (optimal k determined)
- [x] Silhouette analysis completed
- [x] K-Means model trained (5 clusters)
- [x] Clusters visualized in 2D plot
- [x] Cluster characteristics analyzed
- [x] Customer personas created (5 distinct profiles)
- [x] Streamlit web app built (3 interactive tabs)
- [x] Batch prediction functionality implemented
- [x] Model saved (kmeans_model.pkl, scaler.pkl)
- [x] Results documented with summary report
- [x] Code well-commented and organized
- [x] Configuration files created (.streamlit, .gitignore)
- [x] Sample test data provided (25 customers)

---

## 🎓 Learning Outcomes
By completing this task, you will have gained knowledge of:
✅ Unsupervised learning fundamentals  
✅ K-Means clustering algorithm  
✅ Feature scaling and normalization  
✅ Optimal cluster selection methods  
✅ Cluster evaluation techniques  
✅ Data visualization and interpretation  
✅ Business insights from data  

---

**Status:** ✅ Complete and Production Ready  
**Last Updated:** October 20, 2025  
**Author:** Elevvo Internship Program
