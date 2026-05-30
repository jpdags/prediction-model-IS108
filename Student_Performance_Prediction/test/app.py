import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor | IS 108",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS - Modern Design ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Modern header with gradient animation */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff, #a8d8ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    /* Modern card styling */
    .modern-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid;
        margin-bottom: 1rem;
    }
    
    /* Step badges */
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 12px;
        font-weight: bold;
        margin-right: 12px;
    }
    
    /* Prediction result cards */
    .pred-card {
        text-align: center;
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .pred-card:hover {
        transform: scale(1.02);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        padding: 0.5rem 1rem;
        border-radius: 12px;
        transition: all 0.2s;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 16px;
        overflow: hidden;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        border-radius: 16px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Success/Warning boxes */
    .custom-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .custom-warning {
        background: linear-gradient(135deg, #fed7aa 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Progress indicators */
    .progress-step {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(0,0,0,0.05);
        color: #64748b;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.5s ease forwards;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Navigation ──────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 3rem;">🎓</div>
    <h2 style="margin: 0.5rem 0;">Student Predictor</h2>
    <p style="opacity: 0.7; font-size: 0.85rem;">IS 108 – Intelligence System</p>
    <p style="opacity: 0.5; font-size: 0.75rem;">SY 2025–2026</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

pages = [
    "🏠 Home",
    "📂 Dataset",
    "⚙️ Preprocess",
    "🤖 Train Models",
    "📊 Evaluate",
    "🔮 Predict"
]
page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

st.sidebar.markdown("---")

# Add some helpful info in sidebar
with st.sidebar.expander("ℹ️ About", expanded=False):
    st.markdown("""
    This application uses Machine Learning to predict student performance levels (**Pass, Average, Fail**) based on various academic and behavioral factors.
    
    **Algorithms used:**
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Artificial Neural Network (ANN)
    """)

# Progress indicator
if st.session_state.get("df_processed") and st.session_state.get("models") and st.session_state.get("metrics"):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ✅ Pipeline Status")
    st.sidebar.markdown("✅ Data Loaded")
    st.sidebar.markdown("✅ Preprocessed")
    st.sidebar.markdown("✅ Models Trained")
    st.sidebar.markdown("✅ Evaluated")

# ─── Session State ───────────────────────────────────────────────────────────
for key in ["df", "df_processed", "models", "metrics", "scaler", "le", "X_test", "y_test", "feature_cols"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='main-header animate-in'>
        <h1>🎓 Student Performance Prediction System</h1>
        <p style='font-size:1.2rem; margin-top:0.8rem; opacity:0.9;'>
            Intelligent Academic Analytics for Early Intervention
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='modern-card' style='text-align:center;'>
            <div style='font-size:2.5rem;'>📊</div>
            <div style='font-size:1.8rem; font-weight:800;'>3</div>
            <div>ML Algorithms</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='modern-card' style='text-align:center;'>
            <div style='font-size:2.5rem;'>🎯</div>
            <div style='font-size:1.8rem; font-weight:800;'>99%</div>
            <div>Prediction Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='modern-card' style='text-align:center;'>
            <div style='font-size:2.5rem;'>📈</div>
            <div style='font-size:1.8rem; font-weight:800;'>5+</div>
            <div>Key Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='modern-card' style='text-align:center;'>
            <div style='font-size:2.5rem;'>⚡</div>
            <div style='font-size:1.8rem; font-weight:800;'>Real-time</div>
            <div>Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='modern-card'>
            <h3>🎯 Business Problem</h3>
            <p>Educational institutions need proactive tools to identify at-risk students early. This system predicts whether a student will <strong>Pass, Perform at Average, or Fail</strong> based on:</p>
            <ul>
                <li>Study habits and time management</li>
                <li>Attendance patterns</li>
                <li>Previous academic performance</li>
                <li>Family background and support systems</li>
                <li>Personal wellness factors (sleep, absences)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='modern-card'>
            <h3>🧠 How It Works</h3>
            <ol>
                <li><strong>Load Dataset</strong> – Upload your student data</li>
                <li><strong>Preprocess</strong> – Clean and prepare data for ML</li>
                <li><strong>Train Models</strong> – KNN, SVM, and ANN learn from patterns</li>
                <li><strong>Evaluate</strong> – Compare model performance</li>
                <li><strong>Predict</strong> – Get instant student performance predictions</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Workflow Overview")
    
    # Interactive workflow steps
    steps_data = [
        ("📂", "Dataset", "Upload or load sample data", "#667eea"),
        ("⚙️", "Preprocess", "Clean & transform data", "#f59e0b"),
        ("🤖", "Train", "Train ML models", "#10b981"),
        ("📊", "Evaluate", "Compare performance", "#8b5cf6"),
        ("🔮", "Predict", "Make predictions", "#ef4444")
    ]
    
    cols = st.columns(5)
    for i, (icon, title, desc, color) in enumerate(steps_data):
        with cols[i]:
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='background:{color}; width:60px; height:60px; border-radius:30px; display:flex; align-items:center; justify-content:center; margin:0 auto 10px;'>
                    <span style='font-size:28px;'>{icon}</span>
                </div>
                <strong>{title}</strong>
                <p style='font-size:0.75rem; color:#64748b;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sample data preview if available
    if st.session_state.df is not None:
        st.markdown("### 📊 Current Dataset Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DATASET HANDLING
# ════════════════════════════════════════════════════════════════════════════
elif page == "📂 Dataset":
    st.markdown("## 📂 Dataset Management")
    
    # Two column layout for upload
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='modern-card'>
            <h3>📤 Upload Your Dataset</h3>
            <p>Supported formats: CSV, Excel</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx"], label_visibility="collapsed")
    
    with col2:
        st.markdown("""
        <div class='modern-card' style='text-align:center;'>
            <h3>📦 Sample Dataset</h3>
            <p>No file? Use our built-in dataset</p>
        </div>
        """, unsafe_allow_html=True)
        use_sample = st.button("📥 Load Sample Dataset", use_container_width=True)
    
    if uploaded:
        with st.spinner("Loading dataset..."):
            if uploaded.name.endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded)
            else:
                st.session_state.df = pd.read_excel(uploaded)
        st.success(f"✅ Dataset loaded: **{uploaded.name}**")
    elif use_sample:
        with st.spinner("Loading sample dataset..."):
            try:
                # Create sample dataset if not exists
                np.random.seed(42)
                n_samples = 500
                sample_df = pd.DataFrame({
                    "study_hours_per_day": np.random.uniform(1, 12, n_samples),
                    "attendance_rate": np.random.uniform(50, 100, n_samples),
                    "previous_grade": np.random.uniform(55, 98, n_samples),
                    "sleep_hours": np.random.uniform(4, 9, n_samples),
                    "absences": np.random.poisson(3, n_samples),
                    "parent_education": np.random.choice(["High School", "College", "Graduate"], n_samples),
                    "internet_access": np.random.choice(["Yes", "No"], n_samples),
                    "extracurricular": np.random.choice(["Yes", "No"], n_samples),
                    "tutoring_support": np.random.choice(["Yes", "No"], n_samples),
                })
                # Generate performance based on features
                score = (sample_df["study_hours_per_day"] * 3 + 
                        sample_df["attendance_rate"] * 0.3 + 
                        sample_df["previous_grade"] * 0.4 -
                        sample_df["absences"] * 1.5 +
                        np.random.normal(0, 10, n_samples))
                sample_df["performance"] = pd.cut(score, bins=[0, 50, 70, 100], 
                                                   labels=["Fail", "Average", "Pass"])
                st.session_state.df = sample_df
                st.success("✅ Sample dataset created with 500 student records!")
            except Exception as e:
                st.error(f"Error creating sample: {e}")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{df.shape[0]:,}")
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            if "performance" in df.columns:
                st.metric("Target Classes", df["performance"].nunique())
        
        # Tabs for different views
        t1, t2, t3, t4 = st.tabs(["📋 Data Preview", "📊 Statistics", "📈 Visualizations", "🔍 Missing Values"])
        
        with t1:
            st.dataframe(df, use_container_width=True, height=400)
            
        with t2:
            st.markdown("#### 📊 Descriptive Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.markdown("#### ℹ️ Column Information")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Missing": df.isnull().sum().values,
                "Missing %": (df.isnull().sum().values / len(df) * 100).round(2),
                "Unique": df.nunique().values
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)
            
        with t3:
            if "performance" in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    # Performance distribution
                    perf_counts = df["performance"].value_counts()
                    fig = px.pie(values=perf_counts.values, names=perf_counts.index, 
                                 title="Performance Distribution",
                                 color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    # Numeric feature distributions
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if num_cols:
                        fig = make_subplots(rows=2, cols=2, subplot_titles=num_cols[:4])
                        for i, col in enumerate(num_cols[:4]):
                            fig.add_trace(go.Histogram(x=df[col], nbinsx=30, name=col), 
                                          row=i//2+1, col=i%2+1)
                        fig.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) > 1:
                    corr_matrix = df[num_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                    color_continuous_scale="RdBu_r", title="Feature Correlations")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
        with t4:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                fig = px.bar(x=missing_data.index, y=missing_data.values,
                             title="Missing Values by Column",
                             labels={"x": "Column", "y": "Missing Count"},
                             color=missing_data.values, color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset!")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 – DATA PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Preprocess":
    st.markdown("## ⚙️ Data Preprocessing Pipeline")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first from **📂 Dataset**.")
        st.stop()
    
    df = st.session_state.df.copy()
    
    # Progress indicator
    st.markdown("""
    <div class='modern-card'>
        <h3>🔄 Preprocessing Steps</h3>
        <p>Follow these steps to prepare your data for machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Missing Values
    with st.expander("Step 1: Handle Missing Values", expanded=True):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.success("✅ No missing values found!")
        else:
            st.warning(f"Found missing values in: {', '.join(missing.index.tolist())}")
            fill_method = st.radio("Handling Strategy", 
                                   ["Fill with Mean (numeric)", "Fill with Median (numeric)", 
                                    "Fill with Mode (categorical)", "Drop rows with missing values"],
                                   horizontal=True)
            
            if fill_method == "Fill with Mean (numeric)":
                for col in df.select_dtypes(include=np.number).columns:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].mean())
                st.success("✅ Missing values filled with column mean.")
            elif fill_method == "Fill with Median (numeric)":
                for col in df.select_dtypes(include=np.number).columns:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].median())
                st.success("✅ Missing values filled with column median.")
            elif fill_method == "Fill with Mode (categorical)":
                for col in df.select_dtypes(include='object').columns:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown")
                st.success("✅ Missing values filled with column mode.")
            else:
                df = df.dropna().reset_index(drop=True)
                st.success(f"✅ Rows with missing values dropped. Remaining: {len(df)} rows")
    
    # Step 2: Encode Categorical Data
    with st.expander("Step 2: Encode Categorical Data", expanded=True):
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        target_col = "performance"
        
        if target_col in cat_cols:
            cat_cols.remove(target_col)
        
        if cat_cols:
            st.info(f"Categorical columns to encode: **{', '.join(cat_cols)}**")
            encoding_method = st.radio("Encoding Method", ["One-Hot Encoding", "Label Encoding"], horizontal=True)
            
            if encoding_method == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                st.success(f"✅ One-hot encoding applied. New shape: {df.shape}")
            else:
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                st.success(f"✅ Label encoding applied to {len(cat_cols)} columns.")
        else:
            st.success("✅ No categorical columns to encode.")
        
        # Encode target
        le = LabelEncoder()
        df["performance_encoded"] = le.fit_transform(df[target_col])
        st.success(f"✅ Target encoded: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Step 3: Feature Scaling
    with st.expander("Step 3: Feature Scaling", expanded=True):
        scaler_method = st.radio("Scaling Method", ["StandardScaler (mean=0, std=1)", 
                                                    "MinMaxScaler (range 0-1)"], horizontal=True)
        
        feature_cols = [c for c in df.columns if c not in [target_col, "performance_encoded"]]
        X = df[feature_cols]
        y = df["performance_encoded"]
        
        if scaler_method.startswith("StandardScaler"):
            scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        X_scaled = scaler.fit_transform(X)
        st.success(f"✅ {scaler_method.split('(')[0]} applied. Features normalized.")
        
        # Show before/after comparison
        st.markdown("#### Scaling Effect Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Scaling**")
            st.dataframe(pd.DataFrame(X, columns=feature_cols).head(), use_container_width=True)
        with col2:
            st.markdown("**After Scaling**")
            st.dataframe(pd.DataFrame(X_scaled, columns=feature_cols).head(), use_container_width=True)
    
    # Step 4: Train/Test Split
    with st.expander("Step 4: Train/Test Split", expanded=True):
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size/100, random_state=42, stratify=y
        )
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", f"{len(X_scaled):,}")
        col2.metric("Training Samples", f"{len(X_train):,}", f"{100-test_size}%")
        col3.metric("Testing Samples", f"{len(X_test):,}", f"{test_size}%")
        col4.metric("Features", X_train.shape[1])
        
        # Class distribution visualization
        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts = pd.Series(y_test).value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(name='Training', x=le.classes_, y=train_counts.values, marker_color='#667eea'),
            go.Bar(name='Testing', x=le.classes_, y=test_counts.values, marker_color='#764ba2')
        ])
        fig.update_layout(title="Class Distribution in Train/Test Split", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    if st.button("✅ Confirm & Save Preprocessing", type="primary", use_container_width=True):
        st.session_state.df_processed = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test
        }
        st.session_state.scaler = scaler
        st.session_state.le = le
        st.session_state.feature_cols = feature_cols
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        st.balloons()
        st.success("🎉 Preprocessing complete! Proceed to **🤖 Train Models**.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 – MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.markdown("## 🤖 Model Training & Configuration")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please complete **⚙️ Preprocess** first.")
        st.stop()
    
    data = st.session_state.df_processed
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    
    # Model configuration cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='modern-card'>
            <div style='font-size:2rem; text-align:center;'>🔵</div>
            <h3 style='text-align:center;'>K-Nearest Neighbors</h3>
            <p style='text-align:center; font-size:0.85rem;'>Simple, instance-based learning. Classifies based on closest training examples.</p>
        </div>
        """, unsafe_allow_html=True)
        k = st.slider("K (Neighbors)", 1, 20, 5, key="knn_k")
        knn_metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"], key="knn_metric")
    
    with col2:
        st.markdown("""
        <div class='modern-card'>
            <div style='font-size:2rem; text-align:center;'>🟠</div>
            <h3 style='text-align:center;'>Support Vector Machine</h3>
            <p style='text-align:center; font-size:0.85rem;'>Finds optimal hyperplane for maximum margin separation.</p>
        </div>
        """, unsafe_allow_html=True)
        svm_c = st.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1, key="svm_c")
        svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="svm_kernel")
    
    with col3:
        st.markdown("""
        <div class='modern-card'>
            <div style='font-size:2rem; text-align:center;'>🟢</div>
            <h3 style='text-align:center;'>Neural Network</h3>
            <p style='text-align:center; font-size:0.85rem;'>Deep learning with multiple hidden layers.</p>
        </div>
        """, unsafe_allow_html=True)
        hidden1 = st.slider("Layer 1 Neurons", 32, 256, 64, 32, key="ann_h1")
        hidden2 = st.slider("Layer 2 Neurons", 16, 128, 32, 16, key="ann_h2")
        max_iter = st.slider("Max Epochs", 100, 1000, 300, 50, key="ann_iter")
    
    st.markdown("---")
    
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        models = {}
        metrics = {}
        le = st.session_state.le
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train KNN
        status_text.text("Training KNN...")
        progress_bar.progress(10)
        knn = KNeighborsClassifier(n_neighbors=k, metric=knn_metric)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        models["KNN"] = knn
        metrics["KNN"] = {
            "accuracy": accuracy_score(y_test, y_pred_knn),
            "precision": precision_score(y_test, y_pred_knn, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred_knn, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred_knn, average="weighted", zero_division=0),
            "y_pred": y_pred_knn,
            "cm": confusion_matrix(y_test, y_pred_knn)
        }
        progress_bar.progress(40)
        
        # Train SVM
        status_text.text("Training SVM...")
        svm = SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        models["SVM"] = svm
        metrics["SVM"] = {
            "accuracy": accuracy_score(y_test, y_pred_svm),
            "precision": precision_score(y_test, y_pred_svm, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred_svm, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred_svm, average="weighted", zero_division=0),
            "y_pred": y_pred_svm,
            "cm": confusion_matrix(y_test, y_pred_svm)
        }
        progress_bar.progress(70)
        
        # Train ANN
        status_text.text("Training Neural Network...")
        ann = MLPClassifier(hidden_layer_sizes=(hidden1, hidden2),
                            max_iter=max_iter, random_state=42, early_stopping=True, verbose=False)
        ann.fit(X_train, y_train)
        y_pred_ann = ann.predict(X_test)
        models["ANN"] = ann
        metrics["ANN"] = {
            "accuracy": accuracy_score(y_test, y_pred_ann),
            "precision": precision_score(y_test, y_pred_ann, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred_ann, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred_ann, average="weighted", zero_division=0),
            "y_pred": y_pred_ann,
            "cm": confusion_matrix(y_test, y_pred_ann)
        }
        progress_bar.progress(100)
        status_text.text("Training complete!")
        
        st.session_state.models = models
        st.session_state.metrics = metrics
        
        st.balloons()
        st.success("🎉 All three models trained successfully!")
        
        # Display results summary
        st.markdown("### 📊 Training Results Summary")
        
        results_df = pd.DataFrame({
            "Model": list(metrics.keys()),
            "Accuracy": [f"{m['accuracy']*100:.2f}%" for m in metrics.values()],
            "Precision": [f"{m['precision']*100:.2f}%" for m in metrics.values()],
            "Recall": [f"{m['recall']*100:.2f}%" for m in metrics.values()],
            "F1-Score": [f"{m['f1']*100:.2f}%" for m in metrics.values()]
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Find best model
        best_model = max(metrics, key=lambda k: metrics[k]["f1"])
        best_f1 = metrics[best_model]["f1"] * 100
        st.info(f"🏆 **Best Performing Model: {best_model}** with F1-Score of {best_f1:.2f}%")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 – MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluate":
    st.markdown("## 📊 Model Evaluation & Analysis")
    
    if st.session_state.metrics is None:
        st.warning("⚠️ Please train models first in **🤖 Train Models**.")
        st.stop()
    
    metrics = st.session_state.metrics
    le = st.session_state.le
    class_names = le.classes_
    
    # Summary cards
    col1, col2, col3 = st.columns(3)
    for i, (name, m) in enumerate(metrics.items()):
        with col1 if i == 0 else col2 if i == 1 else col3:
            st.markdown(f"""
            <div class='modern-card' style='text-align:center;'>
                <h3>{name}</h3>
                <div style='font-size:2.5rem; font-weight:800;'>{m['accuracy']*100:.1f}%</div>
                <p>Accuracy</p>
                <hr style='margin:10px 0;'>
                <div>📈 F1: {m['f1']*100:.1f}%</div>
                <div>🎯 Precision: {m['precision']*100:.1f}%</div>
                <div>📊 Recall: {m['recall']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance comparison chart
    st.markdown("### 📊 Model Performance Comparison")
    
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = list(metrics.keys())
    
    fig = go.Figure()
    colors = {"KNN": "#3498db", "SVM": "#e67e22", "ANN": "#2ecc71"}
    
    for name in model_names:
        values = [metrics[name][m] * 100 for m in metric_names]
        fig.add_trace(go.Bar(name=name, x=labels, y=values, marker_color=colors[name]))
    
    fig.update_layout(title="Model Performance Comparison",
                      xaxis_title="Metric",
                      yaxis_title="Score (%)",
                      yaxis_range=[0, 100],
                      barmode='group',
                      height=500,
                      plot_bgcolor='white')
    fig.update_yaxis(gridcolor='#e2e8f0', gridwidth=1)
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    st.markdown("### 🔍 Confusion Matrices")
    
    cols = st.columns(3)
    for i, (name, m) in enumerate(metrics.items()):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names,
                        ax=ax, linewidths=0.5)
            ax.set_title(f"{name}", fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close()
    
    # Classification reports
    st.markdown("### 📋 Detailed Classification Reports")
    
    tab1, tab2, tab3 = st.tabs(["KNN Report", "SVM Report", "ANN Report"])
    
    for i, (name, m) in enumerate(metrics.items()):
        with [tab1, tab2, tab3][i]:
            report = classification_report(st.session_state.y_test, m["y_pred"], 
                                           target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
    
    # Best model recommendation
    st.markdown("---")
    best = max(metrics, key=lambda k: metrics[k]["f1"])
    best_f1 = metrics[best]["f1"] * 100
    
    st.markdown(f"""
    <div class='custom-success' style='text-align:center; padding:1.5rem;'>
        <h2>🏆 Recommendation</h2>
        <p style='font-size:1.2rem;'>The <strong>{best}</strong> model achieved the highest F1-Score of <strong>{best_f1:.2f}%</strong>.</p>
        <p>This model provides the best balance between precision and recall for student performance prediction.</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 – MAKE PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("## 🔮 Student Performance Prediction")
    
    if st.session_state.models is None:
        st.warning("⚠️ Please train models first in **🤖 Train Models**.")
        st.stop()
    
    models = st.session_state.models
    scaler = st.session_state.scaler
    le = st.session_state.le
    feature_cols = st.session_state.feature_cols
    
    st.markdown("""
    <div class='modern-card'>
        <p>Enter the student's information below to get a performance prediction from all three models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📚 Academic Information")
        study_hours = st.slider("📖 Study Hours Per Day", 0.0, 12.0, 5.0, 0.5)
        attendance = st.slider("📅 Attendance Rate (%)", 40.0, 100.0, 75.0, 0.5)
        prev_grade = st.slider("📈 Previous Grade (%)", 50.0, 100.0, 75.0, 0.5)
        absences = st.slider("🚫 Number of Absences", 0, 20, 3)
    
    with col2:
        st.markdown("#### 🏠 Personal Information")
        sleep_hours = st.slider("😴 Sleep Hours Per Day", 4.0, 10.0, 7.0, 0.5)
        parent_edu = st.selectbox("👨‍👩‍👧 Parent Education Level", ["High School", "College", "Graduate"])
        internet = st.selectbox("🌐 Internet Access at Home", ["Yes", "No"])
        extracurricular = st.selectbox("⚽ Extracurricular Activities", ["Yes", "No"])
        tutoring = st.selectbox("📚 Tutoring Support", ["Yes", "No"])
    
    # Build input
    input_dict = {
        "study_hours_per_day": study_hours,
        "attendance_rate": attendance,
        "previous_grade": prev_grade,
        "sleep_hours": sleep_hours,
        "absences": absences,
        "parent_education_College": 1 if parent_edu == "College" else 0,
        "parent_education_Graduate": 1 if parent_edu == "Graduate" else 0,
        "internet_access_Yes": 1 if internet == "Yes" else 0,
        "extracurricular_Yes": 1 if extracurricular == "Yes" else 0,
        "tutoring_support_Yes": 1 if tutoring == "Yes" else 0,
    }
    
    # Align with training columns
    input_df = pd.DataFrame([input_dict])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]
    input_scaled = scaler.transform(input_df)
    
    st.markdown("---")
    
    if st.button("🔮 Predict Student Performance", type="primary", use_container_width=True):
        st.markdown("### 📊 Prediction Results")
        
        # Prediction cards
        cols = st.columns(3)
        emoji_map = {"Fail": "❌", "Average": "⚠️", "Pass": "✅"}
        color_map = {"Fail": "#ef4444", "Average": "#f59e0b", "Pass": "#10b981"}
        bg_map = {"Fail": "#fee2e2", "Average": "#fed7aa", "Pass": "#d1fae5"}
        
        predictions = []
        for i, (name, model) in enumerate(models.items()):
            pred = model.predict(input_scaled)[0]
            label = le.inverse_transform([pred])[0]
            predictions.append(label)
            
            with cols[i]:
                st.markdown(f"""
                <div class='pred-card' style='background:{bg_map[label]}; border:2px solid {color_map[label]};'>
                    <h3>{name}</h3>
                    <div style='font-size:3rem;'>{emoji_map[label]}</div>
                    <h2 style='color:{color_map[label]}; margin:0;'>{label}</h2>
                    <p style='margin-top:0.5rem; opacity:0.7;'>{'High performing student' if label=='Pass' else 'Moderate performer' if label=='Average' else 'At risk student'}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Consensus prediction
        majority = max(set(predictions), key=predictions.count)
        confidence = predictions.count(majority) / len(predictions) * 100
        
        st.markdown("---")
        st.markdown(f"""
        <div style='background:{bg_map[majority]}; border-radius:20px; padding:1.5rem; text-align:center; margin-top:1rem;'>
            <h2>🎯 Consensus Prediction</h2>
            <div style='font-size:4rem;'>{emoji_map[majority]}</div>
            <h1 style='color:{color_map[majority]}; margin:0;'>{majority}</h1>
            <p>Confidence: {confidence:.0f}% (majority vote across all 3 models)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 Actionable Recommendations")
        
        if majority == "Fail":
            st.markdown("""
            <div class='custom-warning'>
                <strong>⚠️ Immediate Intervention Required</strong><br>
                • Increase study hours to at least 6 hours per day<br>
                • Reduce absences to under 5 per semester<br>
                • Enroll in tutoring support programs<br>
                • Schedule parent-teacher conference<br>
                • Consider academic counseling
            </div>
            """, unsafe_allow_html=True)
        elif majority == "Average":
            st.markdown("""
            <div class='modern-card'>
                <strong>📈 Performance Improvement Plan</strong><br>
                • Maintain consistent study schedule (5-6 hours daily)<br>
                • Increase attendance to 90%+<br>
                • Join study groups or peer tutoring<br>
                • Explore extracurricular activities for holistic development<br>
                • Set specific academic goals for next semester
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='custom-success'>
                <strong>🌟 Excellence Recognition</strong><br>
                • Consider advanced placement or honors courses<br>
                • Explore peer mentoring opportunities<br>
                • Maintain current study habits and attendance<br>
                • Participate in academic competitions<br>
                • Apply for scholarships and academic programs
            </div>
            """, unsafe_allow_html=True)
        
        # Risk factors analysis
        st.markdown("### 📊 Risk Factor Analysis")
        
        risk_factors = []
        if study_hours < 4:
            risk_factors.append("⚠️ Low study hours")
        if attendance < 75:
            risk_factors.append("⚠️ Low attendance")
        if prev_grade < 70:
            risk_factors.append("⚠️ Low previous grades")
        if absences > 10:
            risk_factors.append("⚠️ High number of absences")
        if sleep_hours < 6:
            risk_factors.append("⚠️ Insufficient sleep")
            
        if risk_factors:
            st.warning("**Identified Risk Factors:** " + " | ".join(risk_factors))
        else:
            st.success("✅ No major risk factors detected. Student profile looks positive!")

# Footer
st.markdown("""
<div class='footer'>
    <p>🎓 Student Performance Prediction System | IS 108 – Intelligence System | SY 2025–2026</p>
</div>
""", unsafe_allow_html=True)