import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor | IS 108",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem; border-radius: 24px; color: white; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #fff, #a8d8ea);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.5rem;
    }
    .modern-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px; padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .modern-card:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(0,0,0,0.1); }
    .pred-card {
        text-align: center; border-radius: 20px; padding: 1.5rem;
        transition: all 0.3s ease; backdrop-filter: blur(10px);
    }
    .pred-card:hover { transform: scale(1.02); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255,255,255,0.05); border-radius: 16px; padding: 0.5rem;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 0.5rem 1rem; border-radius: 12px; transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.1); }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 14px;
        padding: 0.6rem 1.2rem; font-weight: 600; transition: all 0.2s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,0.4); }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: #f1f5f9; border-radius: 16px; padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 12px; padding: 0.5rem 1.5rem; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .custom-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981; border-radius: 12px; padding: 1rem;
    }
    .custom-warning {
        background: linear-gradient(135deg, #fed7aa 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b; border-radius: 12px; padding: 1rem;
    }
    .footer {
        text-align: center; padding: 2rem; margin-top: 2rem;
        border-top: 1px solid rgba(0,0,0,0.05); color: #64748b;
    }
    @keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
    .animate-in { animation: fadeInUp 0.5s ease forwards; }
</style>
""", unsafe_allow_html=True)

# ─── Dataset Schema (single — Student_Performance.csv, 25,000 rows) ──────────
SCHEMA = {
    "filename":    "student_performance.csv",
    "description": "25,000 student records — age, gender, school type, subject scores, study method, overall score",
    "drop_cols":   ["student_id", "final_grade"],
    "rename_map":  {},
    "target_col":  "performance",
    "target_source": "overall_score",
    "target_bins":   [0, 50, 70, 101],
    "target_labels": ["Fail", "Average", "Pass"],
    "numeric_features":     ["age", "study_hours", "attendance_percentage",
                              "math_score", "science_score", "english_score", "overall_score"],
    "categorical_features": ["gender", "school_type", "internet_access", "travel_time",
                              "extra_activities", "parent_education", "study_method"],
    # Predict inputs — every column the user fills in
    "predict_inputs": {
        "age":                  {"type": "slider", "label": "🎂 Age",                    "min": 14,   "max": 19,    "default": 16,   "step": 1},
        "study_hours":          {"type": "slider", "label": "📖 Study Hours Per Day",    "min": 0.5,  "max": 8.0,   "default": 4.0,  "step": 0.5},
        "attendance_percentage":{"type": "slider", "label": "📅 Attendance (%)",         "min": 50.0, "max": 100.0, "default": 75.0, "step": 0.5},
        "math_score":           {"type": "slider", "label": "➕ Math Score",             "min": 0.0,  "max": 100.0, "default": 64.0, "step": 1.0},
        "science_score":        {"type": "slider", "label": "🔬 Science Score",          "min": 0.0,  "max": 100.0, "default": 64.0, "step": 1.0},
        "english_score":        {"type": "slider", "label": "📝 English Score",          "min": 0.0,  "max": 100.0, "default": 64.0, "step": 1.0},
        "gender":           {"type": "select", "label": "⚧ Gender",            "options": ["male", "female", "other"]},
        "school_type":      {"type": "select", "label": "🏫 School Type",       "options": ["public", "private"]},
        "internet_access":  {"type": "select", "label": "🌐 Internet Access",   "options": ["yes", "no"]},
        "travel_time":      {"type": "select", "label": "🚌 Travel Time",       "options": ["<15 min", "15-30 min", "30-60 min", ">60 min"]},
        "extra_activities": {"type": "select", "label": "⚽ Extra Activities",  "options": ["yes", "no"]},
        "parent_education": {"type": "select", "label": "👨‍🎓 Parent Education", "options": ["no formal", "high school", "diploma", "graduate", "post graduate", "phd"]},
        "study_method":     {"type": "select", "label": "📚 Study Method",      "options": ["notes", "textbook", "online videos", "coaching", "group study", "mixed"]},
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_and_adapt(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused cols, rename, derive target label from overall_score."""
    df = raw_df.copy()
    drop = [c for c in SCHEMA["drop_cols"] if c in df.columns]
    if drop:
        df.drop(columns=drop, inplace=True)
    df.rename(columns=SCHEMA["rename_map"], inplace=True)
    target = SCHEMA["target_col"]
    src    = SCHEMA["target_source"]
    if target not in df.columns and src in df.columns:
        df[target] = pd.cut(df[src], bins=SCHEMA["target_bins"], labels=SCHEMA["target_labels"])
    return df

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding:1rem 0;">
    <div style="font-size:3rem;">🎓</div>
    <h2 style="margin:0.5rem 0;">Student Predictor</h2>
    <p style="opacity:0.7; font-size:0.85rem;">IS 108 – Intelligence System</p>
    <p style="opacity:0.5; font-size:0.75rem;">SY 2025–2026</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

pages = ["🏠 Home", "📂 Dataset", "⚙️ Preprocess", "🤖 Train Models", "📊 Evaluate", "🔮 Predict"]
page  = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")
st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ About", expanded=False):
    st.markdown(f"""
    Predicts student performance (**Pass / Average / Fail**) using KNN, SVM, and ANN.

    **Dataset:** `{SCHEMA['filename']}` — {SCHEMA['description']}

    **Target:** `overall_score` binned →
    - 🔴 Fail  (< 50)
    - 🟡 Average (50–70)
    - 🟢 Pass  (> 70)
    """)

if (st.session_state.get("df_processed") and
        st.session_state.get("models") and
        st.session_state.get("metrics")):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ✅ Pipeline Status")
    st.sidebar.markdown("✅ Data Loaded")
    st.sidebar.markdown("✅ Preprocessed")
    st.sidebar.markdown("✅ Models Trained")
    st.sidebar.markdown("✅ Evaluated")

# ─── Session State ────────────────────────────────────────────────────────────
for key in ["df", "df_processed", "models", "metrics", "scaler",
            "le", "X_test", "y_test", "feature_cols"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='main-header animate-in'>
        <h1>🎓 Student Performance Prediction System</h1>
        <p style='font-size:1.2rem; margin-top:0.8rem; opacity:0.9;'>
            Intelligent Academic Analytics for Early Intervention
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label in [
        (c1, "📊", "3",       "ML Algorithms"),
        (c2, "🗂️", "25,000", "Student Records"),
        (c3, "📈", "13",      "Input Features"),
        (c4, "⚡", "Real-time","Predictions"),
    ]:
        with col:
            st.markdown(f"""
            <div class='modern-card' style='text-align:center;'>
                <div style='font-size:2.5rem;'>{icon}</div>
                <div style='font-size:1.8rem; font-weight:800;'>{val}</div>
                <div>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='modern-card'>
            <h3>🎯 Business Problem</h3>
            <p>Educational institutions need proactive tools to identify at-risk students early.
            This system predicts whether a student will <strong>Pass, Perform at Average, or Fail</strong> based on:</p>
            <ul>
                <li>Study hours per day</li>
                <li>Attendance percentage</li>
                <li>Subject scores (Math, Science, English)</li>
                <li>School type and internet access</li>
                <li>Parent education and travel time</li>
                <li>Extra activities and study method</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='modern-card'>
            <h3>🧠 How It Works</h3>
            <ol>
                <li><strong>Upload Dataset</strong> – Upload <code>student_performance.csv</code></li>
                <li><strong>Preprocess</strong> – Clean and prepare data for ML</li>
                <li><strong>Train Models</strong> – KNN, SVM, and ANN learn from patterns</li>
                <li><strong>Evaluate</strong> – Compare model performance</li>
                <li><strong>Predict</strong> – Get instant student performance predictions</li>
            </ol>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Workflow Overview")
    cols = st.columns(5)
    for col, (icon, title, desc, color) in zip(cols, [
        ("📂","Dataset",   "Upload student_performance.csv", "#667eea"),
        ("⚙️","Preprocess","Clean & transform data",         "#f59e0b"),
        ("🤖","Train",     "Train ML models",                "#10b981"),
        ("📊","Evaluate",  "Compare performance",            "#8b5cf6"),
        ("🔮","Predict",   "Make predictions",               "#ef4444"),
    ]):
        with col:
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='background:{color}; width:60px; height:60px; border-radius:30px;
                     display:flex; align-items:center; justify-content:center; margin:0 auto 10px;'>
                    <span style='font-size:28px;'>{icon}</span>
                </div>
                <strong>{title}</strong>
                <p style='font-size:0.75rem; color:#64748b;'>{desc}</p>
            </div>""", unsafe_allow_html=True)

    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### 📊 Current Dataset Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Dataset":
    st.markdown("## 📂 Dataset Management")

    st.markdown(f"""
    <div class='modern-card'>
        <h3>📁 Upload <code>{SCHEMA['filename']}</code></h3>
        <p>{SCHEMA['description']}</p>
        <p style='font-size:0.85rem; color:#64748b;'>
            Rename your file to <strong>student_performance.csv</strong> before uploading,
            or simply upload the file — the app expects the same column structure.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        f"Upload `{SCHEMA['filename']}` (or any CSV with the same columns)",
        type=["csv", "xlsx"],
        key="dataset_upload"
    )

    # Auto-set default message when nothing uploaded yet
    if uploaded is None and st.session_state.df is None:
        st.info("⬆️ Upload `student_performance.csv` to begin. "
                "This is the 25,000-row multi-factor student dataset.")

    if uploaded is not None:
        with st.spinner("Loading and adapting dataset…"):
            raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            df  = load_and_adapt(raw)
        st.session_state.df = df
        st.success(f"✅ Dataset loaded: **{uploaded.name}** — {len(df):,} rows × {df.shape[1]} columns")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{df.shape[0]:,}")
        c2.metric("Features", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())
        target = SCHEMA["target_col"]
        if target in df.columns:
            c4.metric("Target Classes", df[target].nunique())

        t1, t2, t3, t4 = st.tabs(["📋 Data Preview", "📊 Statistics", "📈 Visualizations", "🔍 Missing Values"])

        with t1:
            st.dataframe(df, use_container_width=True, height=400)

        with t2:
            st.markdown("#### 📊 Descriptive Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            info_df = pd.DataFrame({
                "Column":    df.columns,
                "Type":      df.dtypes.values,
                "Missing":   df.isnull().sum().values,
                "Missing %": (df.isnull().sum().values / len(df) * 100).round(2),
                "Unique":    df.nunique().values,
            })
            st.markdown("#### ℹ️ Column Information")
            st.dataframe(info_df, use_container_width=True, hide_index=True)

        with t3:
            target = SCHEMA["target_col"]
            if target in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    perf_counts = df[target].value_counts()
                    fig = px.pie(values=perf_counts.values, names=perf_counts.index,
                                 title="Performance Distribution",
                                 color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    num_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if num_cols:
                        fig = make_subplots(rows=2, cols=2, subplot_titles=num_cols[:4])
                        for i, c in enumerate(num_cols[:4]):
                            fig.add_trace(go.Histogram(x=df[c], nbinsx=30, name=c),
                                          row=i // 2 + 1, col=i % 2 + 1)
                        fig.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) > 1:
                    corr = df[num_cols].corr()
                    fig = px.imshow(corr, text_auto=True, aspect="auto",
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Preprocess":
    st.markdown("## ⚙️ Data Preprocessing Pipeline")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload the dataset first from **📂 Dataset**.")
        st.stop()

    df         = st.session_state.df.copy()
    target_col = SCHEMA["target_col"]

    # Safety: re-derive target if missing
    src = SCHEMA["target_source"]
    if target_col not in df.columns:
        if src in df.columns:
            df[target_col] = pd.cut(df[src], bins=SCHEMA["target_bins"], labels=SCHEMA["target_labels"])
            st.info(f"ℹ️ Target **'{target_col}'** derived from **'{src}'**.")
        else:
            st.error(f"❌ Column '{target_col}' not found. Re-upload the dataset.")
            st.stop()

    st.markdown("""
    <div class='modern-card'>
        <h3>🔄 Preprocessing Steps</h3>
        <p>Dataset: <strong>student_performance.csv</strong> — 25,000 student records</p>
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
                st.success("✅ Filled with column mean.")
            elif fill_method == "Fill with Median (numeric)":
                for col in df.select_dtypes(include=np.number).columns:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].median())
                st.success("✅ Filled with column median.")
            elif fill_method == "Fill with Mode (categorical)":
                for col in df.select_dtypes(include="object").columns:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown")
                st.success("✅ Filled with column mode.")
            else:
                df = df.dropna().reset_index(drop=True)
                st.success(f"✅ Rows dropped. Remaining: {len(df):,}")

    # Step 2: Encode Categorical
    with st.expander("Step 2: Encode Categorical Data", expanded=True):
        cat_cols = [c for c in df.select_dtypes(include="object").columns if c != target_col]
        if cat_cols:
            st.info(f"Categorical columns to encode: **{', '.join(cat_cols)}**")
            encoding_method = st.radio("Encoding Method",
                ["One-Hot Encoding", "Label Encoding"], horizontal=True)
            if encoding_method == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                st.success(f"✅ One-hot encoding applied. New shape: {df.shape}")
            else:
                for col in cat_cols:
                    le_tmp = LabelEncoder()
                    df[col] = le_tmp.fit_transform(df[col].astype(str))
                st.success(f"✅ Label encoding applied to {len(cat_cols)} columns.")
        else:
            st.success("✅ No categorical columns to encode.")

        le = LabelEncoder()
        df[target_col + "_encoded"] = le.fit_transform(df[target_col].astype(str))
        st.success(f"✅ Target encoded: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # Step 3: Feature Scaling
    with st.expander("Step 3: Feature Scaling", expanded=True):
        scaler_method = st.radio("Scaling Method",
            ["StandardScaler (mean=0, std=1)", "MinMaxScaler (range 0-1)"], horizontal=True)

        feature_cols = [c for c in df.columns
                        if c not in [target_col, target_col + "_encoded", SCHEMA["target_source"]]]
        X = df[feature_cols]
        y = df[target_col + "_encoded"]

        if scaler_method.startswith("StandardScaler"):
            scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()

        X_scaled = scaler.fit_transform(X)
        st.success(f"✅ {scaler_method.split('(')[0]} applied. {len(feature_cols)} features normalized.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Scaling (sample)**")
            st.dataframe(pd.DataFrame(X, columns=feature_cols).head(), use_container_width=True)
        with col2:
            st.markdown("**After Scaling (sample)**")
            st.dataframe(pd.DataFrame(X_scaled, columns=feature_cols).head(), use_container_width=True)

    # Step 4: Train/Test Split
    with st.expander("Step 4: Train/Test Split", expanded=True):
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size / 100, random_state=42, stratify=y)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples",    f"{len(X_scaled):,}")
        c2.metric("Training Samples", f"{len(X_train):,}", f"{100 - test_size}%")
        c3.metric("Testing Samples",  f"{len(X_test):,}",  f"{test_size}%")
        c4.metric("Features",         X_train.shape[1])

        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts  = pd.Series(y_test).value_counts().sort_index()
        fig = go.Figure(data=[
            go.Bar(name="Training", x=le.classes_, y=train_counts.values, marker_color="#667eea"),
            go.Bar(name="Testing",  x=le.classes_, y=test_counts.values,  marker_color="#764ba2"),
        ])
        fig.update_layout(title="Class Distribution in Train/Test Split", barmode="group", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if st.button("✅ Confirm & Save Preprocessing", type="primary", use_container_width=True):
        st.session_state.df_processed = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }
        st.session_state.scaler      = scaler
        st.session_state.le          = le
        st.session_state.feature_cols = feature_cols
        st.session_state.X_test      = X_test
        st.session_state.y_test      = y_test
        st.balloons()
        st.success("🎉 Preprocessing complete! Proceed to **🤖 Train Models**.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.markdown("## 🤖 Model Training & Configuration")

    if st.session_state.df_processed is None:
        st.warning("⚠️ Please complete **⚙️ Preprocess** first.")
        st.stop()

    data = st.session_state.df_processed
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='modern-card'>
            <div style='font-size:2rem; text-align:center;'>🔵</div>
            <h3 style='text-align:center;'>K-Nearest Neighbors</h3>
            <p style='text-align:center; font-size:0.85rem;'>Classifies based on closest training examples.</p>
        </div>""", unsafe_allow_html=True)
        k          = st.slider("K (Neighbors)",   1,    20,   5,   key="knn_k")
        knn_metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"], key="knn_metric")

    with col2:
        st.markdown("""
        <div class='modern-card'>
            <div style='font-size:2rem; text-align:center;'>🟠</div>
            <h3 style='text-align:center;'>Support Vector Machine</h3>
            <p style='text-align:center; font-size:0.85rem;'>Finds optimal hyperplane for max-margin separation.</p>
        </div>""", unsafe_allow_html=True)
        svm_c      = st.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1, key="svm_c")
        svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="svm_kernel")

    with col3:
        st.markdown("""
        <div class='modern-card'>
            <div style='font-size:2rem; text-align:center;'>🟢</div>
            <h3 style='text-align:center;'>Neural Network (ANN)</h3>
            <p style='text-align:center; font-size:0.85rem;'>Deep learning with multiple hidden layers.</p>
        </div>""", unsafe_allow_html=True)
        hidden1  = st.slider("Layer 1 Neurons", 32,  256, 64,  32,  key="ann_h1")
        hidden2  = st.slider("Layer 2 Neurons", 16,  128, 32,  16,  key="ann_h2")
        max_iter = st.slider("Max Epochs",      100, 1000, 300, 50,  key="ann_iter")

    st.markdown("---")
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        models, metrics = {}, {}
        le = st.session_state.le
        progress_bar = st.progress(0)
        status_text  = st.empty()

        status_text.text("Training KNN…")
        progress_bar.progress(10)
        knn = KNeighborsClassifier(n_neighbors=k, metric=knn_metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        models["KNN"] = knn
        metrics["KNN"] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "y_pred": y_pred, "cm": confusion_matrix(y_test, y_pred),
        }
        progress_bar.progress(40)

        status_text.text("Training SVM…")
        svm = SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        models["SVM"] = svm
        metrics["SVM"] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "y_pred": y_pred, "cm": confusion_matrix(y_test, y_pred),
        }
        progress_bar.progress(70)

        status_text.text("Training Neural Network…")
        ann = MLPClassifier(hidden_layer_sizes=(hidden1, hidden2),
                            max_iter=max_iter, random_state=42,
                            early_stopping=True, verbose=False)
        ann.fit(X_train, y_train)
        y_pred = ann.predict(X_test)
        models["ANN"] = ann
        metrics["ANN"] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "y_pred": y_pred, "cm": confusion_matrix(y_test, y_pred),
        }
        progress_bar.progress(100)
        status_text.text("Training complete!")

        st.session_state.models  = models
        st.session_state.metrics = metrics
        st.balloons()
        st.success("🎉 All three models trained successfully!")

        results_df = pd.DataFrame({
            "Model":     list(metrics.keys()),
            "Accuracy":  [f"{m['accuracy']*100:.2f}%"  for m in metrics.values()],
            "Precision": [f"{m['precision']*100:.2f}%" for m in metrics.values()],
            "Recall":    [f"{m['recall']*100:.2f}%"    for m in metrics.values()],
            "F1-Score":  [f"{m['f1']*100:.2f}%"        for m in metrics.values()],
        })
        st.markdown("### 📊 Training Results Summary")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        best    = max(metrics, key=lambda k: metrics[k]["f1"])
        best_f1 = metrics[best]["f1"] * 100
        st.info(f"🏆 **Best Model: {best}** with F1-Score of {best_f1:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluate":
    st.markdown("## 📊 Model Evaluation & Analysis")

    if st.session_state.metrics is None:
        st.warning("⚠️ Please train models first in **🤖 Train Models**.")
        st.stop()

    metrics     = st.session_state.metrics
    le          = st.session_state.le
    class_names = le.classes_

    col1, col2, col3 = st.columns(3)
    for i, (name, m) in enumerate(metrics.items()):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class='modern-card' style='text-align:center;'>
                <h3>{name}</h3>
                <div style='font-size:2.5rem; font-weight:800;'>{m['accuracy']*100:.1f}%</div>
                <p>Accuracy</p>
                <hr style='margin:10px 0;'>
                <div>📈 F1: {m['f1']*100:.1f}%</div>
                <div>🎯 Precision: {m['precision']*100:.1f}%</div>
                <div>📊 Recall: {m['recall']*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Performance Comparison")
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels       = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors       = {"KNN": "#3498db", "SVM": "#e67e22", "ANN": "#2ecc71"}
    fig = go.Figure()
    for name in metrics:
        fig.add_trace(go.Bar(name=name,
                             x=labels,
                             y=[metrics[name][m] * 100 for m in metric_names],
                             marker_color=colors[name]))
    fig.update_layout(title="Model Performance Comparison", xaxis_title="Metric",
                      yaxis_title="Score (%)", yaxis_range=[0, 100],
                      barmode="group", height=500, plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#e2e8f0", gridwidth=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Confusion Matrices")
    cols = st.columns(3)
    for i, (name, m) in enumerate(metrics.items()):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names,
                        ax=ax, linewidths=0.5)
            ax.set_title(name, fontweight="bold")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig); plt.close()

    st.markdown("### 📋 Detailed Classification Reports")
    tab1, tab2, tab3 = st.tabs(["KNN Report", "SVM Report", "ANN Report"])
    for i, (name, m) in enumerate(metrics.items()):
        with [tab1, tab2, tab3][i]:
            report = classification_report(st.session_state.y_test, m["y_pred"],
                                           target_names=class_names, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    st.markdown("---")
    best    = max(metrics, key=lambda k: metrics[k]["f1"])
    best_f1 = metrics[best]["f1"] * 100
    st.markdown(f"""
    <div class='custom-success' style='text-align:center; padding:1.5rem;'>
        <h2>🏆 Recommendation</h2>
        <p style='font-size:1.2rem;'>The <strong>{best}</strong> model achieved the highest F1-Score of
        <strong>{best_f1:.2f}%</strong>.</p>
        <p>This model provides the best balance between precision and recall for student performance prediction.</p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("## 🔮 Student Performance Prediction")

    if st.session_state.models is None:
        st.warning("⚠️ Please train models first in **🤖 Train Models**.")
        st.stop()

    models       = st.session_state.models
    scaler       = st.session_state.scaler
    le           = st.session_state.le
    feature_cols = st.session_state.feature_cols
    predict_inputs = SCHEMA["predict_inputs"]

    st.markdown("""
    <div class='modern-card'>
        <h3>📋 Enter Student Information</h3>
        <p>Fill in all 13 features from the dataset. The three trained models will each produce
        a prediction and a consensus result will be shown.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: numeric sliders ────────────────────────────────────────────────
    st.markdown("#### 📊 Academic & Attendance Metrics")
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    input_values = {}
    slider_fields = ["age", "study_hours", "attendance_percentage",
                     "math_score", "science_score", "english_score"]
    slider_cols   = [r1c1, r1c2, r1c3, r2c1, r2c2, r2c3]

    for key, col in zip(slider_fields, slider_cols):
        cfg = predict_inputs[key]
        with col:
            if isinstance(cfg["min"], float) or isinstance(cfg["default"], float):
                input_values[key] = st.slider(cfg["label"],
                    float(cfg["min"]), float(cfg["max"]),
                    float(cfg["default"]), float(cfg.get("step", 0.5)))
            else:
                input_values[key] = st.slider(cfg["label"],
                    int(cfg["min"]), int(cfg["max"]),
                    int(cfg["default"]), int(cfg.get("step", 1)))

    st.markdown("---")

    # ── Row 2: categorical selects ────────────────────────────────────────────
    st.markdown("#### 🏫 Student Background & Environment")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc5, sc6, sc7       = st.columns(3)

    select_fields = ["gender", "school_type", "internet_access", "travel_time",
                     "extra_activities", "parent_education", "study_method"]
    select_cols   = [sc1, sc2, sc3, sc4, sc5, sc6, sc7]

    for key, col in zip(select_fields, select_cols):
        cfg = predict_inputs[key]
        with col:
            input_values[key] = st.selectbox(cfg["label"], cfg["options"])

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────────────────────
    if st.button("🔮 Predict Student Performance", type="primary", use_container_width=True):

        # Build raw input DataFrame and one-hot encode to match training
        input_df = pd.DataFrame([input_values])
        cat_cols_pred = input_df.select_dtypes(include="object").columns.tolist()
        input_encoded = pd.get_dummies(input_df, columns=cat_cols_pred, drop_first=True)

        # Align with training feature columns
        for col in feature_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_cols].fillna(0)
        input_scaled  = scaler.transform(input_encoded)

        st.markdown("### 📊 Prediction Results")
        emoji_map  = {"Fail": "❌", "Average": "⚠️", "Pass": "✅"}
        color_map  = {"Fail": "#ef4444", "Average": "#f59e0b", "Pass": "#10b981"}
        bg_map     = {"Fail": "#fee2e2", "Average": "#fed7aa", "Pass": "#d1fae5"}
        desc_map   = {
            "Pass":    "High performing student",
            "Average": "Moderate performer",
            "Fail":    "At-risk student",
        }

        predictions = []
        cols = st.columns(3)
        for i, (name, model) in enumerate(models.items()):
            pred  = model.predict(input_scaled)[0]
            label = le.inverse_transform([pred])[0]
            predictions.append(label)
            with cols[i]:
                st.markdown(f"""
                <div class='pred-card'
                     style='background:{bg_map.get(label,"#f1f5f9")};
                            border:2px solid {color_map.get(label,"#94a3b8")};'>
                    <h3>{name}</h3>
                    <div style='font-size:3rem;'>{emoji_map.get(label,"🔵")}</div>
                    <h2 style='color:{color_map.get(label,"#334155")}; margin:0;'>{label}</h2>
                    <p style='margin-top:0.5rem; opacity:0.7;'>{desc_map.get(label,"")}</p>
                </div>""", unsafe_allow_html=True)

        majority    = max(set(predictions), key=predictions.count)
        confidence  = predictions.count(majority) / len(predictions) * 100

        st.markdown("---")
        st.markdown(f"""
        <div style='background:{bg_map.get(majority,"#f1f5f9")}; border-radius:20px;
                    padding:1.5rem; text-align:center; margin-top:1rem;'>
            <h2>🎯 Consensus Prediction</h2>
            <div style='font-size:4rem;'>{emoji_map.get(majority,"🔵")}</div>
            <h1 style='color:{color_map.get(majority,"#334155")}; margin:0;'>{majority}</h1>
            <p>Confidence: {confidence:.0f}% (majority vote across all 3 models)</p>
        </div>""", unsafe_allow_html=True)

        # ── Input Summary Card ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Input Summary")
        summary_data = {
            "Feature": [
                "Age", "Study Hours/Day", "Attendance %",
                "Math Score", "Science Score", "English Score",
                "Gender", "School Type", "Internet Access",
                "Travel Time", "Extra Activities", "Parent Education", "Study Method"
            ],
            "Value": [
                input_values["age"], input_values["study_hours"],
                f"{input_values['attendance_percentage']}%",
                input_values["math_score"], input_values["science_score"],
                input_values["english_score"],
                input_values["gender"].title(), input_values["school_type"].title(),
                input_values["internet_access"].title(), input_values["travel_time"],
                input_values["extra_activities"].title(),
                input_values["parent_education"].title(),
                input_values["study_method"].title(),
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown("### 💡 Actionable Recommendations")
        if majority == "Fail":
            st.markdown("""
            <div class='custom-warning'>
                <strong>⚠️ Immediate Intervention Required</strong><br>
                • Significantly increase daily study hours<br>
                • Improve attendance — aim for 90%+<br>
                • Enroll in tutoring or coaching sessions<br>
                • Schedule a parent-teacher conference<br>
                • Consider academic counseling support
            </div>""", unsafe_allow_html=True)
        elif majority == "Average":
            st.markdown("""
            <div class='modern-card'>
                <strong>📈 Performance Improvement Plan</strong><br>
                • Maintain a consistent daily study schedule<br>
                • Aim for 90%+ attendance<br>
                • Join study groups or peer tutoring sessions<br>
                • Explore extracurricular activities for holistic development<br>
                • Set specific academic goals for the next semester
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='custom-success'>
                <strong>🌟 Excellence Recognition</strong><br>
                • Consider advanced placement or honors courses<br>
                • Explore peer mentoring opportunities<br>
                • Maintain current study habits and attendance<br>
                • Participate in academic competitions<br>
                • Apply for scholarships and academic programs
            </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    <p>🎓 Student Performance Prediction System | IS 108 – Intelligence System | SY 2025–2026</p>
</div>
""", unsafe_allow_html=True)