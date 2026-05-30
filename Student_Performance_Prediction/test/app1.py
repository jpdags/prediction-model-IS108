import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f, #2e86ab);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f4f8;
        border-left: 4px solid #2e86ab;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .step-badge {
        background: #2e86ab;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    [data-testid="stSidebar"] {
        background: #1e3a5f;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Navigation ──────────────────────────────────────────────────────
st.sidebar.markdown("## 🎓 Student Predictor")
st.sidebar.markdown("---")
pages = [
    "🏠 Home",
    "📂 Dataset Handling",
    "⚙️ Data Preprocessing",
    "🤖 Model Training",
    "📊 Model Evaluation",
    "🔮 Make Prediction"
]
page = st.sidebar.radio("Navigate", pages)
st.sidebar.markdown("---")
st.sidebar.markdown("**IS 108 – Intelligence System**")
st.sidebar.markdown("SY 2025–2026")

# ─── Session State ───────────────────────────────────────────────────────────
for key in ["df", "df_processed", "models", "metrics", "scaler", "le", "X_test", "y_test", "feature_cols"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='main-header'>
        <h1>🎓 Student Performance Prediction System</h1>
        <p style='font-size:1.1rem; margin-top:0.5rem;'>
            A Business Intelligence Application using KNN · SVM · ANN
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 📌 Business Problem\nPredict whether a student will **Pass, Average, or Fail** based on study habits, attendance, and background.")
    with col2:
        st.info("### 🧠 Algorithms Used\n- K-Nearest Neighbor (KNN)\n- Support Vector Machine (SVM)\n- Artificial Neural Network (ANN)")
    with col3:
        st.info("### 🗂️ Features Used\nStudy hours, Attendance, Previous grade, Parent education, Internet access, Sleep, Tutoring, Absences")

    st.markdown("---")
    st.markdown("### 📋 How to Use This App")
    steps = [
        ("📂 Dataset Handling", "Upload or load the student dataset. View the data and its basic info."),
        ("⚙️ Data Preprocessing", "Handle missing values, encode categories, scale features, and split data."),
        ("🤖 Model Training", "Train KNN, SVM, and ANN on the processed data."),
        ("📊 Model Evaluation", "Compare models using accuracy, precision, recall, F1, and confusion matrix."),
        ("🔮 Make Prediction", "Input a student's info and get a performance prediction from all 3 models."),
    ]
    for title, desc in steps:
        st.markdown(f"**{title}** — {desc}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DATASET HANDLING
# ════════════════════════════════════════════════════════════════════════════
elif page == "📂 Dataset Handling":
    st.markdown("## 📂 Dataset Handling")

    st.markdown("### Load Dataset")
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    with col2:
        use_sample = st.button("📥 Use Sample Dataset", help="Load the built-in student dataset")

    if uploaded:
        if uploaded.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded)
        else:
            st.session_state.df = pd.read_excel(uploaded)
        st.success(f"✅ Dataset loaded from upload: **{uploaded.name}**")
    elif use_sample:
        try:
            st.session_state.df = pd.read_csv("student_performance.csv")
            st.success("✅ Sample dataset loaded successfully!")
        except:
            st.error("Sample dataset not found. Please upload a CSV file.")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")

        # Tabs for dataset info
        t1, t2, t3 = st.tabs(["📋 Data Preview", "📊 Dataset Info", "📈 Distribution"])

        with t1:
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df, use_container_width=True, height=400)

        with t2:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", df.shape[0])
            col2.metric("Total Features", df.shape[1] - 1)
            col3.metric("Missing Values", df.isnull().sum().sum())

            st.markdown("#### Column Information")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.values,
                "Missing": df.isnull().sum().values,
                "Missing %": (df.isnull().sum().values / len(df) * 100).round(2),
                "Unique Values": df.nunique().values
            })
            st.dataframe(info_df, use_container_width=True)

            st.markdown("#### Descriptive Statistics")
            st.dataframe(df.describe(), use_container_width=True)

        with t3:
            if "performance" in df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                # Target distribution
                counts = df["performance"].value_counts()
                colors = ["#e74c3c", "#f39c12", "#2ecc71"]
                axes[0].bar(counts.index, counts.values, color=colors[:len(counts)])
                axes[0].set_title("Student Performance Distribution", fontweight="bold")
                axes[0].set_xlabel("Performance Level")
                axes[0].set_ylabel("Count")
                for i, v in enumerate(counts.values):
                    axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

                # Numeric feature correlations
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    corr = df[num_cols].corr()
                    sns.heatmap(corr, ax=axes[1], cmap="Blues", annot=True, fmt=".2f", linewidths=0.5)
                    axes[1].set_title("Feature Correlation Heatmap", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 – DATA PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Data Preprocessing":
    st.markdown("## ⚙️ Data Preprocessing")

    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first from **📂 Dataset Handling**.")
        st.stop()

    df = st.session_state.df.copy()

    with st.expander("📖 What is Data Preprocessing?", expanded=False):
        st.markdown("""
        Data preprocessing transforms raw data into a format suitable for machine learning:
        - **Missing Values** – Fill or remove incomplete records
        - **Encoding** – Convert text categories to numbers
        - **Feature Scaling** – Normalize numeric values so no feature dominates
        - **Train/Test Split** – Separate data for training and evaluating models
        """)

    st.markdown("### Step 1 — Handle Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        st.success("✅ No missing values found!")
    else:
        st.warning(f"Found missing values in: {', '.join(missing.index.tolist())}")
        fill_method = st.selectbox("How to handle missing values?",
                                   ["Fill with Mean (numeric columns)", "Drop rows with missing values"])
        if fill_method.startswith("Fill"):
            for col in df.select_dtypes(include=np.number).columns:
                df[col] = df[col].fillna(df[col].mean())
            st.success("✅ Missing values filled with column mean.")
        else:
            df = df.dropna().reset_index(drop=True)
            st.success(f"✅ Rows with missing values dropped. Remaining: {len(df)}")

    # Safety net: always fill any remaining NaNs in numeric columns before proceeding
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    st.markdown("### Step 2 — Encode Categorical Data")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    target_col = "performance"
    feature_cat = [c for c in cat_cols if c != target_col]

    if feature_cat:
        st.info(f"Categorical columns detected: **{', '.join(feature_cat)}**")
        df = pd.get_dummies(df, columns=feature_cat, drop_first=True)
        st.success(f"✅ One-hot encoding applied. New shape: {df.shape}")
    else:
        st.success("✅ No categorical feature columns to encode.")

    # Encode target
    le = LabelEncoder()
    df["performance_encoded"] = le.fit_transform(df[target_col])
    st.success(f"✅ Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    st.markdown("### Step 3 — Feature Scaling")
    feature_cols = [c for c in df.columns if c not in [target_col, "performance_encoded"]]
    X = df[feature_cols]
    y = df["performance_encoded"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.success("✅ StandardScaler applied — all numeric features normalized to mean=0, std=1.")

    st.markdown("### Step 4 — Train / Test Split")
    test_size = st.slider("Test set size (%)", 10, 40, 20, 5)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size/100, random_state=42, stratify=y
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(X_scaled))
    col2.metric("Training Samples", len(X_train))
    col3.metric("Testing Samples", len(X_test))
    st.success(f"✅ Dataset split: {100-test_size}% train / {test_size}% test")

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
        st.success("🎉 Preprocessing complete! Proceed to **🤖 Model Training**.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 – MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("## 🤖 Model Training")

    if st.session_state.df_processed is None:
        st.warning("⚠️ Please complete **⚙️ Data Preprocessing** first.")
        st.stop()

    data = st.session_state.df_processed
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    st.markdown("Configure and train all three models below:")

    # ── KNN ──
    with st.expander("🔵 K-Nearest Neighbor (KNN)", expanded=True):
        st.markdown("""
        **How it works:** KNN classifies a data point by looking at the **K closest neighbors** in the dataset
        and voting on the most common class. Simple but effective for smaller datasets.
        """)
        k = st.slider("Number of Neighbors (K)", 1, 20, 5)
        knn_metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"])

    # ── SVM ──
    with st.expander("🟠 Support Vector Machine (SVM)", expanded=True):
        st.markdown("""
        **How it works:** SVM finds the **best boundary (hyperplane)** that separates classes with
        the maximum margin. Excellent for high-dimensional data.
        """)
        svm_c = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
        svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])

    # ── ANN ──
    with st.expander("🟢 Artificial Neural Network (ANN)", expanded=True):
        st.markdown("""
        **How it works:** ANN is inspired by the human brain — it uses **layers of neurons** that learn
        patterns from data through forward and backward propagation.
        """)
        hidden1 = st.slider("Neurons in Hidden Layer 1", 16, 128, 64)
        hidden2 = st.slider("Neurons in Hidden Layer 2", 8, 64, 32)
        max_iter = st.slider("Max Iterations (Epochs)", 100, 1000, 300)

    st.markdown("---")
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        models = {}
        metrics = {}
        le = st.session_state.le

        with st.spinner("Training KNN..."):
            knn = KNeighborsClassifier(n_neighbors=k, metric=knn_metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            models["KNN"] = knn
            metrics["KNN"] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "y_pred": y_pred,
                "cm": confusion_matrix(y_test, y_pred)
            }

        with st.spinner("Training SVM..."):
            svm = SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            models["SVM"] = svm
            metrics["SVM"] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "y_pred": y_pred,
                "cm": confusion_matrix(y_test, y_pred)
            }

        with st.spinner("Training ANN..."):
            ann = MLPClassifier(hidden_layer_sizes=(hidden1, hidden2),
                                max_iter=max_iter, random_state=42, early_stopping=True)
            ann.fit(X_train, y_train)
            y_pred = ann.predict(X_test)
            models["ANN"] = ann
            metrics["ANN"] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "y_pred": y_pred,
                "cm": confusion_matrix(y_test, y_pred)
            }

        st.session_state.models = models
        st.session_state.metrics = metrics

        st.success("🎉 All models trained successfully!")
        st.markdown("### Training Results Summary")
        cols = st.columns(3)
        colors = {"KNN": "🔵", "SVM": "🟠", "ANN": "🟢"}
        for i, (name, m) in enumerate(metrics.items()):
            with cols[i]:
                st.markdown(f"**{colors[name]} {name}**")
                st.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
                st.metric("F1-Score", f"{m['f1']*100:.2f}%")

        st.info("👉 Go to **📊 Model Evaluation** for detailed metrics and confusion matrices.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 – MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Evaluation":
    st.markdown("## 📊 Model Evaluation & Comparison")

    if st.session_state.metrics is None:
        st.warning("⚠️ Please train models first in **🤖 Model Training**.")
        st.stop()

    metrics = st.session_state.metrics
    le = st.session_state.le
    class_names = le.classes_

    # ── Summary Table ──
    st.markdown("### 📋 Performance Comparison")
    summary = pd.DataFrame({
        "Model": list(metrics.keys()),
        "Accuracy (%)": [f"{m['accuracy']*100:.2f}" for m in metrics.values()],
        "Precision (%)": [f"{m['precision']*100:.2f}" for m in metrics.values()],
        "Recall (%)": [f"{m['recall']*100:.2f}" for m in metrics.values()],
        "F1-Score (%)": [f"{m['f1']*100:.2f}" for m in metrics.values()],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Bar Chart Comparison ──
    st.markdown("### 📊 Metric Comparison Chart")
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = list(metrics.keys())
    x = np.arange(len(labels))
    width = 0.25
    colors_map = {"KNN": "#3498db", "SVM": "#e67e22", "ANN": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, m) in enumerate(metrics.items()):
        vals = [m[k]*100 for k in metric_names]
        bars = ax.bar(x + i*width, vals, width, label=name, color=colors_map[name], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 115)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% line')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Confusion Matrices ──
    st.markdown("### 🔲 Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, m) in enumerate(metrics.items()):
        sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i], linewidths=0.5)
        axes[i].set_title(f"{name} — Confusion Matrix", fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Best Model ──
    best = max(metrics, key=lambda k: metrics[k]["f1"])
    st.markdown("---")
    st.success(f"🏆 **Best Performing Model: {best}** (highest F1-Score: {metrics[best]['f1']*100:.2f}%)")

    # ── Interpretation ──
    with st.expander("📖 Understanding the Metrics"):
        st.markdown("""
        | Metric | What it measures |
        |--------|-----------------|
        | **Accuracy** | Overall correct predictions out of all predictions |
        | **Precision** | Of students predicted as Pass/Fail, how many actually were? |
        | **Recall** | Of actual Pass/Fail students, how many did we catch? |
        | **F1-Score** | Balance between Precision and Recall (best overall metric) |
        | **Confusion Matrix** | Rows = Actual class, Columns = Predicted class |
        """)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 – MAKE PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Make Prediction":
    st.markdown("## 🔮 Make a Prediction")

    if st.session_state.models is None:
        st.warning("⚠️ Please train models first in **🤖 Model Training**.")
        st.stop()

    models = st.session_state.models
    scaler = st.session_state.scaler
    le = st.session_state.le
    feature_cols = st.session_state.feature_cols

    st.markdown("### Enter Student Information")
    st.markdown("Fill in the fields below, then click **Predict Performance**.")

    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Study Hours Per Day", 0.0, 12.0, 5.0, 0.5)
        attendance = st.slider("Attendance Rate (%)", 40.0, 100.0, 75.0, 0.5)
        prev_grade = st.slider("Previous Grade (%)", 50.0, 100.0, 75.0, 0.5)
        sleep_hours = st.slider("Sleep Hours Per Day", 4.0, 10.0, 7.0, 0.5)
        absences = st.slider("Number of Absences", 0, 20, 3)

    with col2:
        parent_edu = st.selectbox("Parent Education Level", ["High School", "College", "Graduate"])
        internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        tutoring = st.selectbox("Tutoring Support", ["Yes", "No"])

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
    if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
        st.markdown("### 📊 Prediction Results")
        cols = st.columns(3)
        emoji_map = {"Fail": "❌", "Average": "⚠️", "Pass": "✅"}
        color_map = {"Fail": "#e74c3c", "Average": "#f39c12", "Pass": "#2ecc71"}

        for i, (name, model) in enumerate(models.items()):
            pred = model.predict(input_scaled)[0]
            label = le.inverse_transform([pred])[0]
            with cols[i]:
                st.markdown(f"""
                <div style='text-align:center; background:{color_map[label]}22;
                border:2px solid {color_map[label]}; border-radius:12px; padding:1.5rem;'>
                    <h3>{name}</h3>
                    <h1>{emoji_map[label]}</h1>
                    <h2 style='color:{color_map[label]}'>{label}</h2>
                </div>
                """, unsafe_allow_html=True)

        # Summary
        preds = [le.inverse_transform([m.predict(input_scaled)[0]])[0] for m in models.values()]
        majority = max(set(preds), key=preds.count)
        st.markdown("---")
        st.markdown(f"""
        <div style='background:rgb(245, 73, 39); border-radius:12px; padding:1.5rem; text-align:center;'>
            <h3>🎯 Consensus Prediction: <span style='color:{color_map[majority]}'>{emoji_map[majority]} {majority}</span></h3>
            <p>Based on majority vote across all 3 models</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 💡 Recommendations")
        if majority == "Fail":
            st.error("**This student is at risk.** Recommend: increase study hours, reduce absences, seek tutoring support.")
        elif majority == "Average":
            st.warning("**Performing adequately.** Recommend: maintain consistency, explore extracurricular activities for growth.")
        else:
            st.success("**Excellent performance!** Recommend: challenge with advanced coursework or mentoring roles.")
