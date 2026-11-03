
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split


# -------------------------------
# 1. PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="LinkedIn Profile Checker",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI-Based LinkedIn Profile Checker")
st.write(
    "An experimental machine learning project "
    "that identifies suspicious-looking profile patterns."
)


# -------------------------------
# 2. LOAD DATASET
# -------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("sample_data.csv")

    required_columns = [
        "account_age_months",
        "connections",
        "posts",
        "has_profile_photo",
        "profile_completeness",
        "label"
    ]

    if not all(column in data.columns for column in required_columns):
        raise ValueError("Dataset is missing required columns.")

    if data.empty:
        raise ValueError("Dataset is empty.")

    if data[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    return data


try:
    df = load_data()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Dataset error: {error}")
    st.stop()


# -------------------------------
# 3. PREPARE DATA
# -------------------------------

features = [
    "account_age_months",
    "connections",
    "posts",
    "has_profile_photo",
    "profile_completeness"
]

X = df[features]
y = df["label"]

if y.nunique() < 2:
    st.error("Dataset must contain both classes: 0 and 1.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# -------------------------------
# 4. TRAIN MODEL
# -------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# -------------------------------
# 5. MODEL EVALUATION
# -------------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

st.header("📊 Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Precision", f"{precision * 100:.2f}%")
col3.metric("Recall", f"{recall * 100:.2f}%")


# -------------------------------
# 6. CONFUSION MATRIX
# -------------------------------

st.subheader("Confusion Matrix")

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual Genuine", "Actual Suspicious"],
    columns=["Predicted Genuine", "Predicted Suspicious"]
)

st.dataframe(matrix_df)


# -------------------------------
# 7. FAILURE CASES
# -------------------------------

st.subheader("⚠️ Failure Case Analysis")

test_results = X_test.copy()
test_results["actual_label"] = y_test
test_results["predicted_label"] = predictions

failures = test_results[
    test_results["actual_label"] != test_results["predicted_label"]
]

if failures.empty:
    st.info(
        "No incorrect predictions were found in this test split. "
        "This does not mean the model is always correct."
    )
else:
    st.write("Examples where the model made an incorrect prediction:")
    st.dataframe(failures)


# -------------------------------
# 8. USER INPUT AND PREDICTION
# -------------------------------

st.header("🧪 Try Your Own Example")

account_age = st.number_input(
    "Account age (months)",
    min_value=0,
    max_value=600,
    value=12
)

connections = st.number_input(
    "Number of connections",
    min_value=0,
    max_value=100000,
    value=100
)

posts = st.number_input(
    "Number of posts",
    min_value=0,
    max_value=100000,
    value=10
)

photo = st.selectbox(
    "Has profile photo?",
    ["Yes", "No"]
)

completeness = st.slider(
    "Profile completeness (%)",
    min_value=0,
    max_value=100,
    value=70
)


if st.button("Check Profile"):

    if account_age == 0:
        st.warning("Account age must be greater than zero.")
    else:
        photo_value = 1 if photo == "Yes" else 0

        input_data = pd.DataFrame(
            [[
                account_age,
                connections,
                posts,
                photo_value,
                completeness
            ]],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.warning("⚠️ Suspicious-looking profile")
        else:
            st.success("✅ Genuine-looking profile")

        st.caption(
            "This is an experimental prediction, not proof of fraud."
        )


# -------------------------------
# 9. IMPORTANT LIMITATIONS
# -------------------------------

st.header("ℹ️ Project Limitations")

st.write(
    """
    - The dataset contains synthetic practice examples.
    - The model has not been validated on real LinkedIn profiles.
    - A genuine person may have few connections or no profile photo.
    - A suspicious account may look complete and active.
    - Predictions should not be treated as proof of fraud.
    """
)

st.success("Project demo loaded successfully!")
