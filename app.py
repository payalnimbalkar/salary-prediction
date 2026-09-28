import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="wide"
)


# ==========================================
# Load Model
# ==========================================

try:
    model = joblib.load("salary_model.pkl")
except Exception:
    st.error(
        "salary_model.pkl could not be loaded. "
        "Please run 'python train_model.py' first."
    )
    st.stop()


# ==========================================
# Load Dataset
# ==========================================

try:
    df = pd.read_csv("salary_data.csv")
except Exception:
    st.error("salary_data.csv not found.")
    st.stop()


TARGET = "Salary (USD)"

feature_columns = [
    column for column in df.columns
    if column != TARGET
]


# ==========================================
# Header
# ==========================================

st.title("💰 Employee Salary Prediction")

st.write(
    "Enter employee details below to predict the expected salary."
)

st.divider()


# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("📊 Project Information")

st.sidebar.write(
    f"**Dataset Records:** {len(df)}"
)

st.sidebar.write(
    f"**Features:** {len(feature_columns)}"
)

st.sidebar.write(
    "**Model:** Linear Regression"
)

st.sidebar.write(
    "**Target:** Salary (USD)"
)


# ==========================================
# User Inputs
# ==========================================

st.subheader("👤 Employee Information")

input_data = {}

col1, col2 = st.columns(2)


for index, column in enumerate(feature_columns):

    current_column = col1 if index % 2 == 0 else col2

    with current_column:

        # Numerical Input
        if pd.api.types.is_numeric_dtype(df[column]):

            minimum = float(df[column].min())
            maximum = float(df[column].max())
            median = float(df[column].median())

            input_data[column] = st.number_input(
                column,
                min_value=minimum,
                max_value=maximum,
                value=median
            )

        # Categorical Input
        else:

            options = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            options.sort()

            if options:

                input_data[column] = st.selectbox(
                    column,
                    options
                )

            else:

                input_data[column] = st.text_input(
                    column
                )


st.divider()


# ==========================================
# Prediction
# ==========================================

if st.button(
    "🔮 Predict Salary",
    type="primary",
    use_container_width=True
):

    input_df = pd.DataFrame([input_data])

    try:

        prediction = model.predict(input_df)[0]

        st.success("Salary prediction generated successfully!")

        st.subheader("💰 Predicted Salary")

        st.metric(
            label="Expected Salary",
            value=f"${prediction:,.2f}"
        )


        # ==================================
        # Input Summary
        # ==================================

        st.subheader("📋 Input Summary")

        st.dataframe(
            input_df,
            use_container_width=True
        )


    except Exception as error:

        st.error(
            "Prediction failed. Please make sure the "
            "input columns match the training dataset."
        )

        st.exception(error)


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "Salary Prediction | Machine Learning Project"
)