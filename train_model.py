import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("salary_data.csv")

print("Original Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())


# ==========================================
# 2. Remove Duplicate Records
# ==========================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nDuplicates Removed:", before - after)
print("Dataset Shape After Cleaning:", df.shape)


# ==========================================
# 3. Define Target Variable
# ==========================================

TARGET = "Salary (USD)"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found in dataset."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ==========================================
# 4. Identify Feature Types
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ==========================================
# 5. Numerical Preprocessing
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ==========================================
# 6. Categorical Preprocessing
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ==========================================
# 7. Combine Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==========================================
# 8. Create Model Pipeline
# ==========================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ==========================================
# 9. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==========================================
# 10. Train Model
# ==========================================

print("\nTraining Linear Regression Model...")

model.fit(X_train, y_train)

print("Training Completed.")


# ==========================================
# 11. Generate Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 12. Evaluation Metrics
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(f"MAE  : ${mae:,.2f}")
print(f"MSE  : ${mse:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")

print("==========================================")


# ==========================================
# 13. Save Model
# ==========================================

joblib.dump(model, "salary_model.pkl")

print("\nModel saved successfully!")
print("File: salary_model.pkl")