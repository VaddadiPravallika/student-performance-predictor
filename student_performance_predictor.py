# ----------------------------------------------------------
# STUDENT PERFORMANCE PREDICTOR - MACHINE LEARNING PROJECT
# ----------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------

# Replace with your dataset
df = pd.read_csv("student_data.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ----------------------------------------------------------
# 2. DATA PREPROCESSING
# ----------------------------------------------------------

# Encode categorical columns
label_encoder = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_encoder.fit_transform(df[col])

print("\nAfter Encoding:")
print(df.head())

# Separate features and target
X = df.drop("final_grade", axis=1)     # target column name
y = df["final_grade"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# 3. MODEL TRAINING
# ----------------------------------------------------------

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ----------------------------------------------------------
# 4. MODEL EVALUATION
# ----------------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("----------------------------")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R²   :", round(r2, 3))

# ----------------------------------------------------------
# 5. PREDICTION FUNCTION
# ----------------------------------------------------------

def predict_student_performance(input_data):
    """
    Predicts student performance based on input features.
    input_data must be a dictionary matching dataset feature names.
    """
    df_input = pd.DataFrame([input_data])
    
    # Encode input the same way
    for col in df_input.select_dtypes(include=['object']).columns:
        df_input[col] = label_encoder.fit_transform(df_input[col])
        
    prediction = model.predict(df_input)[0]
    return prediction


# Example prediction
example_student = {
    "study_hours": 5,
    "attendance": 85,
    "previous_score": 78,
    "internet_access": "yes",
    "parent_education": "high"
}

print("\nExample Prediction:", predict_student_performance(example_student))
