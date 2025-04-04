import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def train_model(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()
    
    if "dt" in df.columns:
        df = df.drop(columns=["dt"])
    
    target = "act_prod"
    
    # Now including 'stn_no' and station-dependent values
    numeric_features = ["stn_no", "tgt_prod", "stock_avl", "min_threshold", "max_cap"]
    categorical_features = ["ass_ln", "shift"]
    
    for col in numeric_features:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in the uploaded CSV file.")
            st.stop()
    
    encoder = OneHotEncoder(drop="first")
    encoded_cats = encoder.fit_transform(df[categorical_features]).toarray()
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out())
    
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_features])
    scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_features)
    
    X = pd.concat([scaled_numeric_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate Model Performance
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, encoder, scaler, numeric_features, categorical_features, df, r2, mae, X_test, y_test, y_pred

st.title("Car Assembly Actual Production Predictor")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    model, encoder, scaler, numeric_features, categorical_features, df, r2, mae, X_test, y_test, y_pred = train_model(uploaded_file)
    st.success("Model trained successfully!")
    
    # Show Model Performance
    st.subheader("Model Performance")
    st.write(f"✅ **R² Score:** {r2:.4f}")
    st.write(f"✅ **Mean Absolute Error (MAE):** {mae:.2f}")

    # Plot Actual vs. Predicted Production
    st.subheader("📊 Actual vs. Predicted Production")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predicted vs Actual")
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed", label="Perfect Fit Line")
    ax.set_xlabel("Actual Production")
    ax.set_ylabel("Predicted Production")
    ax.set_title("Actual vs Predicted Production")
    ax.legend()
    st.pyplot(fig)

    # Input for prediction
    st.header("Predict Actual Production")
    
    # User selects Station Number
    stn_no = st.number_input("Station Number", min_value=1, step=1, value=1)

    # Automatically get min_threshold and max_cap for the selected station number
    station_data = df[df["stn_no"] == stn_no]
    if station_data.empty:
        st.error(f"No data found for Station Number {stn_no}")
        st.stop()
    else:
        min_threshold = station_data["min_threshold"].iloc[0]
        max_cap = station_data["max_cap"].iloc[0]

    tgt_prod = st.number_input("Target Production", value=100)
    stock_avl = st.number_input("Stock Available", value=100)
    
    ass_ln = st.selectbox("Assembly Line", options=df["ass_ln"].unique())
    shift = st.selectbox("Shift", options=df["shift"].unique())

    if st.button("Predict Actual Production"):
        # Prepare input data
        input_numeric = pd.DataFrame({
            "stn_no": [stn_no],
            "tgt_prod": [tgt_prod],
            "stock_avl": [stock_avl],
            "min_threshold": [min_threshold],  # Auto-assigned
            "max_cap": [max_cap]  # Auto-assigned
        })
        input_numeric_scaled = pd.DataFrame(scaler.transform(input_numeric), columns=numeric_features)
        
        input_categorical = pd.DataFrame({
            "ass_ln": [ass_ln],
            "shift": [shift]
        })
        input_encoded = pd.DataFrame(encoder.transform(input_categorical).toarray(),
                                     columns=encoder.get_feature_names_out())
        
        input_features = pd.concat([input_numeric_scaled, input_encoded], axis=1)
        
        prediction = model.predict(input_features)
        st.subheader(f"Predicted Actual Production: {prediction[0]:.2f}")
