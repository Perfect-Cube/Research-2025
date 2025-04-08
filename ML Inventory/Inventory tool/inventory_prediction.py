import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def create_synthetic_data(path):   
    df = pd.read_csv(path)
    n_samples = df.shape[0] # Get the number of samples from the dataframe
    
    # Simulate actual production: target_production minus a penalty due to downtime plus noise
    df['actual_production'] = (df['target_production'] - (df['downtime_hrs'] * 10) +
                               np.random.normal(0, 10, n_samples)).astype(int)
    # Ensure actual production is not negative
    df['actual_production'] = df['actual_production'].apply(lambda x: max(x, 0))
    
    # Feature engineering: calculate speed based on effective operation time (8-hour shift minus downtime)
    df['effective_hours'] = 8 - df['downtime_hrs']
    # Avoid division by zero; compute speed as units produced per effective hour
    df['speed'] = df.apply(lambda row: row['actual_production'] / row['effective_hours'] if row['effective_hours'] > 0 else 0, axis=1)
    df.drop(columns=["effective_hours"], inplace=True)
    
    return df

def train_model(df):
    # Use selected features for training the model
    features = [
        "target_production", "defective_units", "stock_available", 
        "downtime_hrs", "shortages", "speed", "station_number"
    ]
    X = df[features]
    y = df["actual_production"]

    # Create a pipeline with imputation and the XGBoost regressor
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('xgb_regressor', XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42))
    ])

    # Train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nXGBoost Model Performance:")
    print("Mean Absolute Error (MAE): {:.2f}".format(mae))
    print("R² Score: {:.2f}".format(r2))
    
    return pipeline

def main():
    # # Get the number of synthetic data samples from the user.
    # try:
    #     n_samples = int(input("Enter the number of synthetic data samples to generate (e.g., 300): ").strip())
    # except ValueError:
    #     print("Invalid number. Using 300 samples by default.")
    #     n_samples = 300

    # Create synthetic data and display a sample
    df = create_synthetic_data("/content/synthetic_assembly_station_data_v3.csv")
    print("\nSample Synthetic Data:")
    print(df.head())
    
    # # Ask for the CSV filename to save the synthetic dataset
    # csv_filename = input("Enter the filename to save the synthetic data (e.g., 'synthetic_data.csv'): ").strip()
    # df.to_csv(csv_filename, index=False)
    # print(f"Synthetic data saved to {csv_filename}")
    
    # Train the XGBoost model on the synthetic data
    model = train_model(df)
    
    # Get new observation input from the user for prediction.
    print("\nEnter new observation details for predicting actual production:")
    try:
        target_production = float(input("Enter target production (number): ").strip())
        defective_units = float(input("Enter number of defective units: ").strip())
        stock_available = float(input("Enter available stock: ").strip())
        downtime_hrs = float(input("Enter downtime (hours): ").strip())
        shortages = float(input("Enter number of shortages: ").strip())
        station_number = int(input("Enter station number: ").strip())
    except ValueError:
        print("One or more inputs are invalid. Exiting.")
        return

    # Compute speed based on the assumption of an 8-hour shift
    effective_hours = 8 - downtime_hrs if 8 - downtime_hrs > 0 else 1  # Avoid division by zero
    # An approximated actual production based on the target may be used to compute an estimated speed.
    # Here we use target_production as a placeholder to compute speed.
    speed = target_production / effective_hours
    
    new_observation = pd.DataFrame({
        "target_production": [target_production],
        "defective_units": [defective_units],
        "stock_available": [stock_available],
        "downtime_hrs": [downtime_hrs],
        "shortages": [shortages],
        "speed": [speed],
        "station_number": [station_number]
    })
    
    predicted_actual = model.predict(new_observation)[0]
    print("\nPredicted Actual Production for the new observation: {:.2f}".format(int(predicted_actual)))
    print("\n Order:",int(abs(predicted_actual-stock_available)))

if __name__ == "__main__":
    main()
