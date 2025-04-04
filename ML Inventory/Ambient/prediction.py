import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Not strictly needed if only using get_dummies
import matplotlib.pyplot as plt
import warnings
import joblib # For saving features list

warnings.filterwarnings('ignore', category=FutureWarning) # Ignore specific future warnings from pandas/sklearn

# --- Configuration ---
DATA_FILE = "synthetic_assembly_station_data_v2.csv"
TARGET_VARIABLE = 'act_prod' # Predict actual production (proxy for parts consumed)
TEST_SIZE = 0.2 # Use last 20% of data for testing

# --- 1. Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    # Read the header to get exact column names
    header_df = pd.read_csv(DATA_FILE, nrows=0)
    # Define a mapping from file header to desired names
    column_mapping = {
        header_df.columns[0]: 'ass_ln',
        header_df.columns[1]: 'stn_no',
        header_df.columns[2]: 'dt',
        header_df.columns[3]: 'shift',
        header_df.columns[4]: 'tgt_prod',
        header_df.columns[5]: 'act_prod',
        header_df.columns[6]: 'eff_perc', # Renamed eff(%)
        header_df.columns[7]: 'def_unit',
        header_df.columns[8]: 'min_threshold',
        header_df.columns[9]: 'max_cap',
        header_df.columns[10]: 'stock_avl',
        header_df.columns[11]: 'downtime_hrs',
        header_df.columns[12]: 'stn_status',
        header_df.columns[13]: 'shortages',
        header_df.columns[14]: 'part_name'
        # Add operator_name if you keep it
    }
    # Try parsing dates, handling potential errors
    try:
        df = pd.read_csv(DATA_FILE)
        df[header_df.columns[2]] = pd.to_datetime(df[header_df.columns[2]], errors='coerce') # Coerce errors to NaT
    except ValueError: # Fallback if initial parsing fails (e.g., mixed formats)
        df = pd.read_csv(DATA_FILE)
        df[header_df.columns[2]] = pd.to_datetime(df[header_df.columns[2]], infer_datetime_format=True, errors='coerce')

    # Check for parsing errors
    if df[header_df.columns[2]].isnull().any():
         print(f"Warning: Some dates could not be parsed in column '{header_df.columns[2]}'. Rows with invalid dates will be dropped.")
         df.dropna(subset=[header_df.columns[2]], inplace=True)

    df.rename(columns=column_mapping, inplace=True)
    print("Data loaded successfully.")
    print(f"Initial shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {DATA_FILE}")
    exit()
except Exception as e:
    print(f"Error loading or parsing data: {e}")
    exit()

# --- 2. Preprocessing ---
print("Preprocessing data...")

# Combine line and station for a unique key
df['station_key'] = df['ass_ln'] + '_S' + df['stn_no'].astype(str)

# Sort data - CRITICAL for time series features and split
df.sort_values(by=['station_key', 'dt', 'shift'], inplace=True)
df.reset_index(drop=True, inplace=True) # Reset index after sort

# --- 3. Feature Engineering ---
print("Engineering features...")

# Date Features
df['year'] = df['dt'].dt.year
df['month'] = df['dt'].dt.month
df['day'] = df['dt'].dt.day
df['dayofweek'] = df['dt'].dt.dayofweek # Monday=0, Sunday=6
df['dayofyear'] = df['dt'].dt.dayofyear
df['weekofyear'] = df['dt'].dt.isocalendar().week.astype(int)
df['quarter'] = df['dt'].dt.quarter

# Lagged Features (Grouped by station)
lag_features = [TARGET_VARIABLE, 'stock_avl', 'downtime_hrs', 'shortages', 'eff_perc', 'def_unit']
lag_periods = [1, 2, 3, 7] # Lags for 1, 2, 3, 7 periods (shifts) back

for feature in lag_features:
    for lag in lag_periods:
        df[f'{feature}_lag_{lag}'] = df.groupby('station_key')[feature].shift(lag)

# Rolling Window Features (Grouped by station)
rolling_windows = [3, 7, 14] # Windows of 3, 7, 14 periods (shifts)
rolling_features = [TARGET_VARIABLE, 'downtime_hrs', 'eff_perc', 'def_unit', 'shortages']

for feature in rolling_features:
     for window in rolling_windows:
        # Calculate rolling mean, std, min, max - excluding the current row
        df[f'{feature}_roll_mean_{window}'] = df.groupby('station_key')[feature].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'{feature}_roll_std_{window}'] = df.groupby('station_key')[feature].shift(1).rolling(window=window, min_periods=1).std()

# Fill NaN values created by lags/rolling windows (using 0 here, consider imputation after split)
df.fillna(0, inplace=True)

print("Feature engineering completed.")
print(f"Shape after feature engineering: {df.shape}")

# Save original info needed later *before* encoding and dropping columns
original_info_cols = ['dt', 'ass_ln', 'stn_no', 'shift', 'part_name', 'station_key', 'stock_avl', 'min_threshold']
df_original_info = df[original_info_cols].copy()

# --- 4. Encoding ---
print("Encoding categorical features...")
categorical_cols = ['shift', 'stn_status', 'station_key']

# Apply One-Hot Encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Encoding completed.")
print(f"Shape after encoding: {df.shape}")

# --- 5. Train/Test Split (Chronological) ---
print("Splitting data into train and test sets...")
split_index = int(len(df) * (1 - TEST_SIZE))
df_train = df.iloc[:split_index].copy()
df_test = df.iloc[split_index:].copy()

# Get corresponding original info for test set using the same index slice
df_test_original_info = df_original_info.iloc[split_index:].copy()

print(f"Training set shape: {df_train.shape}")
print(f"Testing set shape: {df_test.shape}")
# Use original info df for date range printing
print(f"Training data from {df_original_info.iloc[0]['dt'].date()} to {df_original_info.iloc[split_index-1]['dt'].date()}")
print(f"Testing data from {df_original_info.iloc[split_index]['dt'].date()} to {df_original_info.iloc[-1]['dt'].date()}")

# --- Define Features (X) and Target (y) ---
# Drop columns not used as features FROM THE ENCODED DATAFRAME 'df'
features_to_drop = [
    TARGET_VARIABLE, 'dt', 'ass_ln', 'stn_no', 'part_name',
    'maintenance_due_date', # This column may not exist if not included in original mapping
    'eff_perc' # Calculated from target, potential leakage
]
# Also drop original categorical columns if they somehow persisted (shouldn't with get_dummies)
features_to_drop.extend(['shift', 'stn_status', 'station_key'])

# Start with all columns from the encoded df
features = df.columns.tolist()

# Remove columns that should be dropped
features = [col for col in features if col not in features_to_drop]

# Ensure all selected features exist in both train and test DataFrames
features = [f for f in features if f in df_train.columns and f in df_test.columns]

X_train = df_train[features]
y_train = df_train[TARGET_VARIABLE]
X_test = df_test[features]
y_test = df_test[TARGET_VARIABLE]

print(f"Number of features used: {len(features)}")

# --- 6. Model Training ---
print("Training XGBoost model...")
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000, 'learning_rate': 0.05,
    'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'n_jobs': -1,
    'early_stopping_rounds': 50,
    'eval_metric': 'mae',
    'tree_method': 'hist'
}
model = xgb.XGBRegressor(**xgb_params)

eval_set = [(X_test, y_test)]
model.fit(X_train, y_train,
          eval_set=eval_set,
          verbose=100)
print("Model training completed.")

# --- 7. Prediction & Evaluation ---
print("Evaluating model on test set...")
y_pred = model.predict(X_test)
y_pred = np.maximum(0, y_pred).round().astype(int)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} parts")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} parts")
print(f"R-squared (R²): {r2:.4f}")

# --- 8. Feature Importance ---
print("\n--- Feature Importance ---")
try:
    feature_importance = pd.DataFrame({
        'feature': features, # Use the final list of features
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    print(feature_importance.head(20))
except Exception as e:
    print(f"Could not display feature importance: {e}")


# --- Optional: Plot Actual vs Predicted ---
print("\nPlotting actual vs predicted for a sample station...")

# --- START PLOTTING FIX ---
# 1. Get available keys in the test set's original info
available_test_keys = sorted(df_test_original_info['station_key'].unique())
print(f"(Available station keys in test set: {available_test_keys})")

# 2. Define the desired sample key, and select an available one if needed
sample_station_id_str = 'Line B_S2' # The station we ideally want to plot
plot_key_to_use = None # Initialize variable for the key we will actually use

if sample_station_id_str in available_test_keys:
    plot_key_to_use = sample_station_id_str
    print(f"Plotting requested station: {plot_key_to_use}")
elif available_test_keys: # If desired key not found, but others exist
    plot_key_to_use = available_test_keys[0] # Fallback to the first available key
    print(f"Warning: '{sample_station_id_str}' not in test set. Plotting '{plot_key_to_use}' instead.")
else: # No keys available in test set at all
     print("Warning: No station keys found in the test set original info for plotting.")

# 3. Proceed with plotting only if a valid key was determined
if plot_key_to_use:
    # Use the index from df_test_original_info based on the selected plot_key_to_use
    sample_indices = df_test_original_info[df_test_original_info['station_key'] == plot_key_to_use].index

    if not sample_indices.empty:
        # Select predictions and actuals using the test set index alignment
        y_pred_sample = y_pred[df_test.index.isin(sample_indices)]
        y_test_sample = y_test[df_test.index.isin(sample_indices)]

        # Ensure lengths match before plotting
        if len(y_pred_sample) == len(df_test_original_info.loc[sample_indices]):
            plt.figure(figsize=(15, 6))
            plt.plot(df_test_original_info.loc[sample_indices, 'dt'], y_test_sample, label='Actual Production', marker='.', linestyle='-')
            plt.plot(df_test_original_info.loc[sample_indices, 'dt'], y_pred_sample, label='Predicted Production', marker='x', linestyle='--')
            plt.title(f'Actual vs Predicted Production (Sample Station: {plot_key_to_use})') # Use the actual key plotted
            plt.xlabel('Date')
            plt.ylabel('Parts Produced/Consumed')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
             print(f"Plotting Error: Mismatch in lengths for station {plot_key_to_use}. Cannot plot reliably.")
    # else: # No indices found for the selected key (should be unlikely if chosen from available_test_keys)
    #     print(f"Internal Error: Could not find indices for selected plot key '{plot_key_to_use}'.")

# --- END PLOTTING FIX ---

# --- How much stock is needed? ---
# (The rest of the code from the previous correct answer, which correctly uses df_test_original_info)
print("\nCalculating predicted stock needs...")
# Create results_df USING the df_test_original_info DataFrame
results_df = df_test_original_info[[
    'dt', 'ass_ln', 'stn_no', 'shift', 'part_name', 'station_key',
    'stock_avl', 'min_threshold'
]].copy()
results_df.reset_index(drop=True, inplace=True) # Reset index

# Add predictions
if len(results_df) == len(y_pred):
    results_df['predicted_demand'] = y_pred
else:
    print("Error: Length mismatch between original test info and predictions!")
    results_df['predicted_demand'] = 0

# Perform calculations on results_df
results_df['stock_after_pred_demand'] = results_df['stock_avl'] - results_df['predicted_demand']
results_df['stock_needed_to_reach_min'] = results_df.apply(
    lambda row: max(0, row['min_threshold'] - row['stock_after_pred_demand']),
    axis=1
).round().astype(int)

# Recommended order buffer
buffer_shifts = 3
results_df.sort_values(by=['station_key', 'dt', 'shift'], inplace=True)
results_df['avg_pred_demand_recent'] = results_df.groupby('station_key')['predicted_demand'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
results_df['predicted_demand_buffer'] = (results_df['avg_pred_demand_recent'] * buffer_shifts).round().astype(int)
results_df['predicted_demand_buffer'].fillna(0, inplace=True)

results_df['recommended_order'] = results_df['stock_needed_to_reach_min'] + results_df['predicted_demand_buffer']
results_df['recommended_order'] = results_df['recommended_order'].round().astype(int)

print("\n--- Predicted Demand & Stock Needs (Sample) ---")
print(results_df[['dt', 'station_key', 'shift', 'stock_avl', 'predicted_demand', 'stock_after_pred_demand', 'min_threshold', 'stock_needed_to_reach_min', 'predicted_demand_buffer', 'recommended_order']].head(20))

# --- 10. Save Model and Necessary Objects ---
# ... (saving code remains the same) ...
print("\nSaving model and features...")
try:
    model.save_model("xgb_model.json") # Save XGBoost model
    joblib.dump(features, "model_features.joblib") # Save the list of feature names
    # Save the full original info dataframe for history lookup in UI
    df_original_info.to_pickle("original_info_full.pkl")
    print("Model, features, and original info saved.")
except Exception as e:
    print(f"Error saving model/features: {e}")

print("\nScript finished successfully.")