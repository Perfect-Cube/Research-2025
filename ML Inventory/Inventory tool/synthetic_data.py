# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

# ---------------------------
# 1. Generate Synthetic Dataset
# ---------------------------

# Define parameters
num_days = 60
start_date = datetime(2025, 4, 1)
dates = [start_date + timedelta(days=i) for i in range(num_days)]
shifts = ['Day', 'Night']
showrooms = ['Showroom_1', 'Showroom_2', 'Showroom_3']
stations = ['Station_1', 'Station_2', 'Station_3']
lines = ['A', 'B', 'C']
parts = ['Part_1', 'Part_2', 'Part_3']

# Define thresholds and base inventory parameters
initial_stock = 500  # starting stock for each part
threshold_stock = 100  # if stock falls below this after consumption, a refill is triggered

# List to store data rows
data = []

# Seed for reproducibility
np.random.seed(42)

for date in dates:
    for shift in shifts:
        for showroom in showrooms:
            for station in stations:
                for line in lines:
                    for part in parts:
                        # Simulate consumption based on shift and assembly line
                        # (For example: assembly line A might use more parts)
                        base_consumption = np.random.randint(5, 15)
                        if line == 'A':
                            consumption = base_consumption + np.random.randint(5, 10)
                        elif line == 'B':
                            consumption = base_consumption + np.random.randint(0, 5)
                        else:  # line C
                            consumption = base_consumption

                        # Adjust consumption by shift (e.g., night shift slightly lower consumption)
                        if shift == 'Night':
                            consumption = int(consumption * 0.9)
                        
                        # Simulate parts in stock before consumption (randomly vary around a base value)
                        parts_in_stock = np.random.randint(initial_stock - 50, initial_stock + 50)
                        
                        # Calculate remaining stock after consumption
                        remaining_stock = parts_in_stock - consumption
                        
                        # Determine refill trigger: 1 if remaining stock falls below threshold, else 0.
                        refill_trigger = 1 if remaining_stock < threshold_stock else 0
                        
                        # Append record to data list
                        data.append({
                            'date': date,
                            'shift': shift,
                            'showroom': showroom,
                            'station': station,
                            'assembly_line': line,
                            'part': part,
                            'parts_in_stock': parts_in_stock,
                            'parts_used': consumption,
                            'remaining_stock': remaining_stock,
                            'refill_trigger': refill_trigger
                        })

# Create DataFrame
df = pd.DataFrame(data)
print("Synthetic Dataset Sample:")
print(df.head())

# ---------------------------
# 2. Data Preprocessing & Feature Engineering
# ---------------------------

# For modeling, we convert categorical variables to numeric using one-hot encoding.
df_model = pd.get_dummies(df, columns=['shift', 'showroom', 'station', 'assembly_line', 'part'])

# Features and target
X = df_model.drop(['date', 'remaining_stock', 'refill_trigger'], axis=1)
y = df_model['refill_trigger']

# ---------------------------
# 3. Split Data into Training and Testing Sets
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 4. Train an Example ML Model (RandomForest Classifier)
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# 5. Visualization Dashboard: Inventory Flow Over Time
# ---------------------------

# Aggregate daily inventory: average remaining stock per day across all records.
daily_inventory = df.groupby('date')['remaining_stock'].mean().reset_index()

# Create an interactive line chart with Plotly Express
fig = px.line(daily_inventory, x='date', y='remaining_stock',
              title='Average Daily Inventory Remaining Stock Over Time',
              labels={'date': 'Date', 'remaining_stock': 'Avg. Remaining Stock'})
fig.show()

# Additionally, visualize parts consumption distribution across shifts
plt.figure(figsize=(10, 6))
sns.boxplot(x='shift', y='parts_used', data=df)
plt.title('Parts Used Distribution by Shift')
plt.xlabel('Shift')
plt.ylabel('Parts Used')
plt.show()

# ---------------------------
# 6. (Optional) Save the Synthetic Data for Future Use
# ---------------------------
df.to_csv('synthetic_inventory_data.csv', index=False)
print("\nSynthetic dataset saved as 'synthetic_inventory_data.csv'")
