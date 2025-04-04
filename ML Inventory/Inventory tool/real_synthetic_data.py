# import pandas as pd
# import numpy as np
# from datetime import timedelta, datetime

# # ---------------------------
# # 1. Generate Synthetic Dataset (if not already generated)
# # ---------------------------
# num_days = 60
# start_date = datetime(2025, 4, 1)
# dates = [start_date + timedelta(days=i) for i in range(num_days)]
# shifts = ['Day', 'Night']
# showrooms = ['S1', 'S2', 'S3']  # Using showroom IDs similar to your sample (S1, S2)
# stations = ['Station_1', 'Station_2', 'Station_3']
# lines = ['A', 'B', 'C']
# parts = ['P1', 'P2', 'P3']  # Using Part IDs similar to your sample

# initial_stock = 500
# threshold_stock = 100

# data = []
# np.random.seed(42)

# for date in dates:
#     for shift in shifts:
#         for showroom in showrooms:
#             for station in stations:
#                 for line in lines:
#                     for part in parts:
#                         base_consumption = np.random.randint(5, 15)
#                         if line == 'A':
#                             consumption = base_consumption + np.random.randint(5, 10)
#                         elif line == 'B':
#                             consumption = base_consumption + np.random.randint(0, 5)
#                         else:
#                             consumption = base_consumption

#                         if shift == 'Night':
#                             consumption = int(consumption * 0.9)
                        
#                         stock = np.random.randint(initial_stock - 50, initial_stock + 50)
#                         remaining_stock = stock - consumption
#                         refill_trigger = 1 if remaining_stock < threshold_stock else 0
                        
#                         data.append({
#                             'date': date,
#                             'shift': shift,
#                             'showroom': showroom,
#                             'station': station,
#                             'assembly_line': line,
#                             'part': part,
#                             'parts_in_stock': stock,
#                             'parts_used': consumption,
#                             'remaining_stock': remaining_stock,
#                             'refill_trigger': refill_trigger
#                         })

# df_synthetic = pd.DataFrame(data)
# # Save synthetic dataset if needed
# df_synthetic.to_csv('synthetic_inventory_data.csv', index=False)

# # ---------------------------
# # 2. Prepare Synthetic Data for Merging
# # ---------------------------
# # We want to align with these columns:
# # Date, Showroom_ID, Part_ID, Consumption, Inventory_Level, Orders_Placed, Delivery_Received, Production_Planned, Day_of_Week

# # Convert date column and rename to match provided sample data.
# df_synthetic['Date'] = pd.to_datetime(df_synthetic['date'])
# df_synthetic['Showroom_ID'] = df_synthetic['showroom']
# df_synthetic['Part_ID'] = df_synthetic['part']
# df_synthetic['Consumption'] = df_synthetic['parts_used']
# df_synthetic['Inventory_Level'] = df_synthetic['parts_in_stock']

# # Create additional columns with default values
# df_synthetic['Orders_Placed'] = 0
# df_synthetic['Delivery_Received'] = 0
# df_synthetic['Production_Planned'] = 0
# df_synthetic['Day_of_Week'] = df_synthetic['Date'].dt.day_name()

# # Select and reorder columns to match our target structure
# columns_order = ["Date", "Showroom_ID", "Part_ID", "Consumption", "Inventory_Level", 
#                  "Orders_Placed", "Delivery_Received", "Production_Planned", "Day_of_Week"]

# df_synthetic_aligned = df_synthetic[columns_order]

# # ---------------------------
# # 3. Create DataFrame for the Provided Manual Data
# # ---------------------------
# manual_data = [
#     {"Date": "2023-01-01", "Showroom_ID": "S1", "Part_ID": "P1", "Consumption": 10, "Inventory_Level": 90, "Orders_Placed": 0, "Delivery_Received": 0, "Production_Planned": 100, "Day_of_Week": "Monday"},
#     {"Date": "2023-01-01", "Showroom_ID": "S1", "Part_ID": "P2", "Consumption": 5,  "Inventory_Level": 45, "Orders_Placed": 0, "Delivery_Received": 0, "Production_Planned": 50,  "Day_of_Week": "Monday"},
#     {"Date": "2023-01-02", "Showroom_ID": "S1", "Part_ID": "P1", "Consumption": 12, "Inventory_Level": 78, "Orders_Placed": 20, "Delivery_Received": 0, "Production_Planned": 120, "Day_of_Week": "Tuesday"},
#     {"Date": "2023-01-02", "Showroom_ID": "S2", "Part_ID": "P1", "Consumption": 15, "Inventory_Level": 85, "Orders_Placed": 0,  "Delivery_Received": 0, "Production_Planned": 150, "Day_of_Week": "Tuesday"}
# ]

# df_manual = pd.DataFrame(manual_data)
# df_manual['Date'] = pd.to_datetime(df_manual['Date'])

# # ---------------------------
# # 4. Combine the Two DataFrames
# # ---------------------------
# df_combined = pd.concat([df_synthetic_aligned, df_manual], ignore_index=True)

# # Save the combined dataset to a new CSV file
# df_combined.to_csv('combined_inventory_data.csv', index=False)
# print("Combined inventory data saved as 'combined_inventory_data.csv'")



import pandas as pd
import numpy as np
from datetime import timedelta, datetime

# ---------------------------
# Parameters for Synthetic Data Generation
# ---------------------------
num_days = 60  # number of days to simulate
start_date = datetime(2025, 4, 1)
dates = [start_date + timedelta(days=i) for i in range(num_days)]
showrooms = ['S1', 'S2', 'S3']  # list of showroom IDs
parts = ['P1', 'P2', 'P3']      # list of part IDs

# ---------------------------
# Initialize Data Storage
# ---------------------------
data = []
np.random.seed(42)

# ---------------------------
# Generate Synthetic Records
# ---------------------------
for date in dates:
    for showroom in showrooms:
        for part in parts:
            # Simulate consumption: a random integer between 5 and 20
            consumption = np.random.randint(5, 21)
            
            # Assume an initial inventory level for each record (for simplicity)
            initial_inventory = 100  
            inventory_level = initial_inventory - consumption
            
            # Orders placed: randomly decide if an order is triggered (e.g., 30% chance)
            # When triggered, place an order of a random quantity between 10 and 30
            orders_placed = np.random.randint(10, 31) if np.random.rand() < 0.3 else 0
            
            # Delivery received: if an order was placed, assume full delivery is received
            delivery_received = orders_placed
            
            # Production planned: simulate a planned production value, could be based on forecast or random
            production_planned = np.random.randint(80, 151)
            
            # Determine day of the week
            day_of_week = date.strftime("%A")
            
            # Append record to data list
            data.append({
                "Date": date,
                "Showroom_ID": showroom,
                "Part_ID": part,
                "Consumption": consumption,
                "Inventory_Level": inventory_level,
                "Orders_Placed": orders_placed,
                "Delivery_Received": delivery_received,
                "Production_Planned": production_planned,
                "Day_of_Week": day_of_week
            })

# ---------------------------
# Create DataFrame and Save CSV
# ---------------------------
df = pd.DataFrame(data)
csv_filename = 'synthetic_inventory_data_extended.csv'
df.to_csv(csv_filename, index=False)
print(f"Extended synthetic inventory data saved as '{csv_filename}'")
print(df.head())
