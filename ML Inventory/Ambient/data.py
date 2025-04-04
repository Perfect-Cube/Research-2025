import pandas as pd
import numpy as np
import datetime
import random

# --- Configuration ---
START_DATE = datetime.date(2024, 1, 1)
END_DATE = datetime.date(2025, 12, 31) # Approx 2 years of data
OUTPUT_FILE = "synthetic_assembly_station_data2.csv"

ASSEMBLY_LINES = ['Line A', 'Line B', 'Line C', 'Line D']
STATIONS_PER_LINE = [1, 2, 3]
SHIFTS = ['Morning', 'Night']
OPERATORS = [
    'Alice Smith', 'Bob Johnson', 'Charlie Davis', 'Diana Garcia',
    'Ethan Miller', 'Fiona Wilson', 'George Brown', 'Hannah Jones',
    'Ian Taylor', 'Julia Martinez'
]
PART_NAMES = [
    'Engine Oil Filter', 'Air Filter', 'Spark Plug Set', 'Brake Pad Set',
    'Brake Disc', 'Shock Absorber', 'Timing Belt', 'Battery',
    'Headlight Bulb', 'Windshield Wiper'
]
STATION_STATUSES = ['Active', 'Paused', 'Maintenance']

# --- Station Specific Configuration & State ---
station_config = {}
station_state = {} # To hold dynamic values like stock and next maintenance

part_counter = 0
for line in ASSEMBLY_LINES:
    for station in STATIONS_PER_LINE:
        station_id = f"{line}_S{station}"
        assigned_part = PART_NAMES[part_counter % len(PART_NAMES)]
        min_thresh = random.randint(20, 50)
        max_cap = min_thresh + random.randint(100, 200)
        initial_stock = random.randint(min_thresh + 10, max_cap - 10)
        initial_maint_due = START_DATE + datetime.timedelta(days=random.randint(30, 90))

        station_config[station_id] = {
            'line': line,
            'station': station,
            'part_name': assigned_part,
            'min_threshold': min_thresh,
            'max_capacity': max_cap,
            'base_target': random.randint(80, 150) # Base daily target production
        }
        station_state[station_id] = {
            'current_stock': initial_stock,
            'maintenance_due_date': initial_maint_due
        }
        part_counter += 1

# --- Data Generation ---
data = []
current_date = START_DATE

print("Generating synthetic data...")

while current_date <= END_DATE:
    for station_id, config in station_config.items():
        line = config['line']
        station = config['station']
        part_name = config['part_name']
        min_threshold = config['min_threshold']
        max_capacity = config['max_capacity']
        base_target = config['base_target']

        # Get current state for this station
        state = station_state[station_id]
        current_stock = state['current_stock']
        maintenance_due_date = state['maintenance_due_date']

        # Determine Shift and Operator for the day/station
        shift = random.choice(SHIFTS)
        operator_name = random.choice(OPERATORS)

        # Determine Station Status & Downtime
        downtime_hrs = 0.0
        station_status = 'Active' # Default
        perform_maintenance = False

        if current_date == maintenance_due_date:
            station_status = 'Maintenance'
            downtime_hrs = 8.0 # Assume full shift for maintenance
            perform_maintenance = True
        else:
            # Random chance of being paused (e.g., breaks, minor issues)
            if random.random() < 0.10: # 10% chance of being paused
                station_status = 'Paused'
                downtime_hrs = round(random.uniform(0.5, 4.0), 1)
            else:
                # Active: small chance of minor downtime
                 if random.random() < 0.15: # 15% chance of minor downtime even if active
                      downtime_hrs = round(random.uniform(0.1, 1.0), 1)


        # Determine Target Production (add some daily variation)
        target_production = max(0, int(np.random.normal(base_target, base_target * 0.1)))
        if station_status != 'Active':
             # Reduce target significantly if not active (or set to 0?)
             target_production = int(target_production * 0.2) # Allow some target even if paused?

        # Determine Shortages based on stock *before* production
        shortages = 'None'
        # Assume 1 part consumed per unit produced for simplicity
        if current_stock <= 5 and station_status == 'Active': # Threshold for shortage flag
            shortages = part_name
            if random.random() < 0.8: # High chance status becomes Paused due to shortage
                 station_status = 'Paused'
                 downtime_hrs = max(downtime_hrs, round(random.uniform(2.0, 6.0), 1)) # Increase downtime
        elif random.random() < 0.02: # Small random chance of other shortage
             shortages = random.choice([p for p in PART_NAMES if p != part_name]) # Different part short
             station_status = 'Paused' # Other shortages likely pause the station too
             downtime_hrs = max(downtime_hrs, round(random.uniform(1.0, 3.0), 1))


        # Determine Actual Production
        actual_production = 0
        if station_status == 'Active' and shortages != part_name: # Only produce if active and primary part not short
            # Reduce production based on downtime (assume 8hr shifts)
            efficiency_factor = max(0, (8.0 - downtime_hrs)) / 8.0
            # Reduce production slightly based on stock level (approaching min)
            stock_factor = 1.0 if current_stock > min_threshold else max(0.1, current_stock / min_threshold)
            # Random performance variation
            performance_noise = np.random.normal(1.0, 0.08) # +/- 8% noise

            potential_production = target_production * efficiency_factor * stock_factor * performance_noise

            # Can't produce more than available stock allows
            actual_production = max(0, min(potential_production, current_stock))
            actual_production = int(actual_production) # Production is integer units

        elif station_status == 'Paused':
            # Very low production if paused, maybe 0
            actual_production = int(target_production * random.uniform(0.0, 0.1)) if shortages != part_name else 0
            actual_production = max(0, min(actual_production, current_stock))


        # Calculate Efficiency
        if target_production > 0:
            efficiency = round((actual_production / target_production) * 100, 1)
        else:
            efficiency = 0.0 # Avoid division by zero

        # Determine Defective Units
        defect_rate = 0.01 # Base 1% defect rate
        if efficiency < 80 and efficiency > 0: defect_rate = 0.03 # Higher if low efficiency
        if downtime_hrs > 2: defect_rate *= 1.5 # Higher if significant downtime
        defective_units = int(np.random.binomial(actual_production, defect_rate)) # Binomial distribution
        defective_units = min(defective_units, actual_production) # Cannot have more defects than produced

        # Net good production affects stock consumption
        stock_consumed = actual_production # Assume 1 part per unit attempted/produced

        # Update Stock Available (State for next day)
        previous_stock = current_stock # Store for the record
        current_stock -= stock_consumed
        current_stock = max(0, current_stock) # Stock cannot be negative

        # Simulate Stock Replenishment
        replenished = False
        if current_stock < min_threshold and station_status != 'Maintenance':
            # Replenish only if below threshold AND not during maintenance day
            if random.random() < 0.7: # 70% chance to replenish when needed
                replenish_amount = random.randint(int((max_capacity - min_threshold) * 0.5), int((max_capacity - min_threshold) * 1.2))
                current_stock += replenish_amount
                current_stock = min(current_stock, max_capacity) # Don't exceed max capacity
                replenished = True

        state['current_stock'] = current_stock # Update state for next day


        # Update Maintenance Due Date if maintenance was performed
        if perform_maintenance:
            maintenance_due_date = current_date + datetime.timedelta(days=random.randint(60, 120)) # Schedule next one
            state['maintenance_due_date'] = maintenance_due_date


        # Append record for the current day
        data.append({
            'assembly_line': line,
            'station_number': station,
            'date': current_date,
            'shift': shift,
            'operator_name': operator_name,
            'part_name': part_name,
            'target_production': target_production,
            'actual_production': actual_production,
            'efficiency': efficiency,
            'defective_units': defective_units,
            'minimum_threshold': min_threshold,
            'max_capacity': max_capacity,
            'stock_available': previous_stock, # Record stock *before* production/replenishment
            'downtime_hrs': downtime_hrs,
            'station_status': station_status,
            'maintenance_due_date': maintenance_due_date,
            'shortages': shortages,
        })

    # Increment day
    current_date += datetime.timedelta(days=1)
    if current_date.day == 1 and current_date.month % 3 == 1: # Print progress every quarter
         print(f"Generated data up to {current_date}...")


# Create DataFrame
df = pd.DataFrame(data)

# Ensure correct data types
df['date'] = pd.to_datetime(df['date'])
df['station_number'] = df['station_number'].astype(int)
df['target_production'] = df['target_production'].astype(int)
df['actual_production'] = df['actual_production'].astype(int)
df['defective_units'] = df['defective_units'].astype(int)
df['minimum_threshold'] = df['minimum_threshold'].astype(int)
df['max_capacity'] = df['max_capacity'].astype(int)
df['stock_available'] = df['stock_available'].astype(int)
df['downtime_hrs'] = df['downtime_hrs'].astype(float)
df['efficiency'] = df['efficiency'].astype(float)
df['maintenance_due_date'] = pd.to_datetime(df['maintenance_due_date'])

print("Data generation complete. Saving to CSV...")

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False, date_format='%Y-%m-%d')

print(f"Synthetic dataset saved to {OUTPUT_FILE}")
print(f"Generated {len(df)} rows for {len(ASSEMBLY_LINES) * len(STATIONS_PER_LINE)} stations.")
print("\nDataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nExample of data for one station over time:")
print(df[(df['assembly_line'] == 'Line A') & (df['station_number'] == 1)].tail())