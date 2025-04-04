import pandas as pd
import numpy as np
import datetime
import random

# --- Configuration ---
START_DATE = datetime.date(2025, 1, 1)
END_DATE = datetime.date(2025, 4, 3) 
OUTPUT_FILE = "synthetic_assembly_station_data_v2.csv" # New filename

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
STATION_STATUSES = ['Active', 'Paused', 'Maintenance', 'Shortage Pause'] # Added new status

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

print("Generating synthetic data (v2)...")

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
        current_stock_start_of_day = state['current_stock'] # Stock before any activity today
        maintenance_due_date = state['maintenance_due_date']

        # Determine Shift and Operator for the day/station
        shift = random.choice(SHIFTS)
        operator_name = random.choice(OPERATORS)

        # 1. Determine BASE Station Status & Initial Downtime (Maintenance, Random Pauses)
        downtime_hrs = 0.0
        station_status = 'Active' # Default
        perform_maintenance = False

        if current_date == maintenance_due_date:
            station_status = 'Maintenance'
            downtime_hrs = 8.0 # Assume full shift for maintenance
            perform_maintenance = True
        elif random.random() < 0.05: # Lower chance of purely random pause now
            station_status = 'Paused'
            downtime_hrs = round(random.uniform(0.5, 2.0), 1) # Shorter random pauses
        elif station_status == 'Active' and random.random() < 0.10: # Minor downtime if active
             downtime_hrs = round(random.uniform(0.1, 0.5), 1)


        # 2. Determine Target Production
        target_production = max(0, int(np.random.normal(base_target, base_target * 0.1)))
        if station_status != 'Active':
            # Reduce target significantly if initially paused/maintenance
            target_production = int(target_production * 0.1)

        # 3. Calculate POTENTIAL Production & Parts Needed (before stock check)
        potential_production = 0
        parts_needed = 0
        shortage_qty = 0 # Initialize shortage quantity

        if station_status == 'Active': # Only calculate potential if initially active
            # Calculate efficiency based on initial downtime
            efficiency_factor = max(0, (8.0 - downtime_hrs)) / 8.0
            # Random performance variation
            performance_noise = np.random.normal(1.0, 0.08) # +/- 8% noise
            # How many units would we *like* to produce?
            potential_production = target_production * efficiency_factor * performance_noise
            # How many parts are needed for this potential production? (Assume 1:1)
            parts_needed = int(np.ceil(potential_production)) # Need whole parts

            # 4. Calculate Shortage based on current stock
            shortage_qty = max(0, parts_needed - current_stock_start_of_day)

        # 5. Determine Actual Production (Limited by Stock)
        actual_production = 0
        if station_status == 'Active':
            # Actual production is the potential, unless limited by stock
            actual_production = int(min(potential_production, current_stock_start_of_day))
            actual_production = max(0, actual_production) # Ensure non-negative
        # If initially paused or maintenance, actual production remains 0 or very low (not recalculated here)


        # 6. Update Station Status & Downtime DUE TO Shortage
        if shortage_qty > 0:
            station_status = 'Shortage Pause' # Specific status
            # Add downtime specifically because of the shortage
            shortage_downtime = round(random.uniform(2.0, 6.0) * (shortage_qty / parts_needed if parts_needed > 0 else 1.0) , 1) # Scale downtime by severity
            downtime_hrs = max(downtime_hrs, shortage_downtime) # Ensure the largest downtime is recorded
            downtime_hrs = min(downtime_hrs, 8.0) # Cap downtime at shift length


        # 7. Calculate Final Efficiency
        if target_production > 0:
            # Efficiency is based on what was *actually* produced vs target
            efficiency = round((actual_production / target_production) * 100, 1)
        else:
            efficiency = 0.0

        # 8. Determine Defective Units (Based on actual production)
        defect_rate = 0.01 # Base 1% defect rate
        if 0 < efficiency < 80 : defect_rate = 0.03 # Higher if low efficiency
        if downtime_hrs > 3: defect_rate *= 1.5 # Higher if significant downtime
        if shortage_qty > 0 : defect_rate *= 1.2 # Higher defect rate if running short/stopping
        defective_units = 0
        if actual_production > 0:
             defect_rate = max(0, min(defect_rate, 0.5)) # Cap defect rate
             defective_units = int(np.random.binomial(actual_production, defect_rate))
        defective_units = min(defective_units, actual_production) # Cannot have more defects than produced


        # 9. Update Stock Available (State for next day)
        stock_consumed = actual_production # Parts consumed = parts used in actual production
        current_stock_end_of_day = current_stock_start_of_day - stock_consumed
        current_stock_end_of_day = max(0, current_stock_end_of_day) # Stock cannot be negative

        # Simulate Stock Replenishment
        replenished = False
        # Check stock level *after* consumption
        if current_stock_end_of_day < min_threshold and station_status != 'Maintenance':
            if random.random() < 0.75: # Increased chance to replenish
                replenish_amount = random.randint(int((max_capacity - min_threshold) * 0.6), int((max_capacity - min_threshold) * 1.1))
                current_stock_end_of_day += replenish_amount
                current_stock_end_of_day = min(current_stock_end_of_day, max_capacity) # Don't exceed max capacity
                replenished = True

        state['current_stock'] = current_stock_end_of_day # Update state for next day


        # 10. Update Maintenance Due Date if maintenance was performed
        if perform_maintenance:
            next_maint_date = current_date + datetime.timedelta(days=random.randint(60, 120)) # Schedule next one
            state['maintenance_due_date'] = next_maint_date
            maintenance_due_date = next_maint_date # Use updated date for this record


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
            'stock_available': current_stock_start_of_day, # Record stock *before* production
            'downtime_hrs': downtime_hrs,
            'station_status': station_status,
            'maintenance_due_date': maintenance_due_date,
            'shortages': shortage_qty, # Store the *number* of parts short
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
df['shortages'] = df['shortages'].astype(int) # Shortages is now an integer count

print("Data generation complete. Saving to CSV...")

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False, date_format='%Y-%m-%d')

print(f"Synthetic dataset saved to {OUTPUT_FILE}")
print(f"Generated {len(df)} rows for {len(ASSEMBLY_LINES) * len(STATIONS_PER_LINE)} stations.")
print("\nDataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nExample where shortage likely occurred (Shortages > 0):")
print(df[df['shortages'] > 0].head())
print("\nExample of data for one station over time:")
print(df[(df['assembly_line'] == 'Line A') & (df['station_number'] == 1)].tail())