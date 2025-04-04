import os
import csv
import datetime
import uuid
import time
import logging
import threading
import json
# import requests # Not used directly in the provided code
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd # <-- ADDED IMPORT

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AmbientInventorySystem")

# Constants
SHOWROOM_CSV = "showroom1.csv"
WAREHOUSE_CSV = "warehouse1.csv"
SUPPLIER_CSV = "supplier1.csv"
ORDER_HISTORY_CSV = "order_history1.csv"
EVENTS_LOG_CSV = "events_log.csv"
AGENT_DECISIONS_CSV = "agent_decisions.csv"
ASSEMBLY_LINE_CSV = "car_assembly_line_data.csv" # <-- ADDED CONSTANT

# Order status constants
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_IN_TRANSIT = "in_transit"
STATUS_DELIVERED = "delivered"
STATUS_CANCELLED = "cancelled"

# Order type constants
TYPE_SHOWROOM_TO_WAREHOUSE = "showroom_to_warehouse"
TYPE_WAREHOUSE_TO_SUPPLIER = "warehouse_to_supplier"

# Configure Gemini API
# IMPORTANT: Replace with your actual API key retrieval method (e.g., from .env)
api_key = os.getenv("GEMINI_API_KEY") # Use environment variable
if not api_key:
    logger.warning("GEMINI_API_KEY not found in environment variables. AI features will be disabled.")
    # You might want to exit or handle this case differently
    # For demonstration, let's try a placeholder, but real use requires a valid key
    api_key = "YOUR_API_KEY_HERE" # Replace or ensure it's set in .env

# Check if the key is still the placeholder
if api_key == "YOUR_API_KEY_HERE" or not api_key:
     logger.warning("Using placeholder or invalid API key for Gemini. AI features might not work.")
     GEMINI_API_KEY = None # Explicitly disable AI if key isn't valid/set
else:
    try:
        genai.configure(api_key=api_key)
        GEMINI_API_KEY = api_key
        model = genai.GenerativeModel('gemini-1.5-flash-001') # Or other suitable model
        logger.info("Gemini API configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}. AI features disabled.")
        GEMINI_API_KEY = None

# --- InventoryItem, Order, Event, AgentDecision classes remain the same ---
# --- (Keep the classes from your original inventory_system2.py here) ---
class InventoryItem:
    def __init__(self, part_id, part_name, category, quantity, min_threshold=0, max_capacity=100,
                price=0.0, lead_time_days=0, available_quantity=0, historical_demand=None,
                seasonal_factors=None, vendor_reliability=1.0):
        self.part_id = part_id
        self.part_name = part_name
        self.category = category
        self.quantity = int(quantity)
        self.min_threshold = int(min_threshold) if min_threshold else 0
        self.max_capacity = int(max_capacity) if max_capacity else 100
        self.price = float(price) if price else 0.0
        self.lead_time_days = int(lead_time_days) if lead_time_days else 0
        self.available_quantity = int(available_quantity) if available_quantity else 0
        # Simplified historical demand if not provided
        self.historical_demand = historical_demand or []
        self.seasonal_factors = seasonal_factors or {}
        self.vendor_reliability = float(vendor_reliability) if vendor_reliability else 1.0
        self.last_updated = datetime.datetime.now()
        self.demand_trend = 0.0  # Positive means increasing demand

    def __str__(self):
        return f"{self.part_id} - {self.part_name} ({self.quantity} units)"

    def to_dict(self):
        return {
            "part_id": self.part_id,
            "part_name": self.part_name,
            "category": self.category,
            "quantity": self.quantity,
            "min_threshold": self.min_threshold,
            "max_capacity": self.max_capacity,
            "price": self.price,
            "lead_time_days": self.lead_time_days,
            "available_quantity": self.available_quantity,
            "historical_demand": self.historical_demand,
            "seasonal_factors": self.seasonal_factors,
            "vendor_reliability": self.vendor_reliability,
            "demand_trend": self.demand_trend,
            "last_updated": self.last_updated.isoformat() # Added for potential future use
        }

class Order:
    def __init__(self, order_id, part_id, quantity, order_type, status=STATUS_PENDING,
                order_date=None, expected_delivery=None, priority=1,
                ai_recommendation=None, confidence_score=0.0):
        self.order_id = order_id
        self.part_id = part_id
        self.quantity = int(quantity)
        self.order_type = order_type
        self.status = status
        self.order_date = order_date or datetime.datetime.now().strftime("%Y-%m-%d")
        self.expected_delivery = expected_delivery
        self.priority = int(priority) if priority else 1 # 1-5, with 5 being highest priority
        self.ai_recommendation = ai_recommendation
        self.confidence_score = float(confidence_score) if confidence_score else 0.0
        self.status_history = [(status, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]
        self.notes = []

    def __str__(self):
        return f"Order {self.order_id}: {self.quantity} units of {self.part_id}, Status: {self.status}, Priority: {self.priority}"

    def to_dict(self):
        # Convert status history tuples to list of dicts for easier JSON serialization if needed
        status_hist_list = [{"status": s, "timestamp": t} for s, t in self.status_history]
        notes_list = [{"timestamp": t, "note": n} for t, n in self.notes]

        return {
            "order_id": self.order_id,
            "part_id": self.part_id,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status,
            "order_date": self.order_date,
            "expected_delivery": self.expected_delivery,
            "priority": self.priority,
            "ai_recommendation": self.ai_recommendation,
            "confidence_score": self.confidence_score,
            "status_history": status_hist_list,
            "notes": notes_list
        }

    def update_status(self, new_status, note=None):
        """Update order status and record the change in history"""
        if self.status == new_status:
            return # Avoid redundant updates

        self.status = new_status
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status_history.append((new_status, timestamp))
        if note:
            # Store notes as tuples (timestamp, note_text)
            self.notes.append((timestamp, note))
        logger.info(f"Order {self.order_id} status updated to {new_status}. Note: {note or 'N/A'}")


class Event:
    def __init__(self, event_id, event_type, description, timestamp=None,
                 entity_id=None, entity_type=None, severity=1, ai_analysis=None):
        self.event_id = event_id
        self.event_type = event_type
        self.description = description
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.severity = int(severity) if severity else 1 # 1-5, with 5 being most severe
        self.ai_analysis = ai_analysis

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "description": self.description,
            "timestamp": self.timestamp,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "severity": self.severity,
            "ai_analysis": self.ai_analysis
        }

class AgentDecision:
    def __init__(self, decision_id, agent_type, decision_type, entity_id,
                 timestamp=None, reasoning=None, ai_involved=False,
                 confidence_score=0.0, alternatives=None):
        self.decision_id = decision_id
        self.agent_type = agent_type
        self.decision_type = decision_type
        self.entity_id = entity_id # Can be order_id, part_id, event_id etc.
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.reasoning = reasoning
        self.ai_involved = bool(ai_involved)
        self.confidence_score = float(confidence_score) if confidence_score else 0.0
        self.alternatives = alternatives or [] # List of alternative actions considered

    def to_dict(self):
        return {
            "decision_id": self.decision_id,
            "agent_type": self.agent_type,
            "decision_type": self.decision_type,
            "entity_id": self.entity_id,
            "timestamp": self.timestamp,
            "reasoning": self.reasoning,
            "ai_involved": self.ai_involved,
            "confidence_score": self.confidence_score,
            # Ensure alternatives are serializable (e.g., list of strings or simple dicts)
            "alternatives": self.alternatives
        }

class InventoryManager:
    def __init__(self, showroom_file=SHOWROOM_CSV, warehouse_file=WAREHOUSE_CSV,
                supplier_file=SUPPLIER_CSV, order_history_file=ORDER_HISTORY_CSV,
                events_log_file=EVENTS_LOG_CSV, agent_decisions_file=AGENT_DECISIONS_CSV,
                assembly_line_file=ASSEMBLY_LINE_CSV): # <-- ADDED assembly_line_file
        self.showroom_file = showroom_file
        self.warehouse_file = warehouse_file
        self.supplier_file = supplier_file
        self.order_history_file = order_history_file
        self.events_log_file = events_log_file
        self.agent_decisions_file = agent_decisions_file
        self.assembly_line_file = assembly_line_file # <-- STORE FILENAME

        # Load inventory data
        # Use locks early to prevent race conditions during initialization if accessed concurrently
        self.inventory_lock = threading.RLock()
        self.order_lock = threading.RLock()
        self.event_lock = threading.RLock()
        self.decision_lock = threading.RLock()
        self.assembly_data_lock = threading.RLock() # Lock for assembly data access

        with self.inventory_lock:
            self.showroom_inventory = self._load_inventory(showroom_file, "Showroom")
            self.warehouse_inventory = self._load_inventory(warehouse_file, "Warehouse")
            self.supplier_inventory = self._load_inventory(supplier_file, "Supplier", is_supplier=True)
        with self.order_lock:
            self.order_history = self._load_orders(order_history_file)
        with self.event_lock:
            self.events_log = self._load_events(events_log_file)
        with self.decision_lock:
            self.agent_decisions = self._load_decisions(agent_decisions_file)
        with self.assembly_data_lock:
            self.assembly_line_data = self._load_assembly_data(self.assembly_line_file) # <-- LOAD ASSEMBLY DATA

        # Helper maps for quick lookups
        self._build_part_name_maps()


    def _build_part_name_maps(self):
        """Create maps from part_name to part_id for easier lookup."""
        self.warehouse_part_name_to_id = {}
        self.supplier_part_name_to_id = {}

        with self.inventory_lock:
            for part_id, item in self.warehouse_inventory.items():
                self.warehouse_part_name_to_id[item.part_name.lower()] = part_id
            for part_id, item in self.supplier_inventory.items():
                # Suppliers might offer the same part name; store a list of supplier part IDs
                part_name_lower = item.part_name.lower()
                if part_name_lower not in self.supplier_part_name_to_id:
                    self.supplier_part_name_to_id[part_name_lower] = []
                self.supplier_part_name_to_id[part_name_lower].append(part_id)

    def get_warehouse_part_id_by_name(self, part_name):
        """Find warehouse part_id by its name (case-insensitive)."""
        return self.warehouse_part_name_to_id.get(part_name.lower())

    def get_supplier_part_ids_by_name(self, part_name):
        """Find supplier part_ids by name (case-insensitive). Returns a list."""
        return self.supplier_part_name_to_id.get(part_name.lower(), [])


    def _load_inventory(self, filename, inventory_type, is_supplier=False):
        inventory = {}
        required_cols_supplier = ["part_id", "part_name", "category", "available_quantity", "price", "lead_time_days"]
        required_cols_internal = ["part_id", "part_name", "category", "quantity", "min_threshold", "max_capacity"]
        required_cols = required_cols_supplier if is_supplier else required_cols_internal

        if not os.path.exists(filename):
            logger.warning(f"{inventory_type} file {filename} not found. Creating an empty inventory.")
            # Optionally create the file with headers
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                   writer = csv.DictWriter(csvfile, fieldnames=required_cols)
                   writer.writeheader()
                logger.info(f"Created empty {inventory_type} file: {filename}")
            except Exception as e:
                logger.error(f"Could not create empty file {filename}: {e}")
            return inventory

        try:
            with open(filename, 'r', encoding='utf-8-sig', newline='') as csvfile:
                # Check for empty file
                first_char = csvfile.read(1)
                if not first_char:
                    logger.warning(f"{inventory_type} file {filename} is empty. Returning empty inventory.")
                    # Write header if empty
                    csvfile.close()
                    with open(filename, 'w', newline='', encoding='utf-8') as write_csvfile:
                        writer = csv.DictWriter(write_csvfile, fieldnames=required_cols)
                        writer.writeheader()
                    return inventory
                csvfile.seek(0) # Reset cursor

                reader = csv.DictReader(csvfile)

                # Verify header
                header = reader.fieldnames
                if not header or not all(col in header for col in required_cols):
                    logger.error(f"Invalid header in {filename}. Expected {required_cols}, got {header}. Cannot load inventory.")
                    return inventory # Return empty dict as loading failed

                line_num = 1 # For error reporting
                for row in reader:
                    line_num += 1
                    try:
                        # Basic validation - check if required keys exist and have non-empty values
                        if not all(row.get(col) for col in required_cols):
                           logger.warning(f"Skipping row {line_num} in {filename}: Missing or empty required value(s). Row: {row}")
                           continue

                        if is_supplier:
                            item = InventoryItem(
                                row["part_id"], row["part_name"], row["category"],
                                0, # quantity is N/A for supplier view itself
                                0, # min_threshold is N/A
                                0, # max_capacity is N/A
                                row["price"],
                                row["lead_time_days"],
                                row["available_quantity"],
                                vendor_reliability=row.get("vendor_reliability") # Optional
                            )
                        else:
                            item = InventoryItem(
                                row["part_id"], row["part_name"], row["category"],
                                row["quantity"],
                                row.get("min_threshold"), # Optional
                                row.get("max_capacity")  # Optional
                            )

                        if row["part_id"] in inventory:
                             logger.warning(f"Duplicate part_id '{row['part_id']}' found in {filename}. Overwriting previous entry.")
                        inventory[row["part_id"]] = item

                    except KeyError as ke:
                        logger.error(f"Missing expected column {ke} in row {line_num} of {filename}. Skipping row: {row}")
                    except ValueError as ve:
                         logger.error(f"Data conversion error in row {line_num} of {filename}: {ve}. Skipping row: {row}")
                    except Exception as e_row:
                        logger.error(f"Unexpected error processing row {line_num} in {filename}: {e_row}. Skipping row: {row}")

        except FileNotFoundError:
             logger.error(f"{inventory_type} file {filename} not found during read attempt (should have been checked).")
        except Exception as e:
            logger.error(f"Error loading inventory from {filename}: {str(e)}", exc_info=True) # Log traceback
        return inventory

    def _load_orders(self, filename):
        orders = []
        required_cols = ["order_id", "part_id", "quantity", "order_type", "status", "order_date"]
        if not os.path.exists(filename):
            logger.warning(f"Order history file {filename} not found. Creating an empty order list.")
            # Create file with headers
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    # Add optional fields to header creation
                    fieldnames = required_cols + ["expected_delivery", "priority", "ai_recommendation", "confidence_score"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                logger.info(f"Created empty order history file: {filename}")
            except Exception as e:
                logger.error(f"Could not create empty file {filename}: {e}")
            return orders

        try:
            with open(filename, 'r', encoding='utf-8-sig', newline='') as csvfile:
                first_char = csvfile.read(1)
                if not first_char:
                    logger.warning(f"Order history file {filename} is empty.")
                    csvfile.close()
                    # Ensure header exists even if empty
                    with open(filename, 'w', newline='', encoding='utf-8') as write_csvfile:
                         fieldnames = required_cols + ["expected_delivery", "priority", "ai_recommendation", "confidence_score"]
                         writer = csv.DictWriter(write_csvfile, fieldnames=fieldnames)
                         writer.writeheader()
                    return orders
                csvfile.seek(0)

                reader = csv.DictReader(csvfile)
                header = reader.fieldnames
                if not header or not all(col in header for col in required_cols):
                     logger.error(f"Invalid header in {filename}. Expected at least {required_cols}, got {header}. Cannot load orders.")
                     return orders

                line_num = 1
                for row in reader:
                    line_num += 1
                    try:
                         # Basic validation
                        if not all(row.get(col) for col in required_cols):
                           logger.warning(f"Skipping row {line_num} in {filename}: Missing or empty required value(s). Row: {row}")
                           continue

                        order = Order(
                            row["order_id"], row["part_id"], row["quantity"],
                            row["order_type"], row["status"], row["order_date"],
                            row.get("expected_delivery"), # Optional
                            priority=row.get("priority"), # Optional
                            ai_recommendation=row.get("ai_recommendation"), # Optional
                            confidence_score=row.get("confidence_score") # Optional
                        )
                        # TODO: Potentially load status_history and notes if they were saved in a structured way
                        orders.append(order)
                    except KeyError as ke:
                        logger.error(f"Missing expected column {ke} in row {line_num} of {filename}. Skipping row: {row}")
                    except ValueError as ve:
                         logger.error(f"Data conversion error in row {line_num} of {filename}: {ve}. Skipping row: {row}")
                    except Exception as e_row:
                        logger.error(f"Unexpected error processing row {line_num} in {filename}: {e_row}. Skipping row: {row}")

        except FileNotFoundError:
            logger.error(f"Order history file {filename} not found during read.")
        except Exception as e:
            logger.error(f"Error loading orders from {filename}: {str(e)}", exc_info=True)
        return orders

    def _load_events(self, filename):
        events = []
        required_cols = ["event_id", "event_type", "description", "timestamp"]
        # Define headers here for consistency in creation and loading
        fieldnames = ["event_id", "event_type", "description", "timestamp", "entity_id",
                      "entity_type", "severity", "ai_analysis"]

        if not os.path.exists(filename):
            logger.warning(f"Events log file {filename} not found. Creating empty log.")
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                logger.info(f"Created empty events log file: {filename}")
            except Exception as e:
                logger.error(f"Could not create empty file {filename}: {e}")
            return events

        try:
            with open(filename, 'r', encoding='utf-8-sig', newline='') as csvfile:
                first_char = csvfile.read(1)
                if not first_char:
                    logger.warning(f"Events log file {filename} is empty.")
                    csvfile.close()
                    with open(filename, 'w', newline='', encoding='utf-8') as write_csvfile:
                         writer = csv.DictWriter(write_csvfile, fieldnames=fieldnames)
                         writer.writeheader()
                    return events
                csvfile.seek(0)

                reader = csv.DictReader(csvfile)
                header = reader.fieldnames
                # Check against the full expected fieldnames list
                if not header or not all(col in header for col in required_cols):
                    logger.error(f"Invalid header in {filename}. Expected at least {required_cols}, got {header}. Cannot load events.")
                    return events

                line_num = 1
                for row in reader:
                    line_num += 1
                    try:
                        if not all(row.get(col) for col in required_cols):
                           logger.warning(f"Skipping row {line_num} in {filename}: Missing or empty required value(s). Row: {row}")
                           continue

                        event = Event(
                            row["event_id"], row["event_type"], row["description"],
                            row["timestamp"],
                            entity_id=row.get("entity_id"),
                            entity_type=row.get("entity_type"),
                            severity=row.get("severity"), # Let Event class handle default
                            ai_analysis=row.get("ai_analysis")
                        )
                        events.append(event)
                    except KeyError as ke:
                        logger.error(f"Missing expected column {ke} in row {line_num} of {filename}. Skipping row: {row}")
                    except ValueError as ve:
                         logger.error(f"Data conversion error in row {line_num} of {filename}: {ve}. Skipping row: {row}")
                    except Exception as e_row:
                        logger.error(f"Unexpected error processing row {line_num} in {filename}: {e_row}. Skipping row: {row}")

        except FileNotFoundError:
            logger.error(f"Events log file {filename} not found during read.")
        except Exception as e:
            logger.error(f"Error loading events from {filename}: {str(e)}", exc_info=True)
        return events

    def _load_decisions(self, filename):
        decisions = []
        required_cols = ["decision_id", "agent_type", "decision_type", "entity_id", "timestamp"]
        fieldnames = ["decision_id", "agent_type", "decision_type", "entity_id", "timestamp",
                      "reasoning", "ai_involved", "confidence_score", "alternatives"]

        if not os.path.exists(filename):
            logger.warning(f"Decisions log file {filename} not found. Creating empty log.")
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                logger.info(f"Created empty decisions log file: {filename}")
            except Exception as e:
                logger.error(f"Could not create empty file {filename}: {e}")
            return decisions

        try:
            with open(filename, 'r', encoding='utf-8-sig', newline='') as csvfile:
                first_char = csvfile.read(1)
                if not first_char:
                    logger.warning(f"Decisions log file {filename} is empty.")
                    csvfile.close()
                    with open(filename, 'w', newline='', encoding='utf-8') as write_csvfile:
                         writer = csv.DictWriter(write_csvfile, fieldnames=fieldnames)
                         writer.writeheader()
                    return decisions
                csvfile.seek(0)

                reader = csv.DictReader(csvfile)
                header = reader.fieldnames
                if not header or not all(col in header for col in required_cols):
                    logger.error(f"Invalid header in {filename}. Expected at least {required_cols}, got {header}. Cannot load decisions.")
                    return decisions

                line_num = 1
                for row in reader:
                    line_num += 1
                    try:
                        if not all(row.get(col) for col in required_cols):
                           logger.warning(f"Skipping row {line_num} in {filename}: Missing or empty required value(s). Row: {row}")
                           continue

                        alternatives = []
                        if row.get("alternatives"):
                            try:
                                # Attempt to load alternatives, assuming they were saved as JSON strings
                                alts_loaded = json.loads(row["alternatives"])
                                # Basic check if it's a list
                                if isinstance(alts_loaded, list):
                                    alternatives = alts_loaded
                                else:
                                    logger.warning(f"Alternatives in row {line_num} of {filename} is not a list after JSON parsing. Row: {row}")
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse alternatives JSON in row {line_num} of {filename}. Value: '{row['alternatives']}'. Row: {row}")
                            except Exception as e_alt:
                                 logger.error(f"Error processing alternatives in row {line_num} of {filename}: {e_alt}. Row: {row}")


                        decision = AgentDecision(
                            row["decision_id"], row["agent_type"], row["decision_type"],
                            row["entity_id"], row["timestamp"],
                            reasoning=row.get("reasoning"),
                            ai_involved=row.get("ai_involved", "False").lower() == "true", # Safer boolean conversion
                            confidence_score=row.get("confidence_score"), # Let class handle default/conversion
                            alternatives=alternatives
                        )
                        decisions.append(decision)
                    except KeyError as ke:
                        logger.error(f"Missing expected column {ke} in row {line_num} of {filename}. Skipping row: {row}")
                    except ValueError as ve:
                         logger.error(f"Data conversion error in row {line_num} of {filename}: {ve}. Skipping row: {row}")
                    except Exception as e_row:
                        logger.error(f"Unexpected error processing row {line_num} in {filename}: {e_row}. Skipping row: {row}")

        except FileNotFoundError:
            logger.error(f"Decisions log file {filename} not found during read.")
        except Exception as e:
            logger.error(f"Error loading decisions from {filename}: {str(e)}", exc_info=True)
        return decisions

    def _load_assembly_data(self, filename):
        """Loads assembly line data using pandas."""
        if not os.path.exists(filename):
            logger.warning(f"Assembly line data file {filename} not found. Returning empty DataFrame.")
            return pd.DataFrame() # Return empty DataFrame

        try:
            with self.assembly_data_lock: # Use lock for reading
                df = pd.read_csv(filename, encoding='utf-8-sig')
                # Basic validation: Check for essential columns
                required_cols = ['Date', 'Component Shortages', 'Actual Production', 'Assembly Line']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Assembly data file {filename} missing required columns. Expected: {required_cols}. Got: {list(df.columns)}. Returning empty DataFrame.")
                    return pd.DataFrame()

                # Convert 'Date' column to datetime objects
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as date_err:
                    logger.error(f"Error converting 'Date' column in {filename} to datetime: {date_err}. Proceeding without date conversion.", exc_info=True)

                logger.info(f"Successfully loaded assembly data from {filename}. Shape: {df.shape}")
                return df

        except pd.errors.EmptyDataError:
             logger.warning(f"Assembly data file {filename} is empty. Returning empty DataFrame.")
             return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading assembly data from {filename}: {str(e)}", exc_info=True)
            return pd.DataFrame() # Return empty DataFrame on error

    def refresh_assembly_data(self):
        """Reloads assembly data from the file."""
        logger.info(f"Refreshing assembly data from {self.assembly_line_file}...")
        with self.assembly_data_lock: # Lock during update
             self.assembly_line_data = self._load_assembly_data(self.assembly_line_file)
             # Optionally rebuild maps if assembly data influences part names somehow (unlikely here)
             # self._build_part_name_maps() # Probably not needed here


    # --- save_inventory and its helpers remain largely the same ---
    # --- Make sure they use the locks consistently ---
    def save_inventory(self):
        """Save all inventory data to CSV files"""
        logger.info("Attempting to save all data...")
        try:
            with self.inventory_lock:
                self._save_inventory(self.showroom_file, self.showroom_inventory, "Showroom")
                self._save_inventory(self.warehouse_file, self.warehouse_inventory, "Warehouse")
                self._save_supplier_inventory(self.supplier_file, self.supplier_inventory)

            with self.order_lock:
                self._save_orders(self.order_history_file, self.order_history)

            with self.event_lock:
                self._save_events(self.events_log_file, self.events_log)

            with self.decision_lock:
                self._save_decisions(self.agent_decisions_file, self.agent_decisions)
            logger.info("All data saved successfully.")
        except Exception as e:
             logger.error(f"Error during the save_inventory process: {e}", exc_info=True)


    def _save_inventory(self, filename, inventory, inventory_type):
        # Define fields based on InventoryItem attributes (excluding internal ones like last_updated)
        fieldnames = ["part_id", "part_name", "category", "quantity", "min_threshold", "max_capacity"]
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') # Ignore extra fields
                writer.writeheader()
                for item in inventory.values():
                    # Prepare data from item object
                    row_data = {
                        "part_id": item.part_id,
                        "part_name": item.part_name,
                        "category": item.category,
                        "quantity": item.quantity,
                        "min_threshold": item.min_threshold,
                        "max_capacity": item.max_capacity
                    }
                    writer.writerow(row_data)
            logger.debug(f"Saved {inventory_type} inventory to {filename}")
        except Exception as e:
            logger.error(f"Error saving {inventory_type} inventory to {filename}: {str(e)}", exc_info=True)

    def _save_supplier_inventory(self, filename, inventory):
        # Define fields based on InventoryItem attributes relevant to suppliers
        fieldnames = ["part_id", "part_name", "category", "available_quantity", "price", "lead_time_days", "vendor_reliability"]
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for item in inventory.values():
                     # Prepare data from item object
                    row_data = {
                        "part_id": item.part_id,
                        "part_name": item.part_name,
                        "category": item.category,
                        "available_quantity": item.available_quantity,
                        "price": item.price,
                        "lead_time_days": item.lead_time_days,
                        "vendor_reliability": item.vendor_reliability
                    }
                    writer.writerow(row_data)
            logger.debug(f"Saved supplier inventory to {filename}")
        except Exception as e:
            logger.error(f"Error saving supplier inventory to {filename}: {str(e)}", exc_info=True)

    def _save_orders(self, filename, orders):
         # Define fields based on Order attributes, matching loading fields
        fieldnames = ["order_id", "part_id", "quantity", "order_type", "status", "order_date",
                      "expected_delivery", "priority", "ai_recommendation", "confidence_score"]
        # Add fields for status history and notes if you want to save them (might get complex in CSV)
        # fieldnames.extend(["status_history_json", "notes_json"]) # Example

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for order in orders:
                    # Use the to_dict method but only select the relevant fields for CSV
                    order_data = order.to_dict()
                    row_data = {k: order_data.get(k) for k in fieldnames}

                    # Example: Serialize complex fields if needed
                    # row_data["status_history_json"] = json.dumps(order_data["status_history"])
                    # row_data["notes_json"] = json.dumps(order_data["notes"])

                    writer.writerow(row_data)
            logger.debug(f"Saved order history to {filename}")
        except Exception as e:
            logger.error(f"Error saving orders to {filename}: {str(e)}", exc_info=True)

    def _save_events(self, filename, events):
        # Use fields from Event.to_dict() keys
        if not events: # Handle empty list case
             fieldnames = ["event_id", "event_type", "description", "timestamp", "entity_id",
                           "entity_type", "severity", "ai_analysis"] # Default fieldnames
        else:
             fieldnames = list(events[0].to_dict().keys()) # Get keys from first event

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for event in events:
                    writer.writerow(event.to_dict())
            logger.debug(f"Saved events log to {filename}")
        except Exception as e:
            logger.error(f"Error saving events to {filename}: {str(e)}", exc_info=True)

    def _save_decisions(self, filename, decisions):
        # Use fields from AgentDecision.to_dict() keys
        if not decisions:
            fieldnames = ["decision_id", "agent_type", "decision_type", "entity_id", "timestamp",
                          "reasoning", "ai_involved", "confidence_score", "alternatives"] # Default
        else:
            fieldnames = list(decisions[0].to_dict().keys())

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for decision in decisions:
                    row_data = decision.to_dict()
                    # Ensure alternatives is a JSON string for CSV compatibility
                    if isinstance(row_data.get("alternatives"), list):
                         try:
                            row_data["alternatives"] = json.dumps(decision.alternatives)
                         except Exception as json_err:
                             logger.error(f"Could not serialize alternatives for decision {decision.decision_id}: {json_err}")
                             row_data["alternatives"] = "[]" # Save as empty list string on error
                    elif not isinstance(row_data.get("alternatives"), str):
                        row_data["alternatives"] = "[]" # Default if not list or string

                    writer.writerow(row_data)
            logger.debug(f"Saved decisions log to {filename}")
        except Exception as e:
            logger.error(f"Error saving decisions to {filename}: {str(e)}", exc_info=True)

    # --- ID generation, add_event, add_decision remain the same ---
    def generate_id(self, prefix):
        """Generate a unique ID with the given prefix"""
        # Using microseconds and ensuring the random part is hex for consistency
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") # Added microseconds
        random_suffix = uuid.uuid4().hex[:8] # Use hex characters
        return f"{prefix}{timestamp}-{random_suffix}" # Added separator

    def generate_order_id(self):
        """Generate a unique order ID"""
        return self.generate_id("ORD-")

    def generate_event_id(self):
        """Generate a unique event ID"""
        return self.generate_id("EVT-")

    def generate_decision_id(self):
        """Generate a unique decision ID"""
        return self.generate_id("DEC-")

    def add_event(self, event_type, description, entity_id=None, entity_type=None, severity=1, ai_analysis=None):
        """Add a new event to the events log"""
        with self.event_lock:
            event_id = self.generate_event_id()
            event = Event(event_id, event_type, description,
                         entity_id=entity_id, entity_type=entity_type,
                         severity=severity, ai_analysis=ai_analysis)
            self.events_log.append(event)
            logger.info(f"Event added: {event_id} - Type: {event_type}, Severity: {severity}, Entity: {entity_type}/{entity_id}, Desc: {description[:50]}...")
            return event

    def add_decision(self, agent_type, decision_type, entity_id, reasoning=None,
                    ai_involved=False, confidence_score=0.0, alternatives=None):
        """Add a new decision to the agent decisions log"""
        with self.decision_lock:
            decision_id = self.generate_decision_id()
            decision = AgentDecision(decision_id, agent_type, decision_type, entity_id,
                                    reasoning=reasoning, ai_involved=ai_involved,
                                    confidence_score=confidence_score, alternatives=alternatives)
            self.agent_decisions.append(decision)
            logger.info(f"Decision logged: {decision_id} - Agent: {agent_type}, Type: {decision_type}, Entity: {entity_id}, AI: {ai_involved} ({confidence_score:.2f})")
            return decision

    # --- Helper to check for pending supplier orders ---
    def has_pending_supplier_order(self, part_id):
        """Checks if there's a pending or approved supplier order for a part."""
        with self.order_lock:
            for order in self.order_history:
                if (order.part_id == part_id and
                    order.order_type == TYPE_WAREHOUSE_TO_SUPPLIER and
                    order.status in [STATUS_PENDING, STATUS_APPROVED, STATUS_IN_TRANSIT]):
                    return True
        return False

# --- AmbientAgent Base Class (Keep as is) ---
class AmbientAgent:
    """Base class for all ambient agents in the system"""
    def __init__(self, inventory_manager: InventoryManager, agent_type: str):
        self.inventory_manager = inventory_manager
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"AmbientAgent.{agent_type}")
        # Use the globally configured key status
        self.ai_enabled = GEMINI_API_KEY is not None and model is not None
        self.last_check_time = datetime.datetime.now()
        # Use a shared executor or manage individually? Shared might be better.
        # Let's assume it's passed in or created per agent for now.
        self.executor = ThreadPoolExecutor(max_workers=2) # Consider optimal worker count
        self.running = False
        self.thread = None
        self._stop_event = threading.Event() # For graceful stopping

    def start(self):
        """Start the agent's background thread"""
        if self.running:
            self.logger.warning(f"{self.agent_type} agent is already running")
            return

        self.running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._run_agent_loop, name=f"{self.agent_type}AgentThread")
        self.thread.daemon = True # Allow program to exit even if threads are running
        self.thread.start()
        self.logger.info(f"{self.agent_type} agent started")

    def stop(self):
        """Stop the agent's background thread gracefully"""
        if not self.running:
            self.logger.warning(f"{self.agent_type} agent is not running")
            return

        self.running = False
        self._stop_event.set() # Signal the loop to stop
        self.executor.shutdown(wait=True) # Wait for pending tasks in executor

        if self.thread and self.thread.is_alive():
             # Wait for the thread to finish its current cycle
             self.thread.join(timeout=10.0) # Increased timeout
             if self.thread.is_alive():
                 self.logger.warning(f"{self.agent_type} agent thread did not stop gracefully after 10s.")
             else:
                 self.logger.info(f"{self.agent_type} agent stopped.")
        elif not self.thread:
             self.logger.info(f"{self.agent_type} agent stopped (thread was not running).")


    def _run_agent_loop(self):
        """The actual loop run in the thread, checking the stop event."""
        self.logger.info(f"{self.agent_type} agent loop starting.")
        while not self._stop_event.is_set():
            try:
                self._run_agent_cycle() # Execute one cycle of agent logic
            except Exception as e:
                # Log error and continue, maybe with backoff
                self.logger.error(f"Error in {self.agent_type} agent cycle: {e}", exc_info=True)
                # Wait a bit longer after an error before retrying
                self._stop_event.wait(30) # Use wait instead of sleep for responsiveness
            else:
                 # Wait for the configured interval or until stop is signaled
                 # Make interval configurable?
                 self._stop_event.wait(self.get_agent_interval()) # Use wait instead of sleep

        self.logger.info(f"{self.agent_type} agent loop finished.")

    def get_agent_interval(self) -> float:
        """Returns the sleep interval for the agent. Override in subclasses if needed."""
        return 15.0 # Default interval (e.g., 15 seconds)

    def _run_agent_cycle(self):
        """Contains the logic for a single cycle of the agent. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run_agent_cycle method")

    def ask_ai(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, any]:
        """Use Gemini API to get AI recommendations"""
        default_response = {"response": "AI processing failed or disabled.", "confidence": 0.0, "error": None}
        if not self.ai_enabled:
            self.logger.warning("AI features are disabled (no valid API key or model). Cannot process AI request.")
            default_response["response"] = "AI features disabled."
            return default_response

        if not model: # Double check model initialization
            self.logger.error("Gemini model not initialized. Cannot process AI request.")
            default_response["error"] = "Gemini model not initialized."
            return default_response

        self.logger.debug(f"Sending prompt to AI:\nSystem: {system_prompt}\nPrompt: {prompt[:200]}...") # Log truncated prompt

        try:
            # Use recommended settings for gemini-1.5-flash or similar
            generation_config = {
                "temperature": 0.7, # Slightly creative but still grounded
                "top_p": 0.95,
                "top_k": 40,
                 "max_output_tokens": 1024, # Limit response size
            }
            # Define safety settings - adjust thresholds as needed
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            content_to_send = [prompt]
            if system_prompt:
                 # Gemini API often takes system instructions within the user prompt or uses specific roles
                 # For simplicity here, prepend system prompt text if provided. Check API docs for best practices.
                 # content_to_send = [system_prompt, prompt] # Alternative structure if API supports it directly
                 content_to_send = [f"System Instructions: {system_prompt}\n\nUser Request: {prompt}"]


            response = model.generate_content(
                content_to_send,
                generation_config=generation_config,
                safety_settings=safety_settings,
                # stream=False # Ensure we get the full response at once
            )

            # Handle potential blocks or empty responses
            if not response.candidates:
                 # Check for safety feedback if available
                 block_reason = "Unknown"
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason.name
                 self.logger.warning(f"AI response blocked or empty. Reason: {block_reason}")
                 default_response["response"] = f"AI response blocked or empty (Reason: {block_reason})."
                 default_response["error"] = f"Blocked: {block_reason}"
                 return default_response


            # Accessing response text correctly for Gemini API
            # Assuming the first candidate and first part contain the text
            if response.candidates[0].content and response.candidates[0].content.parts:
                 text_response = response.candidates[0].content.parts[0].text
                 self.logger.debug(f"AI Raw Response: {text_response[:200]}...") # Log truncated response

                 # Simple confidence estimation (as before, can be refined)
                 confidence = 0.7 # Default medium-high confidence
                 confidence_words = ["confident", "certain", "definitely", "absolutely", "clearly", "recommend", "optimal"]
                 uncertainty_words = ["maybe", "perhaps", "possibly", "uncertain", "might", "consider", "suggest"]

                 response_lower = text_response.lower()
                 for word in confidence_words:
                     if word in response_lower:
                         confidence += 0.05
                 for word in uncertainty_words:
                     if word in response_lower:
                         confidence -= 0.1

                 # Clamp confidence
                 confidence = max(0.1, min(0.95, confidence))

                 return {"response": text_response.strip(), "confidence": confidence, "error": None}
            else:
                 self.logger.warning("AI response structure unexpected: No text part found.")
                 default_response["response"] = "AI response received but content is missing."
                 default_response["error"] = "Missing content parts"
                 return default_response

        except Exception as e:
            self.logger.error(f"Error during AI request: {str(e)}", exc_info=True)
            default_response["response"] = f"Error during AI request: {str(e)}"
            default_response["error"] = str(e)
            return default_response

    def log_decision(self, decision_type: str, entity_id: str, reasoning: Optional[str] = None,
                   ai_involved: bool = False, confidence_score: float = 0.0, alternatives: Optional[List] = None):
        """Log a decision made by this agent using InventoryManager"""
        try:
            return self.inventory_manager.add_decision(
                self.agent_type, decision_type, entity_id, reasoning,
                ai_involved, confidence_score, alternatives
            )
        except Exception as e:
             self.logger.error(f"Failed to log decision ({decision_type} for {entity_id}): {e}", exc_info=True)
             return None # Indicate failure

# --- ShowroomAmbientAgent (Keep as is, or adjust check_inventory logic if needed) ---
class ShowroomAmbientAgent(AmbientAgent):
    def __init__(self, inventory_manager: InventoryManager):
        super().__init__(inventory_manager, "ShowroomAgent")
        # Showroom might react faster
        self._interval = 10.0 # Check every 10 seconds

    def get_agent_interval(self) -> float:
        return self._interval

    def _run_agent_cycle(self):
        """Main logic for a single cycle of the showroom agent"""
        self.logger.debug(f"{self.agent_type} - Running cycle")
        # Using thread locks from inventory_manager for safety
        with self.inventory_manager.inventory_lock, self.inventory_manager.order_lock:
             orders_created = self.check_inventory_and_order()
             self.process_delivered_orders() # Renamed from process_approved_orders

        # Save state periodically (maybe less frequently than cycle time?)
        # Consider saving only if changes were made, or on a separate timer
        # self.inventory_manager.save_inventory() # Saving moved to main loop or system level

        if orders_created:
             self.logger.info(f"{self.agent_type} created {len(orders_created)} new orders in this cycle.")
        self.logger.debug(f"{self.agent_type} - Cycle finished")


    def check_inventory_and_order(self) -> List[Order]:
        """Check showroom inventory and create orders to Warehouse if below threshold."""
        self.logger.info("Checking showroom inventory for restocking needs...")
        orders_to_create = []
        created_orders = []

        current_time = datetime.datetime.now()
        # Throttle full AI checks? Maybe only check AI every few minutes.
        # run_ai_check = (current_time - self.last_check_time).total_seconds() > 300 # Check AI every 5 mins

        for part_id, item in self.inventory_manager.showroom_inventory.items():
            if item.quantity < item.min_threshold:
                self.logger.warning(f"Showroom low stock: {item.part_name} ({item.quantity}/{item.min_threshold})")

                # Check if an order for this part is already pending/in_transit from Warehouse
                order_exists = False
                for order in self.inventory_manager.order_history:
                    if (order.part_id == part_id and
                        order.order_type == TYPE_SHOWROOM_TO_WAREHOUSE and
                        order.status in [STATUS_PENDING, STATUS_APPROVED, STATUS_IN_TRANSIT]):
                        order_exists = True
                        self.logger.info(f"Existing order {order.order_id} found for {item.part_name}. Skipping new order.")
                        break
                if order_exists:
                    continue

                # Determine quantity needed
                target_level = int(item.max_capacity * 0.8) # Target 80%
                quantity_to_order = max(0, target_level - item.quantity) # Ensure non-negative

                if quantity_to_order == 0:
                    self.logger.info(f"Item {item.part_name} is below threshold but target level reached. No order needed.")
                    continue


                ai_recommendation = None
                confidence_score = 0.0
                reasoning = f"Stock {item.quantity} < threshold {item.min_threshold}. Ordering to target {target_level}."

                # Optional: Use AI to refine order quantity (potentially throttled)
                if self.ai_enabled: # and run_ai_check: # Add throttling if needed
                    self.logger.debug(f"Requesting AI suggestion for order quantity for {item.part_name}")
                    prompt = f"""
                    Analyze the situation for showroom part '{item.part_name}' (ID: {part_id}).
                    Current Quantity: {item.quantity}
                    Min Threshold: {item.min_threshold}
                    Max Capacity: {item.max_capacity}
                    Default target quantity to order: {quantity_to_order} (to reach {target_level}).
                    Warehouse stock for this part: {self.inventory_manager.warehouse_inventory.get(part_id, InventoryItem('','','',0)).quantity} units.

                    Considering the warehouse availability and the goal to avoid overstocking the showroom while meeting the minimum, is {quantity_to_order} a good quantity to request from the warehouse?
                    If not, suggest a better quantity (between 1 and {item.max_capacity - item.quantity}) and provide a brief reason.
                    Output only the recommended quantity number and a short reasoning like: 'QUANTITY: <number>, REASON: <text>'. Example: 'QUANTITY: 50, REASON: Balances threshold and warehouse stock.'
                    """
                    system_prompt = "You are an inventory optimization assistant for a car parts showroom. Focus on practical order quantities based on thresholds and warehouse availability."

                    ai_result = self.ask_ai(prompt, system_prompt)
                    ai_recommendation = ai_result["response"]
                    confidence_score = ai_result["confidence"]

                    if ai_result["error"]:
                         self.logger.error(f"AI suggestion failed for {item.part_name}: {ai_result['error']}")
                    else:
                        # Try to parse the AI response for QUANTITY: <number>
                        import re
                        match = re.search(r"QUANTITY:\s*(\d+)", ai_recommendation, re.IGNORECASE)
                        if match:
                            try:
                                ai_quantity = int(match.group(1))
                                # Validate AI quantity
                                if 0 < ai_quantity <= (item.max_capacity - item.quantity):
                                    if ai_quantity != quantity_to_order:
                                         self.logger.info(f"AI suggested adjusting order for {item.part_name} from {quantity_to_order} to {ai_quantity}. Confidence: {confidence_score:.2f}")
                                         quantity_to_order = ai_quantity
                                         reasoning += f" AI Adjusted Qty: {ai_quantity}. AI Reason: {ai_recommendation}"
                                    else:
                                         reasoning += " AI Confirmed default quantity."
                                else:
                                    self.logger.warning(f"AI suggested invalid quantity {ai_quantity} for {item.part_name}. Using default {quantity_to_order}.")
                                    reasoning += f" AI suggested invalid quantity ({ai_quantity}). Using default."
                            except ValueError:
                                self.logger.warning(f"Could not parse AI quantity from response for {item.part_name}. Using default. Response: {ai_recommendation}")
                                reasoning += " Could not parse AI quantity. Using default."
                        else:
                             self.logger.info(f"AI response for {item.part_name} didn't provide specific quantity. Using default. Response: {ai_recommendation}")
                             reasoning += " AI did not provide specific quantity. Using default."


                # Check if warehouse actually has the stock (redundant check, WarehouseAgent also checks)
                warehouse_item = self.inventory_manager.warehouse_inventory.get(part_id)
                if not warehouse_item or warehouse_item.quantity < quantity_to_order:
                     self.logger.warning(f"Warehouse has insufficient stock ({warehouse_item.quantity if warehouse_item else 0}) for {item.part_name} needed ({quantity_to_order}). Cannot create order now.")
                     # Log event?
                     self.inventory_manager.add_event(
                         "order_blocked",
                         f"Showroom order for {quantity_to_order} of {item.part_name} blocked. Warehouse stock: {warehouse_item.quantity if warehouse_item else 0}.",
                         entity_id=part_id,
                         entity_type="inventory_item",
                         severity=3 # Moderately severe - potential stockout
                     )
                     continue # Skip creating this order


                # Add order details to list for batch creation
                order_details = {
                    "part_id": part_id,
                    "quantity": quantity_to_order,
                    "ai_recommendation": ai_recommendation,
                    "confidence_score": confidence_score,
                    "reasoning": reasoning
                }
                orders_to_create.append(order_details)

        # Batch create orders outside the item loop
        for details in orders_to_create:
            order_id = self.inventory_manager.generate_order_id()
             # Assume 1 day delivery from Warehouse to Showroom for expected date
            expected_del_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            order = Order(
                order_id=order_id,
                part_id=details["part_id"],
                quantity=details["quantity"],
                order_type=TYPE_SHOWROOM_TO_WAREHOUSE,
                status=STATUS_PENDING, # Warehouse needs to approve/process
                order_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                expected_delivery=expected_del_date,
                priority=3, # Medium priority for standard restock
                ai_recommendation=details["ai_recommendation"],
                confidence_score=details["confidence_score"]
            )
            self.inventory_manager.order_history.append(order)
            created_orders.append(order)
            self.logger.info(f"Created pending Showroom->Warehouse order: {order}")

            # Log the decision to create the order
            self.log_decision(
                decision_type="create_order",
                entity_id=order.order_id,
                reasoning=details["reasoning"],
                ai_involved=bool(details["ai_recommendation"]),
                confidence_score=details["confidence_score"]
            )
            # Log the low inventory event that triggered it
            self.inventory_manager.add_event(
                "low_inventory_trigger",
                f"Showroom low stock triggered order {order.order_id} for {order.quantity} of {order.part_id}.",
                entity_id=order.part_id,
                entity_type="inventory_item",
                severity=2
            )

        # Update last check time if AI check was run
        # if run_ai_check: self.last_check_time = current_time
        self.last_check_time = current_time # Update time after every check run

        return created_orders


    def process_delivered_orders(self):
        """Checks for orders marked IN_TRANSIT from Warehouse and marks them DELIVERED, updating stock."""
        self.logger.info("Processing orders in transit to showroom...")
        updated_count = 0

        # Process orders that should be delivered based on expected date (simple simulation)
        today = datetime.datetime.now().date()

        for order in self.inventory_manager.order_history:
             # Only process orders coming TO the showroom that are IN_TRANSIT
            if (order.order_type == TYPE_SHOWROOM_TO_WAREHOUSE and
                order.status == STATUS_IN_TRANSIT):

                # Simulate delivery: If expected delivery date is today or past, mark as delivered
                delivered = False
                if order.expected_delivery:
                     try:
                         expected_date = datetime.datetime.strptime(order.expected_delivery, "%Y-%m-%d").date()
                         if today >= expected_date:
                             delivered = True
                     except ValueError:
                          self.logger.warning(f"Order {order.order_id} has invalid expected_delivery format: {order.expected_delivery}. Cannot determine delivery.")
                         # Optionally, deliver after a fixed delay from order date if expected_delivery is bad
                else:
                     # If no expected date, maybe deliver after 1 day? (Needs refinement)
                     order_date_dt = datetime.datetime.strptime(order.order_date, "%Y-%m-%d").date()
                     if (today - order_date_dt).days >= 1:
                           delivered = True
                           self.logger.warning(f"Order {order.order_id} has no expected delivery date. Marking delivered after 1 day.")


                if delivered:
                    showroom_item = self.inventory_manager.showroom_inventory.get(order.part_id)
                    if showroom_item:
                        # Increase showroom stock
                        showroom_item.quantity += order.quantity
                        showroom_item.last_updated = datetime.datetime.now()

                        # Update order status
                        order.update_status(STATUS_DELIVERED, "Order received at showroom, stock updated.")
                        self.logger.info(f"Order delivered and stock updated: {order}")
                        updated_count += 1

                        # Log the event
                        self.inventory_manager.add_event(
                            "order_delivered_showroom",
                            f"Order {order.order_id} ({order.quantity}x {order.part_id}) delivered to showroom.",
                            entity_id=order.order_id,
                            entity_type="order",
                            severity=1 # Informational
                        )
                        # Log decision? (Implicit decision based on time/status)
                        self.log_decision(
                            decision_type="mark_delivered",
                            entity_id=order.order_id,
                            reasoning=f"Order marked delivered based on expected date {order.expected_delivery} or passage of time.",
                            ai_involved=False
                        )
                    else:
                        # This shouldn't happen if order was created correctly, but handle defensively
                        order.update_status(STATUS_CANCELLED, f"Delivery failed: Part ID {order.part_id} not found in showroom inventory.")
                        self.logger.error(f"Cannot deliver order {order.order_id}: Part ID {order.part_id} missing in showroom!")
                        # Log critical event
                        self.inventory_manager.add_event(
                            "delivery_error",
                            f"Showroom delivery failed for order {order.order_id}. Part ID {order.part_id} not found.",
                            entity_id=order.order_id,
                            entity_type="order",
                            severity=4 # High severity - data inconsistency
                        )
        if updated_count > 0:
            self.logger.info(f"Processed {updated_count} delivered orders to showroom.")

    # Note: detect_sales_patterns was in the original but might be better suited
    # for a dedicated analytics module or integrated into the AI prompt. Removed for brevity here.

# --- WarehouseAgent (Keep as is, but ensure it handles order approval/fulfillment) ---
class WarehouseAgent(AmbientAgent):
    def __init__(self, inventory_manager: InventoryManager):
        super().__init__(inventory_manager, "WarehouseAgent")
        self._interval = 12.0 # Check slightly less often than showroom

    def get_agent_interval(self) -> float:
        return self._interval

    def _run_agent_cycle(self):
        """Main logic for a single cycle of the warehouse agent"""
        self.logger.debug(f"{self.agent_type} - Running cycle")
        with self.inventory_manager.inventory_lock, self.inventory_manager.order_lock:
            # 1. Review orders coming FROM Showroom (approve/reject based on stock)
            self.review_and_process_showroom_orders()

            # 2. Check own inventory levels and order FROM Supplier if needed
            supplier_orders_created = self.check_inventory_and_order_supplier()

            # 3. Process orders arriving FROM Supplier (mark delivered, update stock)
            self.process_delivered_supplier_orders()

        # self.inventory_manager.save_inventory() # Saving moved

        if supplier_orders_created:
             self.logger.info(f"{self.agent_type} created {len(supplier_orders_created)} new supplier orders in this cycle.")
        self.logger.debug(f"{self.agent_type} - Cycle finished")


    def review_and_process_showroom_orders(self):
        """Review PENDING orders from Showroom. Approve if stock available, then mark IN_TRANSIT."""
        self.logger.info("Reviewing pending orders from showroom...")
        processed_count = 0
        approved_count = 0
        cancelled_count = 0

        for order in self.inventory_manager.order_history:
            # Look for orders TO warehouse (i.e., from showroom) that are PENDING
            if (order.order_type == TYPE_SHOWROOM_TO_WAREHOUSE and
                order.status == STATUS_PENDING):

                processed_count += 1
                warehouse_item = self.inventory_manager.warehouse_inventory.get(order.part_id)
                reasoning = ""
                decision_type = ""

                if warehouse_item and warehouse_item.quantity >= order.quantity:
                    # Approve and fulfill (decrease warehouse stock, mark in_transit)
                    warehouse_item.quantity -= order.quantity
                    warehouse_item.last_updated = datetime.datetime.now()
                    order.update_status(STATUS_IN_TRANSIT, f"Warehouse approved. Stock decreased by {order.quantity}. In transit to showroom.")
                    self.logger.info(f"Approved and processing order: {order}. Warehouse stock for {order.part_id} now {warehouse_item.quantity}")
                    approved_count += 1
                    reasoning = f"Approved: Warehouse stock ({warehouse_item.quantity + order.quantity}) >= requested ({order.quantity}). Stock updated."
                    decision_type = "approve_order"
                    # Log event for fulfillment start
                    self.inventory_manager.add_event(
                        "order_fulfillment_start",
                        f"Warehouse started fulfilling order {order.order_id} ({order.quantity}x {order.part_id}) for showroom.",
                        entity_id=order.order_id,
                        entity_type="order",
                        severity=1
                    )
                else:
                    # Cancel due to insufficient stock
                    current_stock = warehouse_item.quantity if warehouse_item else 0
                    order.update_status(STATUS_CANCELLED, f"Warehouse cancelled: Insufficient stock ({current_stock}) for requested quantity ({order.quantity}).")
                    self.logger.warning(f"Cancelled order due to insufficient stock: {order}. Warehouse has {current_stock}")
                    cancelled_count += 1
                    reasoning = f"Cancelled: Warehouse stock ({current_stock}) < requested ({order.quantity})."
                    decision_type = "cancel_order"
                     # Log event for cancellation
                    self.inventory_manager.add_event(
                        "order_cancelled_stock",
                        f"Warehouse cancelled order {order.order_id} ({order.quantity}x {order.part_id}) due to insufficient stock ({current_stock}).",
                        entity_id=order.order_id,
                        entity_type="order",
                        severity=3 # Moderate issue
                    )

                # Log the decision
                self.log_decision(
                    decision_type=decision_type,
                    entity_id=order.order_id,
                    reasoning=reasoning,
                    ai_involved=False # This part is rule-based
                )

        if processed_count > 0:
             self.logger.info(f"Reviewed {processed_count} showroom orders: {approved_count} approved, {cancelled_count} cancelled.")


    def check_inventory_and_order_supplier(self) -> List[Order]:
        """Check warehouse inventory and create orders to Supplier if below threshold."""
        self.logger.info("Checking warehouse inventory for supplier restocking needs...")
        orders_to_create = []
        created_orders = []
        current_time = datetime.datetime.now()

        for part_id, item in self.inventory_manager.warehouse_inventory.items():
            # Check if below threshold OR if predicted shortage requires ordering (handled by AssemblyLineAgent now)
            # This agent focuses on threshold-based ordering for the warehouse itself.
            if item.quantity < item.min_threshold:
                self.logger.warning(f"Warehouse low stock: {item.part_name} ({item.quantity}/{item.min_threshold})")

                # Check if a supplier order is already pending/approved/in_transit
                if self.inventory_manager.has_pending_supplier_order(part_id):
                     self.logger.info(f"Existing supplier order found for {item.part_name}. Skipping new order.")
                     continue

                # Determine quantity needed to reach, e.g., 90% capacity
                target_level = int(item.max_capacity * 0.9)
                quantity_to_order = max(0, target_level - item.quantity)

                if quantity_to_order == 0:
                     self.logger.info(f"Warehouse item {item.part_name} below threshold but target level reached. No supplier order needed.")
                     continue

                # Find the best supplier (based on availability, lead time, price, reliability - simplified here)
                supplier_part_ids = self.inventory_manager.get_supplier_part_ids_by_name(item.part_name)
                best_supplier_part_id = None
                supplier_lead_time = 7 # Default lead time
                supplier_info_for_order = None

                if not supplier_part_ids:
                     self.logger.error(f"Cannot order {item.part_name}: No supplier found offering this part name.")
                     self.inventory_manager.add_event("order_failed_no_supplier", f"Warehouse order for {item.part_name} failed: No supplier found.", entity_id=part_id, entity_type="inventory_item", severity=4)
                     continue

                # Simple selection: Choose first available supplier who has enough stock
                # TODO: Implement better supplier selection logic (cost, lead time, reliability)
                for sup_part_id in supplier_part_ids:
                    supplier_item = self.inventory_manager.supplier_inventory.get(sup_part_id)
                    if supplier_item and supplier_item.available_quantity >= quantity_to_order:
                         best_supplier_part_id = sup_part_id
                         supplier_lead_time = supplier_item.lead_time_days
                         supplier_info_for_order = supplier_item # Store for potential AI use
                         self.logger.info(f"Selected supplier part {sup_part_id} for {item.part_name}. Available: {supplier_item.available_quantity}, Lead time: {supplier_lead_time} days.")
                         break # Found a suitable supplier

                if not best_supplier_part_id:
                     self.logger.warning(f"Cannot order {quantity_to_order} of {item.part_name}: No supplier has sufficient available quantity (checked {len(supplier_part_ids)} potential suppliers).")
                     # Find max available quantity among suppliers
                     max_avail = 0
                     for sup_part_id in supplier_part_ids:
                          supplier_item = self.inventory_manager.supplier_inventory.get(sup_part_id)
                          if supplier_item: max_avail = max(max_avail, supplier_item.available_quantity)
                     self.inventory_manager.add_event("order_failed_supplier_stock", f"Warehouse order for {item.part_name} ({quantity_to_order}) failed: Max available from suppliers is {max_avail}.", entity_id=part_id, entity_type="inventory_item", severity=3)
                     continue # Skip order creation


                # --- AI Quantity Check (Optional, similar to Showroom) ---
                ai_recommendation = None
                confidence_score = 0.0
                reasoning = f"Warehouse stock {item.quantity} < threshold {item.min_threshold}. Ordering {quantity_to_order} to target {target_level} from supplier part {best_supplier_part_id}."

                if self.ai_enabled and supplier_info_for_order:
                    prompt = f"""
                    Analyze situation for warehouse part '{item.part_name}' (ID: {part_id}).
                    Current Qty: {item.quantity}, Min Threshold: {item.min_threshold}, Max Capacity: {item.max_capacity}.
                    Default order qty: {quantity_to_order} (to reach {target_level}).
                    Selected Supplier Part ID: {best_supplier_part_id}
                    Supplier Available Qty: {supplier_info_for_order.available_quantity}
                    Supplier Lead Time: {supplier_info_for_order.lead_time_days} days
                    Supplier Price: {supplier_info_for_order.price}
                    Supplier Reliability: {supplier_info_for_order.vendor_reliability:.2f}

                    Considering lead time, supplier availability, and warehouse capacity, is {quantity_to_order} the optimal quantity to order from the supplier?
                    Suggest adjustments if needed (e.g., order more to buffer for lead time, order less if capacity is tight).
                    Output only the recommended quantity number and a short reasoning like: 'QUANTITY: <number>, REASON: <text>'. Example: 'QUANTITY: 150, REASON: Accounts for lead time demand.'
                    """
                    system_prompt = "You are an inventory optimization assistant for a car parts warehouse ordering from suppliers. Consider lead times, costs (implicitly), and reliability."
                    ai_result = self.ask_ai(prompt, system_prompt)
                    ai_recommendation = ai_result["response"]
                    confidence_score = ai_result["confidence"]

                    if not ai_result["error"]:
                         # Parse AI response (similar logic as showroom)
                         import re
                         match = re.search(r"QUANTITY:\s*(\d+)", ai_recommendation, re.IGNORECASE)
                         if match:
                            try:
                                ai_quantity = int(match.group(1))
                                # Validate AI quantity (must be <= supplier available, <= space in warehouse)
                                space_available = item.max_capacity - item.quantity
                                order_limit = min(supplier_info_for_order.available_quantity, space_available)

                                if 0 < ai_quantity <= order_limit:
                                    if ai_quantity != quantity_to_order:
                                        self.logger.info(f"AI suggested adjusting supplier order for {item.part_name} from {quantity_to_order} to {ai_quantity}. Limit: {order_limit}. Confidence: {confidence_score:.2f}")
                                        quantity_to_order = ai_quantity
                                        reasoning += f" AI Adjusted Qty: {ai_quantity}. AI Reason: {ai_recommendation}"
                                    else:
                                         reasoning += " AI Confirmed default quantity."
                                else:
                                    self.logger.warning(f"AI suggested invalid quantity {ai_quantity} for {item.part_name} (Limit: {order_limit}). Using default {quantity_to_order}.")
                                    reasoning += f" AI suggested invalid quantity ({ai_quantity}). Using default."
                            except ValueError:
                                 reasoning += " Could not parse AI quantity. Using default."
                         else:
                             reasoning += " AI did not provide specific quantity. Using default."
                    else:
                         reasoning += f" AI suggestion failed: {ai_result['error']}"
                # --- End AI Check ---


                 # Add order details for batch creation
                order_details = {
                    "part_id": part_id, # Warehouse part ID
                    "supplier_part_id": best_supplier_part_id, # Needed for supplier processing? No, Order obj uses warehouse part_id
                    "quantity": quantity_to_order,
                    "lead_time": supplier_lead_time,
                    "ai_recommendation": ai_recommendation,
                    "confidence_score": confidence_score,
                    "reasoning": reasoning
                }
                orders_to_create.append(order_details)


        # Batch create supplier orders
        for details in orders_to_create:
            order_id = self.inventory_manager.generate_order_id()
            expected_del_date = (datetime.datetime.now() + datetime.timedelta(days=details["lead_time"])).strftime("%Y-%m-%d")

            order = Order(
                order_id=order_id,
                part_id=details["part_id"], # Use the warehouse part ID
                quantity=details["quantity"],
                order_type=TYPE_WAREHOUSE_TO_SUPPLIER,
                status=STATUS_PENDING, # Supplier needs to approve
                order_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                expected_delivery=expected_del_date,
                priority=2, # Lower priority than showroom restock? Or higher because it enables showroom? Let's say 2.
                ai_recommendation=details["ai_recommendation"],
                confidence_score=details["confidence_score"]
            )
            self.inventory_manager.order_history.append(order)
            created_orders.append(order)
            self.logger.info(f"Created pending Warehouse->Supplier order: {order}")

            # Log decision
            self.log_decision(
                decision_type="create_supplier_order",
                entity_id=order.order_id,
                reasoning=details["reasoning"],
                ai_involved=bool(details["ai_recommendation"]),
                confidence_score=details["confidence_score"]
            )
             # Log event
            self.inventory_manager.add_event(
                "low_inventory_trigger_wh",
                f"Warehouse low stock triggered supplier order {order.order_id} for {order.quantity} of {order.part_id}.",
                entity_id=order.part_id,
                entity_type="inventory_item",
                severity=2
            )

        self.last_check_time = current_time
        return created_orders


    def process_delivered_supplier_orders(self):
        """Checks orders IN_TRANSIT from Supplier, marks DELIVERED, updates warehouse stock."""
        self.logger.info("Processing orders in transit from supplier...")
        updated_count = 0
        today = datetime.datetime.now().date()

        for order in self.inventory_manager.order_history:
            # Only process orders coming TO the warehouse (from supplier) that are IN_TRANSIT
            if (order.order_type == TYPE_WAREHOUSE_TO_SUPPLIER and
                order.status == STATUS_IN_TRANSIT):

                 # Simulate delivery based on expected date
                delivered = False
                if order.expected_delivery:
                    try:
                        expected_date = datetime.datetime.strptime(order.expected_delivery, "%Y-%m-%d").date()
                        if today >= expected_date:
                            delivered = True
                    except ValueError:
                         self.logger.warning(f"Supplier order {order.order_id} has invalid expected_delivery format: {order.expected_delivery}.")
                         # Decide fallback: Maybe deliver after lead time from order date?
                         # supplier_item = self.inventory_manager.supplier_inventory.get(...) # Need supplier part ID mapping... complex
                         # For now, don't auto-deliver if format is bad. Requires manual intervention or better logic.

                else:
                     # No expected delivery date - should not happen if created correctly
                     self.logger.warning(f"Supplier order {order.order_id} is IN_TRANSIT but has no expected delivery date. Cannot auto-deliver.")


                if delivered:
                    warehouse_item = self.inventory_manager.warehouse_inventory.get(order.part_id)
                    if warehouse_item:
                        # Increase warehouse stock
                        old_qty = warehouse_item.quantity
                        warehouse_item.quantity += order.quantity
                        warehouse_item.last_updated = datetime.datetime.now()

                        # Update order status
                        order.update_status(STATUS_DELIVERED, f"Order received at warehouse from supplier. Stock updated from {old_qty} to {warehouse_item.quantity}.")
                        self.logger.info(f"Supplier order delivered and stock updated: {order}")
                        updated_count += 1

                        # Log event
                        self.inventory_manager.add_event(
                            "order_delivered_warehouse",
                            f"Supplier order {order.order_id} ({order.quantity}x {order.part_id}) delivered to warehouse.",
                            entity_id=order.order_id,
                            entity_type="order",
                            severity=1
                        )
                        # Log decision
                        self.log_decision(
                            decision_type="mark_delivered_supplier",
                            entity_id=order.order_id,
                            reasoning=f"Order marked delivered based on expected date {order.expected_delivery}.",
                            ai_involved=False
                        )
                    else:
                        # Critical error: Warehouse part ID doesn't exist
                        order.update_status(STATUS_CANCELLED, f"Delivery failed: Part ID {order.part_id} not found in warehouse inventory.")
                        self.logger.error(f"Cannot process delivered supplier order {order.order_id}: Part ID {order.part_id} missing in warehouse!")
                        self.inventory_manager.add_event(
                            "delivery_error_supplier",
                            f"Warehouse delivery failed for supplier order {order.order_id}. Part ID {order.part_id} not found.",
                            entity_id=order.order_id,
                            entity_type="order",
                            severity=5 # Critical data inconsistency
                        )

        if updated_count > 0:
            self.logger.info(f"Processed {updated_count} delivered orders from suppliers.")

    # create_supplier_order method seems redundant now as logic is in check_inventory_and_order_supplier
    # process_approved_orders seems redundant / split into review_showroom_orders and process_delivered_supplier_orders


# --- SupplierAgent (Keep as is, mainly approves/rejects warehouse orders) ---
class SupplierAgent(AmbientAgent):
    def __init__(self, inventory_manager: InventoryManager):
        super().__init__(inventory_manager, "SupplierAgent")
        # Supplier might operate on a daily schedule or less frequently
        self._interval = 60.0 * 5 # Check every 5 minutes for demo; could be hours

    def get_agent_interval(self) -> float:
        return self._interval

    def _run_agent_cycle(self):
        """Main logic for a single cycle of the supplier agent"""
        self.logger.debug(f"{self.agent_type} - Running cycle")
        with self.inventory_manager.inventory_lock, self.inventory_manager.order_lock:
             self.review_and_process_warehouse_orders()
             # Simulate supplier stock changes (e.g., new production, other sales) - Optional advanced feature
             # self.simulate_supplier_stock_changes()

        # self.inventory_manager.save_inventory() # Saving moved
        self.logger.debug(f"{self.agent_type} - Cycle finished")


    def review_and_process_warehouse_orders(self):
        """Review PENDING orders from Warehouse. Approve if stock available, mark IN_TRANSIT."""
        self.logger.info("Reviewing pending orders from warehouse...")
        processed_count = 0
        approved_count = 0
        cancelled_count = 0

        for order in self.inventory_manager.order_history:
             # Look for orders TO supplier (i.e., from warehouse) that are PENDING
            if (order.order_type == TYPE_WAREHOUSE_TO_SUPPLIER and
                order.status == STATUS_PENDING):

                processed_count += 1
                reasoning = ""
                decision_type = ""

                # Find the corresponding supplier part(s) based on the warehouse part name
                warehouse_part = self.inventory_manager.warehouse_inventory.get(order.part_id)
                if not warehouse_part:
                    # This indicates an issue - order created for non-existent warehouse part
                    order.update_status(STATUS_CANCELLED, f"Supplier cancelled: Warehouse Part ID {order.part_id} does not exist.")
                    self.logger.error(f"Cannot process order {order.order_id}: Warehouse Part ID {order.part_id} not found in inventory.")
                    cancelled_count += 1
                    reasoning = f"Cancelled: Warehouse part ID {order.part_id} unknown."
                    decision_type = "cancel_order_bad_partid"
                    self.inventory_manager.add_event("order_error_bad_partid", f"Supplier received order {order.order_id} for unknown warehouse part {order.part_id}.", entity_id=order.order_id, severity=4)
                    continue # Skip to next order

                warehouse_part_name = warehouse_part.part_name
                supplier_part_ids = self.inventory_manager.get_supplier_part_ids_by_name(warehouse_part_name)

                # Simple: Assume the order implicitly refers to the 'best' supplier found by WarehouseAgent
                # More robust: Order could store supplier_part_id, or SupplierAgent iterates potential suppliers
                supplier_item_found = None
                supplier_part_id_used = None
                for sup_part_id in supplier_part_ids:
                    supplier_item = self.inventory_manager.supplier_inventory.get(sup_part_id)
                    # Check stock AND maybe reliability/preference?
                    if supplier_item and supplier_item.available_quantity >= order.quantity:
                         # Assume this is the intended supplier for this order
                         supplier_item_found = supplier_item
                         supplier_part_id_used = sup_part_id
                         break # Found one that can fulfill

                if supplier_item_found:
                    # Approve: Decrease available quantity, mark IN_TRANSIT
                    supplier_item_found.available_quantity -= order.quantity
                    supplier_item_found.last_updated = datetime.datetime.now()
                    # Status becomes IN_TRANSIT (logically shipped)
                    order.update_status(STATUS_IN_TRANSIT, f"Supplier {supplier_part_id_used} approved & shipped. Available stock reduced by {order.quantity}.")
                    self.logger.info(f"Supplier approved and shipped order: {order}. Supplier stock for {supplier_part_id_used} now {supplier_item_found.available_quantity}")
                    approved_count += 1
                    reasoning = f"Approved: Supplier {supplier_part_id_used} stock ({supplier_item_found.available_quantity + order.quantity}) >= requested ({order.quantity}). Stock updated."
                    decision_type = "approve_supplier_order"
                    self.inventory_manager.add_event("order_shipped_supplier", f"Supplier {supplier_part_id_used} shipped order {order.order_id} ({order.quantity}x {warehouse_part_name}).", entity_id=order.order_id, severity=1)

                else:
                    # Cancel: No suitable supplier part found or insufficient stock
                    max_avail = 0
                    for sup_part_id in supplier_part_ids:
                         item = self.inventory_manager.supplier_inventory.get(sup_part_id)
                         if item: max_avail = max(max_avail, item.available_quantity)

                    current_stock_info = f"Max available from suppliers for '{warehouse_part_name}': {max_avail}"
                    order.update_status(STATUS_CANCELLED, f"Supplier cancelled: Insufficient stock available for {warehouse_part_name} ({order.quantity} requested). {current_stock_info}.")
                    self.logger.warning(f"Supplier cancelled order due to insufficient stock: {order}. {current_stock_info}")
                    cancelled_count += 1
                    reasoning = f"Cancelled: Insufficient supplier stock for {warehouse_part_name}. Requested: {order.quantity}, {current_stock_info}."
                    decision_type = "cancel_supplier_order_stock"
                    self.inventory_manager.add_event("order_cancelled_supplier", f"Supplier cancelled order {order.order_id} ({order.quantity}x {warehouse_part_name}) due to stock. {current_stock_info}.", entity_id=order.order_id, severity=3)


                # Log the decision
                self.log_decision(
                    decision_type=decision_type,
                    entity_id=order.order_id,
                    reasoning=reasoning,
                    ai_involved=False # Rule-based approval
                )

        if processed_count > 0:
             self.logger.info(f"Reviewed {processed_count} warehouse orders: {approved_count} approved, {cancelled_count} cancelled.")

    # detect_sales_patterns seems out of scope for SupplierAgent based on this data
    # def detect_sales_patterns(self): ...


# --- NEW: Assembly Line Agent ---
class AssemblyLineAgent(AmbientAgent):
    def __init__(self, inventory_manager: InventoryManager, assembly_data_file: str):
        super().__init__(inventory_manager, "AssemblyLineAgent")
        self.assembly_data_file = assembly_data_file
        self._interval = 60.0 * 10 # Check assembly data less frequently, e.g., every 10 mins
        self.prediction_horizon_days = 7 # Predict needs for the next 7 days
        self.min_confidence_for_action = 0.65 # Min AI confidence to trigger proactive order

    def get_agent_interval(self) -> float:
        return self._interval

    def _run_agent_cycle(self):
        """Main logic for a single cycle of the assembly line agent."""
        self.logger.debug(f"{self.agent_type} - Running cycle")

        # 1. Refresh assembly data
        self.inventory_manager.refresh_assembly_data()
        assembly_df = self.inventory_manager.assembly_line_data
        if assembly_df.empty:
             self.logger.warning("Assembly data is empty, cannot perform prediction.")
             return # Skip cycle if no data

        # 2. Predict potential shortages
        potential_shortages = self.predict_component_shortages(assembly_df)
        if not potential_shortages:
            self.logger.info("No potential component shortages predicted in the near future.")
            return

        # 3. Check inventory and take action for predicted shortages
        self.take_proactive_measures(potential_shortages)

        # No direct saving here, relies on system-level save
        self.logger.debug(f"{self.agent_type} - Cycle finished")

    def predict_component_shortages(self, assembly_df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyzes recent assembly data and uses AI (if enabled) or heuristics
        to predict likely component shortages in the near future.
        Returns a dictionary: {component_name: likelihood_score}.
        """
        self.logger.info(f"Predicting component shortages for the next {self.prediction_horizon_days} days...")
        potential_shortages = {}

        if assembly_df.empty:
             return potential_shortages

        # --- Method 1: Heuristic (Frequency of Past Shortages) ---
        # Look at data from the last N days (e.g., 30)
        try:
             recent_data = assembly_df[assembly_df['Date'] > (pd.Timestamp.now() - pd.Timedelta(days=30))]
             if recent_data.empty:
                  self.logger.warning("No recent assembly data (last 30 days) found for heuristic prediction.")
                  # Fallback or proceed to AI if enabled
             else:
                 # Count non-'None' shortages
                 shortage_counts = recent_data[recent_data['Component Shortages'].str.lower() != 'none']['Component Shortages'].value_counts()
                 total_recent_entries = len(recent_data)

                 if not shortage_counts.empty:
                      self.logger.info(f"Recent Shortage Counts (last 30 days):\n{shortage_counts}")
                      for component, count in shortage_counts.items():
                           # Simple likelihood: frequency
                           likelihood = count / total_recent_entries
                           # Boost likelihood if shortage happened very recently (e.g., last 7 days)
                           if recent_data[recent_data['Date'] > (pd.Timestamp.now() - pd.Timedelta(days=7))]['Component Shortages'].eq(component).any():
                                likelihood = min(1.0, likelihood * 1.5) # Boost score

                           potential_shortages[component] = round(likelihood, 3) # Store likelihood

        except Exception as e:
            self.logger.error(f"Error during heuristic shortage prediction: {e}", exc_info=True)


        # --- Method 2: AI-based Prediction (using Gemini) ---
        if self.ai_enabled:
            self.logger.info("Using AI to enhance/generate shortage predictions...")
            # Prepare a summary of recent data for the prompt
            try:
                if 'Date' not in assembly_df.columns or not pd.api.types.is_datetime64_any_dtype(assembly_df['Date']):
                     logger.error("AI Prediction skipped: 'Date' column is missing or not datetime type.")
                     return potential_shortages # Return heuristic results if AI fails

                recent_data_ai = assembly_df[assembly_df['Date'] > (pd.Timestamp.now() - pd.Timedelta(days=14))] # Use last 14 days for AI context
                if recent_data_ai.empty:
                     summary = "No production data recorded in the last 14 days."
                else:
                    avg_prod = recent_data_ai['Actual Production'].mean()
                    avg_defect = recent_data_ai['Defective Units'].mean()
                    recent_shortages = recent_data_ai[recent_data_ai['Component Shortages'].str.lower() != 'none']['Component Shortages'].unique().tolist()
                    lines_active = recent_data_ai['Assembly Line'].nunique()

                    summary = (f"Recent Assembly Summary (last 14 days):\n"
                               f"- Avg Daily Production per Line: {avg_prod:.1f} units\n"
                               f"- Avg Daily Defects per Line: {avg_defect:.1f} units\n"
                               f"- Components reported short recently: {', '.join(recent_shortages) if recent_shortages else 'None'}\n"
                               f"- Number of Active Lines: {lines_active}\n"
                               f"- Most Recent Date in Data: {recent_data_ai['Date'].max().strftime('%Y-%m-%d')}")

                prompt = f"""
                {summary}

                Based *only* on the provided recent assembly performance summary, predict which components, if any, are most likely to experience shortages in the next {self.prediction_horizon_days} days.
                List the component names you predict might face shortages. If multiple, list the most likely first.
                If no shortages seem likely based *only* on this data, state "No specific shortages predicted".

                Format your response as:
                PREDICTED_SHORTAGES: [Component Name 1, Component Name 2, ...] or PREDICTED_SHORTAGES: [None]
                """
                system_prompt = f"You are an AI assistant analyzing car assembly line data to predict potential component shortages within the next {self.prediction_horizon_days} days. Focus *only* on the provided summary data."

                ai_result = self.ask_ai(prompt, system_prompt)

                if not ai_result["error"] and ai_result["confidence"] > 0.5: # Trust AI prediction if confidence is reasonable
                    response_text = ai_result["response"]
                    self.logger.info(f"AI Prediction Response (Conf: {ai_result['confidence']:.2f}): {response_text}")
                    # Parse the response
                    import re
                    match = re.search(r"PREDICTED_SHORTAGES:\s*\[(.*?)\]", response_text, re.IGNORECASE | re.DOTALL)
                    if match:
                         predicted_list_str = match.group(1).strip()
                         if predicted_list_str.lower() != 'none' and predicted_list_str:
                             # Split by comma, strip whitespace and quotes
                             predicted_components = [p.strip().strip("'\"") for p in predicted_list_str.split(',')]
                             # Update/add likelihoods based on AI prediction order (higher likelihood for earlier items)
                             num_predicted = len(predicted_components)
                             for i, comp in enumerate(predicted_components):
                                 if comp: # Ensure not empty string
                                     # Assign higher likelihood to items mentioned by AI, especially first ones
                                     ai_likelihood = max(0.6, 1.0 - (i * 0.1)) # e.g., 1.0, 0.9, 0.8... capped at 0.6
                                     # Combine with heuristic score if exists (e.g., average or max)
                                     heuristic_likelihood = potential_shortages.get(comp, 0.0)
                                     combined_likelihood = max(ai_likelihood, heuristic_likelihood) # Take the max score
                                     potential_shortages[comp] = round(combined_likelihood, 3)
                                     self.logger.info(f"AI identified potential shortage for '{comp}' with combined likelihood {potential_shortages[comp]:.3f}")

                         else:
                              self.logger.info("AI explicitly predicted no specific shortages.")
                              # Optionally clear heuristic predictions if AI is confident? Risky.
                    else:
                         self.logger.warning("Could not parse PREDICTED_SHORTAGES from AI response.")
                elif ai_result["error"]:
                     self.logger.error(f"AI prediction failed: {ai_result['error']}. Relying on heuristics.")
                else:
                    self.logger.warning(f"AI prediction confidence ({ai_result['confidence']:.2f}) too low. Relying on heuristics.")

            except Exception as e:
                self.logger.error(f"Error during AI shortage prediction: {e}", exc_info=True)
                # Fallback to heuristic results stored earlier


        # Filter predictions below a certain threshold? Maybe not, let take_proactive_measures decide.
        if potential_shortages:
            self.logger.warning(f"Potential component shortages identified: {potential_shortages}")
        else:
             self.logger.info("Analysis complete. No significant shortage risks identified.")

        return potential_shortages


    def take_proactive_measures(self, potential_shortages: Dict[str, float]):
        """
        Checks inventory for potentially short components and creates supplier orders if needed.
        """
        self.logger.info("Taking proactive measures based on predicted shortages...")

        with self.inventory_manager.inventory_lock, self.inventory_manager.order_lock:
            for component_name, likelihood in potential_shortages.items():
                if likelihood < 0.5: # Don't act on very low likelihoods (heuristic noise)
                    self.logger.info(f"Skipping action for '{component_name}': Likelihood ({likelihood:.2f}) below threshold (0.5).")
                    continue

                self.logger.warning(f"Processing potential shortage for '{component_name}' (Likelihood: {likelihood:.2f})")

                # 1. Find the corresponding part ID in the warehouse inventory
                warehouse_part_id = self.inventory_manager.get_warehouse_part_id_by_name(component_name)
                if not warehouse_part_id:
                    self.logger.error(f"Cannot take action for '{component_name}': Part name not found in warehouse inventory map.")
                    self.inventory_manager.add_event("prediction_action_fail", f"Action failed for predicted shortage of '{component_name}': Part not in warehouse.", entity_id=component_name, severity=4)
                    continue

                warehouse_item = self.inventory_manager.warehouse_inventory.get(warehouse_part_id)
                if not warehouse_item: # Should not happen if map is correct, but check
                     self.logger.error(f"Data inconsistency: Warehouse part ID {warehouse_part_id} found for '{component_name}' but item not in inventory dict.")
                     continue

                # 2. Check if a supplier order is already pending/in-transit
                if self.inventory_manager.has_pending_supplier_order(warehouse_part_id):
                    self.logger.info(f"Action skipped for '{component_name}': Existing supplier order found for part {warehouse_part_id}.")
                    continue

                # 3. Assess current stock vs. threshold + buffer
                # Define a buffer based on prediction horizon and maybe lead time
                # Simple buffer: aim to be above min_threshold + X days of consumption
                # Estimate consumption (very rough): Avg daily prod * num lines (needs better model)
                # Let's use a simpler threshold check for now: Is stock below min + 25%?
                safety_threshold = warehouse_item.min_threshold * 1.25
                current_stock = warehouse_item.quantity

                if current_stock >= safety_threshold:
                    self.logger.info(f"Action skipped for '{component_name}': Current stock ({current_stock}) is above safety threshold ({safety_threshold:.1f}).")
                    continue

                self.logger.warning(f"Proactive order needed for '{component_name}' ({warehouse_part_id}): Stock {current_stock} < Safety Threshold {safety_threshold:.1f}. Likelihood: {likelihood:.2f}")

                # 4. Determine order quantity (similar to WarehouseAgent's logic)
                target_level = int(warehouse_item.max_capacity * 0.9)
                quantity_to_order = max(0, target_level - current_stock)

                if quantity_to_order == 0:
                     self.logger.info(f"No order needed for {component_name}, target level met despite being below safety threshold.")
                     continue

                # 5. Find Supplier and check availability (reuse WarehouseAgent logic basis)
                supplier_part_ids = self.inventory_manager.get_supplier_part_ids_by_name(component_name)
                best_supplier_part_id = None
                supplier_lead_time = 7 # Default
                supplier_info_for_order = None

                if not supplier_part_ids:
                    self.logger.error(f"Cannot create proactive order for '{component_name}': No supplier found.")
                    self.inventory_manager.add_event("proactive_order_fail", f"Proactive order for '{component_name}' failed: No supplier.", entity_id=warehouse_part_id, severity=4)
                    continue

                # Simple supplier selection (first with enough stock)
                for sup_part_id in supplier_part_ids:
                    supplier_item = self.inventory_manager.supplier_inventory.get(sup_part_id)
                    if supplier_item and supplier_item.available_quantity >= quantity_to_order:
                        best_supplier_part_id = sup_part_id
                        supplier_lead_time = supplier_item.lead_time_days
                        supplier_info_for_order = supplier_item
                        break

                if not best_supplier_part_id:
                    max_avail = 0
                    for sup_part_id in supplier_part_ids:
                         item = self.inventory_manager.supplier_inventory.get(sup_part_id)
                         if item: max_avail = max(max_avail, item.available_quantity)
                    self.logger.warning(f"Cannot create proactive order for {quantity_to_order} of '{component_name}': No supplier has enough stock (max avail: {max_avail}).")
                    self.inventory_manager.add_event("proactive_order_fail", f"Proactive order for '{component_name}' ({quantity_to_order}) failed: Supplier stock low (max: {max_avail}).", entity_id=warehouse_part_id, severity=3)
                    continue

                # --- Optional: AI Refinement for Proactive Order Qty ---
                ai_recommendation = None
                ai_confidence = 0.0
                reasoning = f"Proactive order due to predicted shortage (Likelihood: {likelihood:.2f}). Stock {current_stock} < safety {safety_threshold:.1f}. Ordering {quantity_to_order} to target {target_level} from supplier part {best_supplier_part_id}."

                if self.ai_enabled and likelihood >= self.min_confidence_for_action and supplier_info_for_order:
                    # Similar prompt as WarehouseAgent but emphasizing the prediction
                    prompt = f"""
                    ACTION: Proactive Supplier Order Recommendation.
                    REASON: Predicted shortage risk (Likelihood: {likelihood:.2f}) for warehouse part '{component_name}' (ID: {warehouse_part_id}).
                    Current Warehouse Qty: {current_stock}, Min Threshold: {warehouse_item.min_threshold}, Max Capacity: {warehouse_item.max_capacity}, Safety Threshold: {safety_threshold:.1f}.
                    Default Proactive Order Qty: {quantity_to_order} (to reach target {target_level}).
                    Selected Supplier Part ID: {best_supplier_part_id}
                    Supplier Available Qty: {supplier_info_for_order.available_quantity}, Lead Time: {supplier_info_for_order.lead_time_days} days.

                    Considering the prediction likelihood, lead time, and current stock relative to safety threshold, is {quantity_to_order} the right proactive quantity?
                    Suggest adjustments if the risk or lead time warrants ordering more (up to available space/supplier stock) or less.
                    Output only: 'QUANTITY: <number>, REASON: <text>'. Example: 'QUANTITY: 200, REASON: High likelihood and lead time require larger buffer.'
                    """
                    system_prompt = "You are an inventory optimization assistant validating proactive orders based on shortage predictions for a car parts warehouse."
                    ai_result = self.ask_ai(prompt, system_prompt)
                    ai_recommendation = ai_result["response"]
                    ai_confidence = ai_result["confidence"]

                    if not ai_result["error"] and ai_confidence > 0.5:
                        # Parse and potentially adjust quantity_to_order (same logic as WarehouseAgent AI check)
                         import re
                         match = re.search(r"QUANTITY:\s*(\d+)", ai_recommendation, re.IGNORECASE)
                         if match:
                            try:
                                ai_quantity = int(match.group(1))
                                space_available = warehouse_item.max_capacity - current_stock
                                order_limit = min(supplier_info_for_order.available_quantity, space_available)
                                if 0 < ai_quantity <= order_limit:
                                    if ai_quantity != quantity_to_order:
                                        self.logger.info(f"AI suggested adjusting proactive order for {component_name} from {quantity_to_order} to {ai_quantity}. Limit: {order_limit}. Conf: {ai_confidence:.2f}")
                                        quantity_to_order = ai_quantity
                                        reasoning += f" AI Adjusted Qty: {ai_quantity}. AI Reason: {ai_recommendation}"
                                    else: reasoning += " AI Confirmed proactive qty."
                                else:
                                     self.logger.warning(f"AI suggested invalid proactive quantity {ai_quantity} for {component_name} (Limit: {order_limit}). Using default {quantity_to_order}.")
                                     reasoning += f" AI suggested invalid qty ({ai_quantity}). Using default."
                            except ValueError: reasoning += " Could not parse AI proactive qty. Using default."
                         else: reasoning += " AI did not provide specific proactive qty. Using default."
                    elif ai_result["error"]: reasoning += f" AI suggestion failed: {ai_result['error']}"
                    else: reasoning += f" AI confidence low ({ai_confidence:.2f}). Using default proactive qty."
                # --- End AI Refinement ---


                # 6. Create the PENDING Supplier Order
                order_id = self.inventory_manager.generate_order_id()
                expected_del_date = (datetime.datetime.now() + datetime.timedelta(days=supplier_lead_time)).strftime("%Y-%m-%d")

                order = Order(
                    order_id=order_id,
                    part_id=warehouse_part_id, # Warehouse part ID
                    quantity=quantity_to_order,
                    order_type=TYPE_WAREHOUSE_TO_SUPPLIER,
                    status=STATUS_PENDING, # Supplier needs to approve
                    order_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                    expected_delivery=expected_del_date,
                    priority=4, # High priority due to predicted shortage risk
                    ai_recommendation=ai_recommendation,
                    confidence_score=ai_confidence
                )
                self.inventory_manager.order_history.append(order)
                self.logger.warning(f"CREATED PROACTIVE supplier order: {order}") # Log as warning because it's a significant action

                # Log decision
                self.log_decision(
                    decision_type="create_proactive_supplier_order",
                    entity_id=order.order_id,
                    reasoning=reasoning,
                    ai_involved=bool(ai_recommendation),
                    confidence_score=ai_confidence
                )
                # Log event
                self.inventory_manager.add_event(
                    "proactive_order_created",
                    f"Predicted shortage risk for '{component_name}' (Likelihood: {likelihood:.2f}) triggered proactive supplier order {order_id}.",
                    entity_id=warehouse_part_id,
                    entity_type="inventory_item",
                    severity=2 # Informational / Warning
                )


# --- InventorySystem Class (Integrate AssemblyLineAgent) ---
class InventorySystem:
    def __init__(self):
        logger.info("Initializing Inventory System...")
        self.inventory_manager = InventoryManager(
            showroom_file=SHOWROOM_CSV,
            warehouse_file=WAREHOUSE_CSV,
            supplier_file=SUPPLIER_CSV,
            order_history_file=ORDER_HISTORY_CSV,
            events_log_file=EVENTS_LOG_CSV,
            agent_decisions_file=AGENT_DECISIONS_CSV,
            assembly_line_file=ASSEMBLY_LINE_CSV # Pass assembly file path
        )
        self.showroom_agent = ShowroomAmbientAgent(self.inventory_manager)
        self.warehouse_agent = WarehouseAgent(self.inventory_manager)
        self.supplier_agent = SupplierAgent(self.inventory_manager)
        self.assembly_agent = AssemblyLineAgent(self.inventory_manager, ASSEMBLY_LINE_CSV) # <-- Initialize Assembly Agent

        self.agents = [
            self.showroom_agent,
            self.warehouse_agent,
            self.supplier_agent,
            self.assembly_agent # <-- Add to list
        ]
        self.running = False
        self._shutdown_event = threading.Event()
        self._save_timer_thread = None

    def _periodic_save(self, interval_seconds=300): # Save every 5 minutes
         """Runs in a separate thread to periodically save data."""
         logger.info(f"Periodic save task started. Interval: {interval_seconds} seconds.")
         while not self._shutdown_event.wait(interval_seconds): # Wait for interval or shutdown
             logger.info("Performing periodic data save...")
             try:
                 self.inventory_manager.save_inventory()
             except Exception as e:
                  logger.error(f"Error during periodic save: {e}", exc_info=True)
         logger.info("Periodic save task stopped.")


    def start(self):
        """Start the inventory system and all its agents."""
        if self.running:
            logger.warning("System is already running")
            return

        logger.info("Starting Inventory System...")
        self.running = True
        self._shutdown_event.clear()

        # Start the periodic save thread
        self._save_timer_thread = threading.Thread(target=self._periodic_save, name="SaveTimerThread")
        self._save_timer_thread.daemon = True
        self._save_timer_thread.start()

        # Start all agents
        for agent in self.agents:
            agent.start()

        logger.info("Inventory system and all agents started successfully.")

    def stop(self):
        """Stop the inventory system and all its agents gracefully."""
        if not self.running:
            logger.warning("System is not running")
            return

        logger.info("Stopping Inventory System...")
        self.running = False

        # Signal and stop all agents
        logger.info("Signalling agents to stop...")
        for agent in self.agents:
            agent.stop() # This now waits for threads

        # Signal and stop the periodic save timer
        logger.info("Stopping periodic save timer...")
        self._shutdown_event.set()
        if self._save_timer_thread and self._save_timer_thread.is_alive():
             self._save_timer_thread.join(timeout=5.0) # Wait for save thread
             if self._save_timer_thread.is_alive():
                   logger.warning("Save timer thread did not exit cleanly.")


        logger.info("Performing final data save...")
        # Perform one final save after all agents have stopped processing
        try:
             self.inventory_manager.save_inventory()
             logger.info("Final data save complete.")
        except Exception as e:
             logger.error(f"Error during final data save: {e}", exc_info=True)


        logger.info("Inventory system stopped.")


# Main function to run the inventory system
def main():
    logger.info("--- Ambient Inventory System with Assembly Prediction ---")

    # Ensure required CSV files exist with headers before starting
    # (InventoryManager attempts this, but good to be sure)
    # You might add explicit checks/creation here if needed

    inventory_system = InventorySystem()

    try:
        inventory_system.start()

        # Keep the main thread alive, waiting for keyboard interrupt
        while inventory_system.running:
            # You could add commands here like 'status', 'force_save', etc.
            # Example: Check agent status periodically
            # time.sleep(60)
            # for agent in inventory_system.agents:
            #    status = "Running" if agent.running and agent.thread and agent.thread.is_alive() else "Stopped"
            #    logger.debug(f"Agent {agent.agent_type} status: {status}")

            # Simple sleep to prevent busy-waiting
             time.sleep(1.0)


    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Initiating shutdown...")
    except Exception as e:
         logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        logger.info("Starting system shutdown process...")
        inventory_system.stop()
        logger.info("--- System Shutdown Complete ---")

if __name__ == "__main__":
    # Add a check for the API key at the very start
    if not GEMINI_API_KEY:
        logger.critical("CRITICAL: Gemini API Key is missing or invalid. AI features will be DISABLED.")
        user_input = input("Continue without AI features? (yes/no): ")
        if user_input.lower() != 'yes':
            print("Exiting.")
            exit(1) # Exit if user doesn't want to continue without AI

    main()