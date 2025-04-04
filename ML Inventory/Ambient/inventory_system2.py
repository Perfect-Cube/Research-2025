import os
import csv
import datetime
import uuid
import time
import logging
import threading
import json
import requests
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from dotenv import load_dotenv

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
api_key = "AIzaSyABOm1epbQ6T4bX-hjKLrgfTrKoxQ3Jnl0"  # Replace with your actual API key
genai.configure(api_key=api_key)
GEMINI_API_KEY =api_key
 
model = genai.GenerativeModel('gemini-1.5-flash-001')

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
        self.historical_demand = historical_demand or [1, 2, 1, 3, 2]
        self.seasonal_factors = seasonal_factors or {}
        self.vendor_reliability = float(vendor_reliability)
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
            "demand_trend": self.demand_trend
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
        self.priority = int(priority)  # 1-5, with 5 being highest priority
        self.ai_recommendation = ai_recommendation
        self.confidence_score = float(confidence_score)
        self.status_history = [(status, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]
        self.notes = []

    def __str__(self):
        return f"Order {self.order_id}: {self.quantity} units of {self.part_id}, Status: {self.status}, Priority: {self.priority}"
    
    def to_dict(self):
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
            "status_history": self.status_history,
            "notes": self.notes
        }
    
    def update_status(self, new_status, note=None):
        """Update order status and record the change in history"""
        self.status = new_status
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status_history.append((new_status, timestamp))
        if note:
            self.notes.append((timestamp, note))

class Event:
    def __init__(self, event_id, event_type, description, timestamp=None, 
                 entity_id=None, entity_type=None, severity=1, ai_analysis=None):
        self.event_id = event_id
        self.event_type = event_type
        self.description = description
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.severity = severity  # 1-5, with 5 being most severe
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
        self.entity_id = entity_id
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.reasoning = reasoning
        self.ai_involved = ai_involved
        self.confidence_score = confidence_score
        self.alternatives = alternatives or []
    
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
            "alternatives": self.alternatives
        }

class InventoryManager:
    def __init__(self, showroom_file=SHOWROOM_CSV, warehouse_file=WAREHOUSE_CSV, 
                supplier_file=SUPPLIER_CSV, order_history_file=ORDER_HISTORY_CSV,
                events_log_file=EVENTS_LOG_CSV, agent_decisions_file=AGENT_DECISIONS_CSV):
        self.showroom_file = showroom_file
        self.warehouse_file = warehouse_file
        self.supplier_file = supplier_file
        self.order_history_file = order_history_file
        self.events_log_file = events_log_file
        self.agent_decisions_file = agent_decisions_file
        
        # Load inventory data
        self.showroom_inventory = self._load_inventory(showroom_file)
        self.warehouse_inventory = self._load_inventory(warehouse_file)
        self.supplier_inventory = self._load_inventory(supplier_file, is_supplier=True)
        self.order_history = self._load_orders(order_history_file)
        self.events_log = self._load_events(events_log_file)
        self.agent_decisions = self._load_decisions(agent_decisions_file)
        
        # Initialize locks for thread safety
        self.inventory_lock = threading.RLock()
        self.order_lock = threading.RLock()
        self.event_lock = threading.RLock()
        self.decision_lock = threading.RLock()

    def _load_inventory(self, filename, is_supplier=False):
        inventory = {}
        if not os.path.exists(filename):
            logger.warning(f"File {filename} not found. Creating an empty inventory.")
            return inventory

        try:
            with open(filename, 'r', encoding='utf-8-sig', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if is_supplier:
                        item = InventoryItem(
                            row["part_id"], row["part_name"], row["category"],
                            0, 0, 0, row["price"], row["lead_time_days"], row["available_quantity"],
                            vendor_reliability=row.get("vendor_reliability", 1.0)
                        )
                    else:
                        item = InventoryItem(
                            row["part_id"], row["part_name"], row["category"],
                            row["quantity"], row["min_threshold"], row["max_capacity"]
                        )
                    inventory[row["part_id"]] = item
        except Exception as e:
            logger.error(f"Error loading inventory from {filename}: {str(e)}")
        return inventory

    def _load_orders(self, filename):
        orders = []
        if not os.path.exists(filename):
            logger.warning(f"File {filename} not found. Creating an empty order history.")
            return orders

        try:
            with open(filename, 'r', encoding='utf-8-sig', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    order = Order(
                        row["order_id"], row["part_id"], row["quantity"],
                        row["order_type"], row["status"], row["order_date"],
                        row.get("expected_delivery"),
                        priority=row.get("priority", 1),
                        ai_recommendation=row.get("ai_recommendation"),
                        confidence_score=row.get("confidence_score", 0.0)
                    )
                    orders.append(order)
        except Exception as e:
            logger.error(f"Error loading orders from {filename}: {str(e)}")
        return orders

    def _load_events(self, filename):
        events = []
        if not os.path.exists(filename):
            logger.warning(f"File {filename} not found. Creating an empty events log.")
            return events

        try:
            with open(filename, 'r', encoding='utf-8-sig', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    event = Event(
                        row["event_id"], row["event_type"], row["description"],
                        row["timestamp"], row.get("entity_id"), row.get("entity_type"),
                        int(row.get("severity", 1)), row.get("ai_analysis")
                    )
                    events.append(event)
        except Exception as e:
            logger.error(f"Error loading events from {filename}: {str(e)}")
        return events

    def _load_decisions(self, filename):
        decisions = []
        if not os.path.exists(filename):
            logger.warning(f"File {filename} not found. Creating an empty decisions log.")
            return decisions

        try:
            with open(filename, 'r', encoding='utf-8-sig', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    alternatives = []
                    if row.get("alternatives"):
                        try:
                            alternatives = json.loads(row["alternatives"])
                        except:
                            pass
                    
                    decision = AgentDecision(
                        row["decision_id"], row["agent_type"], row["decision_type"],
                        row["entity_id"], row["timestamp"], row.get("reasoning"),
                        row.get("ai_involved", "False").lower() == "true",
                        float(row.get("confidence_score", 0.0)), alternatives
                    )
                    decisions.append(decision)
        except Exception as e:
            logger.error(f"Error loading decisions from {filename}: {str(e)}")
        return decisions

    def save_inventory(self):
        """Save all inventory data to CSV files"""
        with self.inventory_lock:
            self._save_inventory(self.showroom_file, self.showroom_inventory)
            self._save_inventory(self.warehouse_file, self.warehouse_inventory)
            self._save_supplier_inventory(self.supplier_file, self.supplier_inventory)
        
        with self.order_lock:
            self._save_orders(self.order_history_file, self.order_history)
        
        with self.event_lock:
            self._save_events(self.events_log_file, self.events_log)
        
        with self.decision_lock:
            self._save_decisions(self.agent_decisions_file, self.agent_decisions)

    def _save_inventory(self, filename, inventory):
        fieldnames = ["part_id", "part_name", "category", "quantity", "min_threshold", "max_capacity"]
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for item in inventory.values():
                    writer.writerow({
                        "part_id": item.part_id,
                        "part_name": item.part_name,
                        "category": item.category,
                        "quantity": item.quantity,
                        "min_threshold": item.min_threshold,
                        "max_capacity": item.max_capacity
                    })
        except Exception as e:
            logger.error(f"Error saving inventory to {filename}: {str(e)}")

    def _save_supplier_inventory(self, filename, inventory):
        fieldnames = ["part_id", "part_name", "category", "available_quantity", "price", "lead_time_days", "vendor_reliability"]
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for item in inventory.values():
                    writer.writerow({
                        "part_id": item.part_id,
                        "part_name": item.part_name,
                        "category": item.category,
                        "available_quantity": item.available_quantity,
                        "price": item.price,
                        "lead_time_days": item.lead_time_days,
                        "vendor_reliability": item.vendor_reliability
                    })
        except Exception as e:
            logger.error(f"Error saving supplier inventory to {filename}: {str(e)}")

    def _save_orders(self, filename, orders):
        fieldnames = ["order_id", "part_id", "quantity", "order_type", "status", "order_date", 
                      "expected_delivery", "priority", "ai_recommendation", "confidence_score"]
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for order in orders:
                    writer.writerow({
                        "order_id": order.order_id,
                        "part_id": order.part_id,
                        "quantity": order.quantity,
                        "order_type": order.order_type,
                        "status": order.status,
                        "order_date": order.order_date,
                        "expected_delivery": order.expected_delivery,
                        "priority": order.priority,
                        "ai_recommendation": order.ai_recommendation,
                        "confidence_score": order.confidence_score
                    })
        except Exception as e:
            logger.error(f"Error saving orders to {filename}: {str(e)}")

    def _save_events(self, filename, events):
        fieldnames = ["event_id", "event_type", "description", "timestamp", "entity_id", 
                      "entity_type", "severity", "ai_analysis"]
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for event in events:
                    writer.writerow(event.to_dict())
        except Exception as e:
            logger.error(f"Error saving events to {filename}: {str(e)}")

    def _save_decisions(self, filename, decisions):
        fieldnames = ["decision_id", "agent_type", "decision_type", "entity_id", "timestamp", 
                      "reasoning", "ai_involved", "confidence_score", "alternatives"]
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for decision in decisions:
                    row_data = decision.to_dict()
                    row_data["alternatives"] = json.dumps(decision.alternatives)
                    writer.writerow(row_data)
        except Exception as e:
            logger.error(f"Error saving decisions to {filename}: {str(e)}")

    def generate_id(self, prefix):
        """Generate a unique ID with the given prefix"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = str(uuid.uuid4())[:8]
        return f"{prefix}{timestamp}{random_suffix}"

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
            return decision

class AmbientAgent:
    """Base class for all ambient agents in the system"""
    def __init__(self, inventory_manager, agent_type):
        self.inventory_manager = inventory_manager
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"AmbientAgent.{agent_type}")
        self.ai_enabled = GEMINI_API_KEY is not None
        self.last_check_time = datetime.datetime.now()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the agent's background thread"""
        if self.running:
            self.logger.warning(f"{self.agent_type} is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_agent)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info(f"{self.agent_type} started")
    
    def stop(self):
        """Stop the agent's background thread"""
        if not self.running:
            self.logger.warning(f"{self.agent_type} is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        self.logger.info(f"{self.agent_type} stopped")
    
    def _run_agent(self):
        """Main loop for the agent - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _run_agent method")
    
    def ask_ai(self, prompt, system_prompt=None):
        """Use Gemini API to get AI recommendations"""
        if not self.ai_enabled:
            return {"response": "AI features are disabled", "confidence": 0.0}
        
        try:
            if system_prompt:
                generation_config = {"temperature": 0.2, "top_p": 0.8, "top_k": 40}
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
                
                response = model.generate_content(
                    [system_prompt, prompt],
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
            else:
                response = model.generate_content(prompt)
            
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                text_response = response.candidates[0].content.parts[0].text
                # Estimate confidence based on the decisive language in the response
                confidence_words = ["confident", "certain", "definitely", "absolutely", "clearly"]
                uncertainty_words = ["maybe", "perhaps", "possibly", "uncertain", "might"]
                
                confidence = 0.7  # Default medium-high confidence
                for word in confidence_words:
                    if word in text_response.lower():
                        confidence += 0.05
                for word in uncertainty_words:
                    if word in text_response.lower():
                        confidence -= 0.1
                
                confidence = max(0.1, min(0.95, confidence))  # Clamp between 0.1 and 0.95
                
                return {"response": text_response, "confidence": confidence}
            else:
                return {"response": "No response generated", "confidence": 0.0}
        except Exception as e:
            self.logger.error(f"Error in AI request: {str(e)}")
            return {"response": f"Error: {str(e)}", "confidence": 0.0}
    
    def log_decision(self, decision_type, entity_id, reasoning=None, ai_involved=False, 
                   confidence_score=0.0, alternatives=None):
        """Log a decision made by this agent"""
        return self.inventory_manager.add_decision(
            self.agent_type, decision_type, entity_id, reasoning, 
            ai_involved, confidence_score, alternatives
        )

class ShowroomAmbientAgent(AmbientAgent):
    def __init__(self, inventory_manager):
        super().__init__(inventory_manager, "ShowroomAgent")
        
    def _run_agent(self):
        """Main loop for the showroom agent"""
        while self.running:
            try:
                # Perform periodic tasks
                self.check_inventory()
                self.process_approved_orders()
                self.detect_sales_patterns()
                
                # Save state
                self.inventory_manager.save_inventory()
                
                # Wait for next cycle
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in showroom agent loop: {str(e)}")
                time.sleep(30)  # Wait longer if there's an error
    
    def detect_sales_patterns(self):
        """Analyze sales data to identify patterns and trends"""
        self.logger.info("Detecting sales patterns...")
        
        for part_id, item in self.inventory_manager.showroom_inventory.items():
            # Analyze historical demand
            if item.historical_demand:
                average_demand = sum(item.historical_demand) / len(item.historical_demand)
                current_demand = item.quantity  # Current stock can be used as a proxy for demand
                
                # Simple trend detection
                if current_demand < average_demand * 0.5:
                    self.logger.warning(f"Sales pattern detected: Low demand for {item.part_name}. Consider increasing stock.")
                elif current_demand > average_demand * 1.5:
                    self.logger.info(f"Sales pattern detected: High demand for {item.part_name}. Consider ordering more.")
            else:
                self.logger.info(f"No historical demand data for {item.part_name}.")
    
    def check_inventory(self):
        """Check showroom inventory for items below threshold"""
        self.logger.info("Checking showroom inventory...")
        current_time = datetime.datetime.now()
        
        # Only run full check once per hour
        if (current_time - self.last_check_time).total_seconds() < 3600:
            return []
        
        self.last_check_time = current_time
        orders = []
        
        with self.inventory_manager.inventory_lock:
            for part_id, item in self.inventory_manager.showroom_inventory.items():
                if item.quantity < item.min_threshold:
                    # Use AI to determine optimal order quantity if enabled
                    optimal_level = int(item.max_capacity * 0.75)  # Default strategy
                    quantity_to_order = optimal_level - item.quantity
                    ai_recommendation = None
                    confidence_score = 0.0
                    
                    if self.ai_enabled:
                        # Get AI recommendation for order quantity
                        prompt = f"""
                        I need to determine the optimal order quantity for a car part in our showroom inventory.
                        
                        Part ID: {part_id}
                        Part Name: {item.part_name}
                        Category: {item.category}
                        Current Quantity: {item.quantity}
                        Minimum Threshold: {item.min_threshold}
                        Maximum Capacity: {item.max_capacity}
                        
                        Our default strategy is to order up to 75% of maximum capacity.
                        Based on this information, should we follow our default strategy or adjust the order quantity?
                        If we should adjust, what quantity should we order and why?
                        
                        Please provide a specific quantity recommendation and your reasoning.
                        """
                        
                        system_prompt = """
                        You are an automotive parts inventory optimization assistant. Your goal is to help determine
                        optimal order quantities based on the current inventory state, historical patterns, and 
                        business constraints. Provide specific numerical recommendations with brief explanations.
                        """
                        
                        ai_result = self.ask_ai(prompt, system_prompt)
                        ai_recommendation = ai_result["response"]
                        confidence_score = ai_result["confidence"]
                        
                        # Try to extract a recommended quantity from the AI response
                        import re
                        quantity_match = re.search(r'recommend ordering (\d+)', ai_recommendation)
                        if quantity_match:
                            ai_quantity = int(quantity_match.group(1))
                            if ai_quantity > 0 and ai_quantity <= item.max_capacity - item.quantity:
                                quantity_to_order = ai_quantity
                    
                    # Check if warehouse has enough stock
                    if (part_id in self.inventory_manager.warehouse_inventory and 
                        self.inventory_manager.warehouse_inventory[part_id].quantity >= quantity_to_order):
                        
                        # Create order from showroom to warehouse
                        with self.inventory_manager.order_lock:
                            order_id = self.inventory_manager.generate_order_id()
                            order = Order(
                                order_id=order_id,
                                part_id=part_id,
                                quantity=quantity_to_order,
                                order_type=TYPE_SHOWROOM_TO_WAREHOUSE,
                                status=STATUS_PENDING,
                                order_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                                expected_delivery=(datetime.datetime.now() + 
                                                datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                                priority=3,  # Medium priority
                                ai_recommendation=ai_recommendation,
                                confidence_score=confidence_score
                            )
                            self.inventory_manager.order_history.append(order)
                            orders.append(order)
                        
                        self.logger.info(f"Created order: {order}")
                        
                        # Log the decision
                        self.log_decision(
                            "create_order", order.order_id, 
                            f"Created order for {quantity_to_order} units of {item.part_name} due to low inventory",
                            ai_involved=bool(ai_recommendation),
                            confidence_score=confidence_score
                        )
                        
                        # Log the event
                        self.inventory_manager.add_event(
                            "low_inventory", 
                            f"Showroom inventory for {item.part_name} below threshold. Created order {order.order_id}",
                            entity_id=part_id,
                            entity_type="inventory_item",
                            severity=2,
                            ai_analysis=ai_recommendation
                        )
        
        return orders

    def process_approved_orders(self):
        """Process orders that have been approved"""
        self.logger.info("Processing approved orders for showroom...")
        
        with self.inventory_manager.order_lock:
            for order in self.inventory_manager.order_history:
                if (order.status == STATUS_APPROVED and 
                    order.order_type == TYPE_SHOWROOM_TO_WAREHOUSE):
                    
                    # Update order status to in_transit
                    order.update_status(STATUS_IN_TRANSIT, "Order in transit from warehouse to showroom")
                    self.logger.info(f"Updated order status to in_transit: {order}")
                    
                    # Simulate delivery time (in a real system, this would be a separate process)
                    # For simplicity, we'll just set orders that are a day old to delivered
                    order_date = datetime.datetime.strptime(order.order_date, "%Y-%m-%d")
                    if (datetime.datetime.now() - order_date).days >= 1:
                        self.deliver_order(order)
    
    def deliver_order(self, order):
        """Deliver an order to the showroom"""
        if order.order_type == TYPE_SHOWROOM_TO_WAREHOUSE and order.status == STATUS_IN_TRANSIT:
            # Update inventory
            showroom_item = self.inventory_manager.showroom_inventory.get(order.part_id)
            warehouse_item = self.inventory_manager.warehouse_inventory.get(order.part_id)
            
            if showroom_item and warehouse_item and warehouse_item.quantity >= order.quantity:
                # Transfer from warehouse to showroom
                warehouse_item.quantity -= order.quantity
                showroom_item.quantity += order.quantity
                
                # Update order status
                order.update_status(STATUS_DELIVERED, "Order delivered to showroom")
                self.logger.info(f"Order delivered: {order}")
                
                # Log the event
                self.inventory_manager.add_event(
                    "order_delivered", 
                    f"Order {order.order_id} delivered to showroom for {order.part_id}",
                    entity_id=order.order_id,
                    entity_type="order",
                    severity=1
                )
                return True
            else:
                self.logger.warning(f"Cannot deliver order {order.order_id}: Insufficient inventory at warehouse")
                return False
        return False
    
class WarehouseAgent(AmbientAgent):
    def __init__(self, inventory_manager):
        super().__init__(inventory_manager, "WarehouseAgent")
        
    def _run_agent(self):
        """Main loop for the warehouse agent"""
        while self.running:
            try:
                self.check_inventory()
                self.review_showroom_orders()
                self.process_approved_orders()
                self.inventory_manager.save_inventory()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in warehouse agent loop: {str(e)}")
                time.sleep(30)  # Wait longer if there's an error

    def check_inventory(self):
        """Check warehouse inventory for items below threshold"""
        self.logger.info("Checking warehouse inventory...")
        for part_id, item in self.inventory_manager.warehouse_inventory.items():
            if item.quantity < item.min_threshold:
                # Create order to supplier if needed
                self.create_supplier_order(part_id, item)
    def process_approved_orders(self):

        """Process orders that have been approved"""

        self.logger.info("Processing approved orders for warehouse...")

        

        with self.inventory_manager.order_lock:

            for order in self.inventory_manager.order_history:

                if order.status == STATUS_APPROVED:

                    # Check if the warehouse can fulfill the order

                    warehouse_item = self.inventory_manager.warehouse_inventory.get(order.part_id)

                    

                    if warehouse_item and warehouse_item.quantity >= order.quantity:

                        # Fulfill the order

                        warehouse_item.quantity -= order.quantity

                        order.update_status(STATUS_IN_TRANSIT, "Order is now in transit to showroom")

                        self.logger.info(f"Order {order.order_id} is now in transit.")

                    else:

                        # If not enough stock, cancel the order

                        order.update_status(STATUS_CANCELLED, "Insufficient stock to fulfill the order")

                        self.logger.warning(f"Order {order.order_id} cancelled due to insufficient stock.")


    

    def create_supplier_order(self, part_id, warehouse_item):
        """Create an order from warehouse to supplier"""
        quantity_to_order = warehouse_item.max_capacity - warehouse_item.quantity
        if quantity_to_order > 0:
            order_id = self.inventory_manager.generate_order_id()
            order = Order(
                order_id=order_id,
                part_id=part_id,
                quantity=quantity_to_order,
                order_type=TYPE_WAREHOUSE_TO_SUPPLIER,
                status=STATUS_PENDING,
                order_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                expected_delivery=(datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            )
            self.inventory_manager.order_history.append(order)
            self.logger.info(f"Created supplier order: {order}")
            self.inventory_manager.add_event("supplier_order", f"Created order {order_id} for {quantity_to_order} units of {part_id}")

    def review_showroom_orders(self):
        """Review and approve/reject showroom orders"""
        self.logger.info("Reviewing showroom orders...")
        for order in self.inventory_manager.order_history:
            if order.status == STATUS_PENDING and order.order_type == TYPE_SHOWROOM_TO_WAREHOUSE:
                showroom_item = self.inventory_manager.showroom_inventory.get(order.part_id)
                if showroom_item and showroom_item.quantity >= order.quantity:
                    order.update_status(STATUS_APPROVED, "Order approved by warehouse")
                    self.logger.info(f"Approved order: {order}")
                else:
                    order.update_status(STATUS_CANCELLED, "Order cancelled due to insufficient stock")
                    self.logger.warning(f"Cancelled order: {order}")

class SupplierAgent(AmbientAgent):
    def __init__(self, inventory_manager):
        super().__init__(inventory_manager, "SupplierAgent")
        
    def _run_agent(self):
        """Main loop for the supplier agent"""
        while self.running:
            try:
                self.review_warehouse_orders()
                self.inventory_manager.save_inventory()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in supplier agent loop: {str(e)}")
                time.sleep(30)  # Wait longer if there's an error

    def review_warehouse_orders(self):
        """Review and approve/reject warehouse orders"""
        self.logger.info("Reviewing warehouse orders...")
        for order in self.inventory_manager.order_history:
            if order.status == STATUS_PENDING and order.order_type == TYPE_WAREHOUSE_TO_SUPPLIER:
                supplier_item = self.inventory_manager.supplier_inventory.get(order.part_id)
                if supplier_item and supplier_item.available_quantity >= order.quantity:
                    order.update_status(STATUS_APPROVED, "Order approved by supplier")
                    self.logger.info(f"Approved order: {order}")
                else:
                    order.update_status(STATUS_CANCELLED, "Order cancelled due to insufficient stock")
                    self.logger.warning(f"Cancelled order: {order}")
    def detect_sales_patterns(self):
        """Analyze sales data to identify patterns and trends"""
        self.logger.info("Detecting sales patterns...")
        
        for part_id, item in self.inventory_manager.showroom_inventory.items():
            # Analyze historical demand
            if item.historical_demand:
                average_demand = sum(item.historical_demand) / len(item.historical_demand)
                current_demand = item.quantity  # Current stock can be used as a proxy for demand
                
                # Simple trend detection
                if current_demand < average_demand * 0.5:
                    self.logger.warning(f"Sales pattern detected: Low demand for {item.part_name}. Consider increasing stock.")
                elif current_demand > average_demand * 1.5:
                    self.logger.info(f"Sales pattern detected: High demand for {item.part_name}. Consider ordering more.")
            else:
                self.logger.info(f"No historical demand data for {item.part_name}.")


class InventorySystem:

    def __init__(self):

        self.inventory_manager = InventoryManager()

        self.showroom_agent = ShowroomAmbientAgent(self.inventory_manager)

        self.warehouse_agent = WarehouseAgent(self.inventory_manager)

        self.supplier_agent = SupplierAgent(self.inventory_manager)

        self.running = False

        self.thread = None

    

    def start(self):

        """Start the inventory system"""

        if self.running:

            logger.warning("System is already running")

            return

        

        self.running = True

        self.showroom_agent.start()

        self.warehouse_agent.start()

        self.supplier_agent.start()

        logger.info("Inventory system started")

    

    def stop(self):

        """Stop the inventory system"""

        if not self.running:

            logger.warning("System is not running")

            return

        

        self.running = False

        self.showroom_agent.stop()

        self.warehouse_agent.stop()

        self.supplier_agent.stop()

        logger.info("Inventory system stopped")

        

        # Save all data before exiting

        self.inventory_manager.save_inventory()

# Main function to run the inventory system
def main():
    logger.info("Starting Ambient Inventory System")
    inventory_manager = InventoryManager()
    showroom_agent = ShowroomAmbientAgent(inventory_manager)
    
    try:
        showroom_agent.start()
        
        # Keep the main thread running
        while True:
            command = input("Enter 'exit' to stop: ")
            if command.lower() == 'exit':
                break
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Stopping system...")
    finally:
        showroom_agent.stop()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()