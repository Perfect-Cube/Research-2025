import os
import csv
import datetime
import uuid
import time
import logging
import threading
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CarPartsInventory")

# Constants
SHOWROOM_CSV = "showroom.csv"
WAREHOUSE_CSV = "warehouse.csv"
SUPPLIER_CSV = "supplier.csv"
ORDER_HISTORY_CSV = "order_history.csv"

# Order status constants
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_IN_TRANSIT = "in_transit"
STATUS_DELIVERED = "delivered"
STATUS_CANCELLED = "cancelled"

# Order type constants
TYPE_SHOWROOM_TO_WAREHOUSE = "showroom_to_warehouse"
TYPE_WAREHOUSE_TO_SUPPLIER = "warehouse_to_supplier"

class InventoryItem:
    def __init__(self, part_id, part_name, category, quantity, min_threshold=0, max_capacity=100, 
                price=0.0, lead_time_days=0, available_quantity=0):
        self.part_id = part_id
        self.part_name = part_name
        self.category = category
        self.quantity = int(quantity)
        self.min_threshold = int(min_threshold) if min_threshold else 0
        self.max_capacity = int(max_capacity) if max_capacity else 100
        self.price = float(price) if price else 0.0
        self.lead_time_days = int(lead_time_days) if lead_time_days else 0
        self.available_quantity = int(available_quantity) if available_quantity else 0

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
            "available_quantity": self.available_quantity
        }

class Order:
    def __init__(self, order_id, part_id, quantity, order_type, status=STATUS_PENDING, 
                order_date=None, expected_delivery=None):
        self.order_id = order_id
        self.part_id = part_id
        self.quantity = int(quantity)
        self.order_type = order_type
        self.status = status
        self.order_date = order_date or datetime.datetime.now().strftime("%Y-%m-%d")
        self.expected_delivery = expected_delivery

    def __str__(self):
        return f"Order {self.order_id}: {self.quantity} units of {self.part_id}, Status: {self.status}"
    
    def to_dict(self):
        return {
            "order_id": self.order_id,
            "part_id": self.part_id,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status,
            "order_date": self.order_date,
            "expected_delivery": self.expected_delivery
        }

class InventoryManager:
    def __init__(self, showroom_file=SHOWROOM_CSV, warehouse_file=WAREHOUSE_CSV, 
                supplier_file=SUPPLIER_CSV, order_history_file=ORDER_HISTORY_CSV):
        self.showroom_file = showroom_file
        self.warehouse_file = warehouse_file
        self.supplier_file = supplier_file
        self.order_history_file = order_history_file
        
        # Load inventory data
        self.showroom_inventory = self._load_inventory(showroom_file)
        self.warehouse_inventory = self._load_inventory(warehouse_file)
        self.supplier_inventory = self._load_inventory(supplier_file, is_supplier=True)
        self.order_history = self._load_orders(order_history_file)

    def _load_inventory(self, filename, is_supplier=False):
        inventory = {}
        if not os.path.exists(filename):
            logger.warning(f"File {filename} not found. Creating an empty inventory.")
            return inventory

        with open(filename, 'r',encoding='utf-8-sig',newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            print(f"CSV headers: {reader.fieldnames}")
            for row in reader:
                if is_supplier:
                    item = InventoryItem(
                        row["part_id"], row["part_name"], row["category"],
                        0, 0, 0, row["price"], row["lead_time_days"], row["available_quantity"]
                    )
                else:
                    item = InventoryItem(
                        row["part_id"], row["part_name"], row["category"],
                        row["quantity"], row["min_threshold"], row["max_capacity"]
                    )
                inventory[row["part_id"]] = item
        return inventory

    def _load_orders(self, filename):
        orders = []
        if not os.path.exists(filename):
            logger.warning(f"File {filename} not found. Creating an empty order history.")
            return orders

        with open(filename, 'r', encoding='latin-1',newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                order = Order(
                    row["order_id"], row["part_id"], row["quantity"],
                    row["order_type"], row["status"], row["order_date"],
                    row["expected_delivery"]
                )
                orders.append(order)
        return orders

    def save_inventory(self):
        """Save all inventory data to CSV files"""
        self._save_inventory(self.showroom_file, self.showroom_inventory)
        self._save_inventory(self.warehouse_file, self.warehouse_inventory)
        self._save_supplier_inventory(self.supplier_file, self.supplier_inventory)
        self._save_orders(self.order_history_file, self.order_history)

    def _save_inventory(self, filename, inventory):
        fieldnames = ["part_id", "part_name", "category", "quantity", "min_threshold", "max_capacity"]
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

    def _save_supplier_inventory(self, filename, inventory):
        fieldnames = ["part_id", "part_name", "category", "available_quantity", "price", "lead_time_days"]
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
                    "lead_time_days": item.lead_time_days
                })

    def _save_orders(self, filename, orders):
        fieldnames = ["order_id", "part_id", "quantity", "order_type", "status", "order_date", "expected_delivery"]
        with open(filename, 'w',newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for order in orders:
                writer.writerow(order.to_dict())

    def generate_order_id(self):
        """Generate a unique order ID"""
        prefix = "ORD"
        existing_ids = [order.order_id for order in self.order_history]
        counter = 1
        while True:
            new_id = f"{prefix}{counter:03d}"
            if new_id not in existing_ids:
                return new_id
            counter += 1

class ShowroomAgent:
    def __init__(self, inventory_manager):
        self.inventory_manager = inventory_manager
        self.logger = logging.getLogger("ShowroomAgent")

    def check_inventory(self):
        """Check showroom inventory for items below threshold"""
        self.logger.info("Checking showroom inventory...")
        orders = []
        
        # Create a set of part_ids that already have active orders
        active_order_parts = set()
        for existing_order in self.inventory_manager.order_history:
            if (existing_order.order_type == TYPE_SHOWROOM_TO_WAREHOUSE and 
                existing_order.status in [STATUS_PENDING, STATUS_APPROVED, STATUS_IN_TRANSIT]):
                active_order_parts.add(existing_order.part_id)
        
        for part_id, item in self.inventory_manager.showroom_inventory.items():
            # Skip if there's already an active order for this part
            if part_id in active_order_parts:
                self.logger.info(f"Skipping order creation for {part_id}: active order already exists")
                continue
                
            if item.quantity < item.min_threshold:
                # Calculate how many to order to reach optimal level (75% of max capacity)
                optimal_level = int(item.max_capacity * 0.75)
                quantity_to_order = optimal_level - item.quantity
                
                # Check if warehouse has enough stock
                if (part_id in self.inventory_manager.warehouse_inventory and 
                    self.inventory_manager.warehouse_inventory[part_id].quantity >= quantity_to_order):
                    
                    # Create order from showroom to warehouse
                    order_id = self.inventory_manager.generate_order_id()
                    order = Order(
                        order_id=order_id,
                        part_id=part_id,
                        quantity=quantity_to_order,
                        order_type=TYPE_SHOWROOM_TO_WAREHOUSE,
                        status=STATUS_PENDING,
                        order_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                        expected_delivery=(datetime.datetime.now() + 
                                        datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                    )
                    orders.append(order)
                    self.logger.info(f"Created order: {order}")
        
        return orders

    def process_approved_orders(self):
        """Process orders that have been approved"""
        self.logger.info("Processing approved orders for showroom...")
        for order in self.inventory_manager.order_history:
            if (order.status == STATUS_APPROVED and 
                order.order_type == TYPE_SHOWROOM_TO_WAREHOUSE):
                
                # Update order status to in_transit
                order.status = STATUS_IN_TRANSIT
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
                order.status = STATUS_DELIVERED
                self.logger.info(f"Order delivered: {order}")
                return True
            else:
                self.logger.warning(f"Cannot deliver order {order.order_id}: Insufficient inventory")
                return False
        return False

class WarehouseAgent:
    def __init__(self, inventory_manager):
        self.inventory_manager = inventory_manager
        self.logger = logging.getLogger("WarehouseAgent")
        
    def check_inventory(self):
        """Check warehouse inventory for items below threshold"""
        self.logger.info("Checking warehouse inventory...")
        orders = []
        
        # Create a set of part_ids that already have active orders
        active_order_parts = set()
        for existing_order in self.inventory_manager.order_history:
            if (existing_order.order_type == TYPE_WAREHOUSE_TO_SUPPLIER and 
                existing_order.status in [STATUS_PENDING, STATUS_APPROVED, STATUS_IN_TRANSIT]):
                active_order_parts.add(existing_order.part_id)
        
        for part_id, item in self.inventory_manager.warehouse_inventory.items():
            # Skip if there's already an active order for this part
            if part_id in active_order_parts:
                self.logger.info(f"Skipping order creation for {part_id}: active order already exists")
                continue
            if item.quantity < item.min_threshold:
                # Calculate how many to order to reach optimal level (75% of max capacity)
                optimal_level = int(item.max_capacity * 0.75)
                quantity_to_order = optimal_level - item.quantity
                
                # Check if supplier has enough stock
                if (part_id in self.inventory_manager.supplier_inventory and 
                    self.inventory_manager.supplier_inventory[part_id].available_quantity >= quantity_to_order):
                    
                    # Create order from warehouse to supplier
                    order_id = self.inventory_manager.generate_order_id()
                    supplier_item = self.inventory_manager.supplier_inventory[part_id]
                    
                    order = Order(
                        order_id=order_id,
                        part_id=part_id,
                        quantity=quantity_to_order,
                        order_type=TYPE_WAREHOUSE_TO_SUPPLIER,
                        status=STATUS_PENDING,
                        order_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                        expected_delivery=(datetime.datetime.now() + 
                                          datetime.timedelta(days=supplier_item.lead_time_days)).strftime("%Y-%m-%d")
                    )
                    orders.append(order)
                    self.logger.info(f"Created order: {order}")
        
        return orders
    
    def review_showroom_orders(self):
        """Review and approve/reject showroom orders"""
        self.logger.info("Reviewing showroom orders...")
        for order in self.inventory_manager.order_history:
            if (order.status == STATUS_PENDING and 
                order.order_type == TYPE_SHOWROOM_TO_WAREHOUSE):
                
                warehouse_item = self.inventory_manager.warehouse_inventory.get(order.part_id)
                
                if warehouse_item and warehouse_item.quantity >= order.quantity:
                    # Approve the order
                    order.status = STATUS_APPROVED
                    self.logger.info(f"Approved order: {order}")
                else:
                    # Order to supplier if warehouse can't fulfill
                    self.logger.info(f"Warehouse cannot fulfill order {order.order_id}. Ordering from supplier...")
                    self.create_supplier_order(order.part_id, order.quantity)

    def process_approved_orders(self):
        """Process orders that have been approved"""
        self.logger.info("Processing approved orders for warehouse...")
        for order in self.inventory_manager.order_history:
            if (order.status == STATUS_APPROVED and 
                order.order_type == TYPE_WAREHOUSE_TO_SUPPLIER):
                
                # Update order status to in_transit
                order.status = STATUS_IN_TRANSIT
                self.logger.info(f"Updated order status to in_transit: {order}")
                
                # Simulate delivery time based on supplier lead time
                order_date = datetime.datetime.strptime(order.order_date, "%Y-%m-%d")
                expected_delivery = datetime.datetime.strptime(order.expected_delivery, "%Y-%m-%d")
                
                if datetime.datetime.now() >= expected_delivery:
                    self.deliver_order(order)
    
    def deliver_order(self, order):
        """Deliver an order to the warehouse"""
        if order.order_type == TYPE_WAREHOUSE_TO_SUPPLIER and order.status == STATUS_IN_TRANSIT:
            # Update inventory
            warehouse_item = self.inventory_manager.warehouse_inventory.get(order.part_id)
            supplier_item = self.inventory_manager.supplier_inventory.get(order.part_id)
            
            if warehouse_item and supplier_item and supplier_item.available_quantity >= order.quantity:
                # Transfer from supplier to warehouse
                supplier_item.available_quantity -= order.quantity
                warehouse_item.quantity += order.quantity
                
                # Update order status
                order.status = STATUS_DELIVERED
                self.logger.info(f"Order delivered: {order}")
                return True
            else:
                self.logger.warning(f"Cannot deliver order {order.order_id}: Insufficient inventory at supplier")
                return False
        return False
    
    def create_supplier_order(self, part_id, quantity):
        """Create an order from warehouse to supplier"""
        supplier_item = self.inventory_manager.supplier_inventory.get(part_id)
        if supplier_item and supplier_item.available_quantity >= quantity:
            order_id = self.inventory_manager.generate_order_id()
            order = Order(
                order_id=order_id,
                part_id=part_id,
                quantity=quantity,
                order_type=TYPE_WAREHOUSE_TO_SUPPLIER,
                status=STATUS_PENDING,
                order_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                expected_delivery=(datetime.datetime.now() + 
                                  datetime.timedelta(days=supplier_item.lead_time_days)).strftime("%Y-%m-%d")
            )
            self.inventory_manager.order_history.append(order)
            self.logger.info(f"Created supplier order: {order}")
            return order
        else:
            self.logger.warning(f"Cannot create supplier order for {part_id}: Insufficient inventory at supplier")
            return None

class SupplierAgent:
    def __init__(self, inventory_manager):
        self.inventory_manager = inventory_manager
        self.logger = logging.getLogger("SupplierAgent")
        
    def review_warehouse_orders(self):
        """Review and approve/reject warehouse orders"""
        self.logger.info("Reviewing warehouse orders...")
        for order in self.inventory_manager.order_history:
            if (order.status == STATUS_PENDING and 
                order.order_type == TYPE_WAREHOUSE_TO_SUPPLIER):
                
                supplier_item = self.inventory_manager.supplier_inventory.get(order.part_id)
                
                if supplier_item and supplier_item.available_quantity >= order.quantity:
                    # Approve the order
                    order.status = STATUS_APPROVED
                    self.logger.info(f"Approved order: {order}")
                else:
                    # Reject the order
                    order.status = STATUS_CANCELLED
                    self.logger.warning(f"Rejected order {order.order_id}: Insufficient inventory")

class InventorySystem:
    def __init__(self):
        self.inventory_manager = InventoryManager()
        self.showroom_agent = ShowroomAgent(self.inventory_manager)
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
        self.thread = threading.Thread(target=self._run_system)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Inventory system started")
    
    def stop(self):
        """Stop the inventory system"""
        if not self.running:
            logger.warning("System is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("Inventory system stopped")
        
        # Save all data before exiting
        self.inventory_manager.save_inventory()
    
    def _run_system(self):
        """Main loop for the inventory system"""
        while self.running:
            try:
                # Check inventory levels and create orders
                showroom_orders = self.showroom_agent.check_inventory()
                warehouse_orders = self.warehouse_agent.check_inventory()
                
                # Add new orders to the history
                self.inventory_manager.order_history.extend(showroom_orders + warehouse_orders)
                
                # Review and approve/reject orders
                self.warehouse_agent.review_showroom_orders()
                self.supplier_agent.review_warehouse_orders()
                
                # Process approved orders
                self.showroom_agent.process_approved_orders()
                self.warehouse_agent.process_approved_orders()
                
                # Save inventory data
                self.inventory_manager.save_inventory()
                
                # Print current inventory status
                self._print_inventory_status()
                
                # Wait for next cycle
                time.sleep(5)  # Check every 5 seconds (adjust as needed)
            except Exception as e:
                logger.error(f"Error in system loop: {str(e)}")
                time.sleep(10)  # Wait longer if there's an error
    
    def _print_inventory_status(self):
        """Print current inventory status"""
        logger.info("\n===== CURRENT INVENTORY STATUS =====")
        logger.info("SHOWROOM INVENTORY:")
        for item in self.inventory_manager.showroom_inventory.values():
            logger.info(f"  {item}")
        
        logger.info("\nWAREHOUSE INVENTORY:")
        for item in self.inventory_manager.warehouse_inventory.values():
            logger.info(f"  {item}")
        
        logger.info("\nRECENT ORDERS:")
        recent_orders = sorted(self.inventory_manager.order_history, 
                             key=lambda o: o.order_date, reverse=True)[:5]
        for order in recent_orders:
            logger.info(f"  {order}")
        logger.info("=====================================\n")

def main():
    logger.info("Starting Car Parts Inventory System")
    system = InventorySystem()
    
    try:
        system.start()
        
        # Keep the main thread running
        while True:
            command = input("Enter 'exit' to stop: ")
            if command.lower() == 'exit':
                break
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Stopping system...")
    finally:
        system.stop()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()