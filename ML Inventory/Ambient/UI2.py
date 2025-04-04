import tkinter as tk
from tkinter import ttk, messagebox
import csv
import os
import sys
import logging
from datetime import datetime, timedelta

# Import the inventory system
from inventory_system2 import InventorySystem, Order, InventoryManager, ShowroomAmbientAgent, WarehouseAgent, SupplierAgent

class InventoryUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Parts Inventory Management System")
        self.root.geometry("1200x700")
        
        # Initialize inventory system
        self.inventory_system = InventorySystem()
        
        # Create the UI
        self._create_ui()
        
        # Update timer
        self.update_timer = None
        self.start_update_timer()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _create_ui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.showroom_tab = ttk.Frame(self.notebook)
        self.warehouse_tab = ttk.Frame(self.notebook)
        self.orders_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.showroom_tab, text="Showroom Inventory")
        self.notebook.add(self.warehouse_tab, text="Warehouse Inventory")
        self.notebook.add(self.orders_tab, text="Orders")
        
        # Setup each tab
        self._setup_dashboard()
        self._setup_showroom_tab()
        self._setup_warehouse_tab()
        self._setup_orders_tab()
        
        # Control frame at the bottom
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Buttons
        ttk.Button(control_frame, text="Start System", command=self.start_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop System", command=self.stop_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh Data", command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="System Status: Stopped")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.RIGHT, padx=10)
    
    def _setup_dashboard(self):
        # Create frames
        top_frame = ttk.Frame(self.dashboard_tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Stats frames
        stats_frame = ttk.LabelFrame(top_frame, text="Inventory Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Stats grid
        self.showroom_count_var = tk.StringVar(value="Total Parts: 0")
        self.warehouse_count_var = tk.StringVar(value="Total Parts: 0")
        self.pending_orders_var = tk.StringVar(value="Pending Orders: 0")
        self.in_transit_var = tk.StringVar(value="In Transit: 0")
        
        ttk.Label(stats_frame, text="Showroom:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(stats_frame, textvariable=self.showroom_count_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Warehouse:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(stats_frame, textvariable=self.warehouse_count_var).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Orders:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(stats_frame, textvariable=self.pending_orders_var).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Shipments:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(stats_frame, textvariable=self.in_transit_var).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Recent activities
        activities_frame = ttk.LabelFrame(self.dashboard_tab, text="Recent Activities")
        activities_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Activities treeview
        self.activities_tree = ttk.Treeview(activities_frame, columns=("timestamp", "description"), show="headings")
        self.activities_tree.heading("timestamp", text="Timestamp")
        self.activities_tree.heading("description", text="Activity")
        self.activities_tree.column("timestamp", width=150)
        self.activities_tree.column("description", width=800)
        self.activities_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.activities_tree, orient="vertical", command=self.activities_tree.yview)
        self.activities_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_showroom_tab(self):
        # Create frames
        controls_frame = ttk.Frame(self.showroom_tab)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Search field
        ttk.Label(controls_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.showroom_search_var = tk.StringVar()
        ttk.Entry(controls_frame, textvariable=self.showroom_search_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Search", command=self.search_showroom).pack(side=tk.LEFT, padx=5)
        
        # Category filter
        ttk.Label(controls_frame, text="Category:").pack(side=tk.LEFT, padx=5)
        self.showroom_category_var = tk.StringVar(value="All")
        self.showroom_category_combo = ttk.Combobox(controls_frame, textvariable=self.showroom_category_var, 
                                                   values=["All", "Engine", "Braking", "Suspension", "Electrical", "Exterior"])
        self.showroom_category_combo.pack(side=tk.LEFT, padx=5)
        self.showroom_category_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_showroom())
        
        # Inventory tree
        inventory_frame = ttk.Frame(self.showroom_tab)
        inventory_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.showroom_tree = ttk.Treeview(inventory_frame, 
                                         columns=("id", "name", "category", "quantity", "min", "max", "status"),
                                         show="headings")
        self.showroom_tree.heading("id", text="Part ID")
        self.showroom_tree.heading("name", text="Part Name")
        self.showroom_tree.heading("category", text="Category")
        self.showroom_tree.heading("quantity", text="Quantity")
        self.showroom_tree.heading("min", text="Min Threshold")
        self.showroom_tree.heading("max", text="Max Capacity")
        self.showroom_tree.heading("status", text="Status")
        
        self.showroom_tree.column("id", width=80)
        self.showroom_tree.column("name", width=200)
        self.showroom_tree.column("category", width=100)
        self.showroom_tree.column("quantity", width=80)
        self.showroom_tree.column("min", width=100)
        self.showroom_tree.column("max", width=100)
        self.showroom_tree.column("status", width=100)
        
        self.showroom_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(inventory_frame, orient="vertical", command=self.showroom_tree.yview)
        self.showroom_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        action_frame = ttk.Frame(self.showroom_tab)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(action_frame, text="Order From Warehouse", 
                  command=self.order_from_warehouse).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="View Order History", 
                  command=lambda: self.notebook.select(self.orders_tab)).pack(side=tk.LEFT, padx=5)
    
    def _setup_warehouse_tab(self):
        # Create frames
        controls_frame = ttk.Frame(self.warehouse_tab)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Search field
        ttk.Label(controls_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.warehouse_search_var = tk.StringVar()
        ttk.Entry(controls_frame, textvariable=self.warehouse_search_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Search", command=self.search_warehouse).pack(side=tk.LEFT, padx=5)
        
        # Category filter
        ttk.Label(controls_frame, text="Category:").pack(side=tk.LEFT, padx=5)
        self.warehouse_category_var = tk.StringVar(value="All")
        self.warehouse_category_combo = ttk.Combobox(controls_frame, textvariable=self.warehouse_category_var, 
                                                   values=["All", "Engine", "Braking", "Suspension", "Electrical", "Exterior"])
        self.warehouse_category_combo.pack(side=tk.LEFT, padx=5)
        self.warehouse_category_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_warehouse())
        
        # Inventory tree
        inventory_frame = ttk.Frame(self.warehouse_tab)
        inventory_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.warehouse_tree = ttk.Treeview(inventory_frame, 
                                          columns=("id", "name", "category", "quantity", "min", "max", "status"),
                                          show="headings")
        self.warehouse_tree.heading("id", text="Part ID")
        self.warehouse_tree.heading("name", text="Part Name")
        self.warehouse_tree.heading("category", text="Category")
        self.warehouse_tree.heading("quantity", text="Quantity")
        self.warehouse_tree.heading("min", text="Min Threshold")
        self.warehouse_tree.heading("max", text="Max Capacity")
        self.warehouse_tree.heading("status", text="Status")
        
        self.warehouse_tree.column("id", width=80)
        self.warehouse_tree.column("name", width=200)
        self.warehouse_tree.column("category", width=100)
        self.warehouse_tree.column("quantity", width=80)
        self.warehouse_tree.column("min", width=100)
        self.warehouse_tree.column("max", width=100)
        self.warehouse_tree.column("status", width=100)
        
        self.warehouse_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(inventory_frame, orient="vertical", command=self.warehouse_tree.yview)
        self.warehouse_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        action_frame = ttk.Frame(self.warehouse_tab)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(action_frame, text="Order From Supplier", 
                  command=self.order_from_supplier).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="View Order History", 
                  command=lambda: self.notebook.select(self.orders_tab)).pack(side=tk.LEFT, padx=5)
    
    def _setup_orders_tab(self):
        # Create frames
        controls_frame = ttk.Frame(self.orders_tab)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Filter options
        ttk.Label(controls_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        self.order_status_var = tk.StringVar(value="All")
        self.order_status_combo = ttk.Combobox(controls_frame, textvariable=self.order_status_var, 
                                             values=["All", "pending", "approved", "in_transit", "delivered", "cancelled"])
        self.order_status_combo.pack(side=tk.LEFT, padx=5)
        self.order_status_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_orders())
        
        ttk.Label(controls_frame, text="Order Type:").pack(side=tk.LEFT, padx=5)
        self.order_type_var = tk.StringVar(value="All")
        self.order_type_combo = ttk.Combobox(controls_frame, textvariable=self.order_type_var, 
                                           values=["All", "showroom_to_warehouse", "warehouse_to_supplier"])
        self.order_type_combo.pack(side=tk.LEFT, padx=5)
        self.order_type_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_orders())
        
        # Orders tree
        orders_frame = ttk.Frame(self.orders_tab)
        orders_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.orders_tree = ttk.Treeview(orders_frame, 
                                       columns=("id", "part_id", "quantity", "type", "status", "order_date", "delivery"),
                                       show="headings")
        self.orders_tree.heading("id", text="Order ID")
        self.orders_tree.heading("part_id", text="Part ID")
        self.orders_tree.heading("quantity", text="Quantity")
        self.orders_tree.heading("type", text="Order Type")
        self.orders_tree.heading("status", text="Status")
        self.orders_tree.heading("order_date", text="Order Date")
        self.orders_tree.heading("delivery", text="Expected Delivery")
        
        self.orders_tree.column("id", width=80)
        self.orders_tree.column("part_id", width=80)
        self.orders_tree.column("quantity", width=80)
        self.orders_tree.column("type", width=150)
        self.orders_tree.column("status", width=100)
        self.orders_tree.column("order_date", width=100)
        self.orders_tree.column("delivery", width=120)
        
        self.orders_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(orders_frame, orient="vertical", command=self.orders_tree.yview)
        self.orders_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        action_frame = ttk.Frame(self.orders_tab)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(action_frame, text="View Order Details", 
                  command=self.view_order_details).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Mark as Approved", 
                  command=self.approve_order).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Mark as Delivered", 
                  command=self.deliver_order).pack(side=tk.LEFT, padx=5)
    
    def start_system(self):
        """Start the inventory system"""
        if not self.inventory_system.running:
            self.inventory_system.start()
            self.status_var.set("System Status: Running")
            messagebox.showinfo("System Control", "Inventory system started successfully.")
            self.add_activity("System started")
            self.refresh_data()
    
    def stop_system(self):
        """Stop the inventory system"""
        if self.inventory_system.running:
            self.inventory_system.stop()
            self.status_var.set("System Status: Stopped")
            messagebox.showinfo("System Control", "Inventory system stopped successfully.")
            self.add_activity("System stopped")
    
    def refresh_data(self):
        """Refresh all data displayed in the UI"""
        self.refresh_showroom()
        self.refresh_warehouse()
        self.refresh_orders()
        self.update_statistics()
    
    def refresh_showroom(self):
        """Refresh showroom inventory data"""
        # Clear existing data
        for i in self.showroom_tree.get_children():
            self.showroom_tree.delete(i)
            
        # Get filter values
        search_text = self.showroom_search_var.get().lower()
        category_filter = self.showroom_category_var.get()
        
        # Populate with fresh data
        for part_id, item in self.inventory_system.inventory_manager.showroom_inventory.items():
            # Apply filters
            if search_text and search_text not in item.part_name.lower() and search_text not in part_id.lower():
                continue
                
            if category_filter != "All" and category_filter != item.category:
                continue
            
            # Determine status
            if item.quantity <= item.min_threshold:
                status = "Low"
            elif item.quantity >= item.max_capacity * 0.9:
                status = "Full"
            else:
                status = "OK"
                
            # Insert item
            self.showroom_tree.insert("", "end", values=(
                item.part_id, item.part_name, item.category, item.quantity,
                item.min_threshold, item.max_capacity, status
            ))
    
    def refresh_warehouse(self):
        """Refresh warehouse inventory data"""
        # Clear existing data
        for i in self.warehouse_tree.get_children():
            self.warehouse_tree.delete(i)
            
        # Get filter values
        search_text = self.warehouse_search_var.get().lower()
        category_filter = self.warehouse_category_var.get()
        
        # Populate with fresh data
        for part_id, item in self.inventory_system.inventory_manager.warehouse_inventory.items():
            # Apply filters
            if search_text and search_text not in item.part_name.lower() and search_text not in part_id.lower():
                continue
                
            if category_filter != "All" and category_filter != item.category:
                continue
            
            # Determine status
            if item.quantity <= item.min_threshold:
                status = "Low"
            elif item.quantity >= item.max_capacity * 0.9:
                status = "Full"
            else:
                status = "OK"
                
            # Insert item
            self.warehouse_tree.insert("", "end", values=(
                item.part_id, item.part_name, item.category, item.quantity,
                item.min_threshold, item.max_capacity, status
            ))
    
    def refresh_orders(self):
        """Refresh orders data"""
        # Clear existing data
        for i in self.orders_tree.get_children():
            self.orders_tree.delete(i)
            
        # Get filter values
        status_filter = self.order_status_var.get()
        type_filter = self.order_type_var.get()
        
        # Populate with fresh data
        for order in self.inventory_system.inventory_manager.order_history:
            # Apply filters
            if status_filter != "All" and status_filter != order.status:
                continue
                
            if type_filter != "All" and type_filter != order.order_type:
                continue
            
            # Insert order
            self.orders_tree.insert("", "end", values=(
                order.order_id, order.part_id, order.quantity, order.order_type,
                order.status, order.order_date, order.expected_delivery
            ))
    
    def update_statistics(self):
        """Update statistics displayed on the dashboard"""
        # Calculate showroom stats
        showroom_total = sum(item.quantity for item in self.inventory_system.inventory_manager.showroom_inventory.values())
        self.showroom_count_var.set(f"Total Parts: {showroom_total}")
        
        # Calculate warehouse stats
        warehouse_total = sum(item.quantity for item in self.inventory_system.inventory_manager.warehouse_inventory.values())
        self.warehouse_count_var.set(f"Total Parts: {warehouse_total}")
        
        # Calculate order stats
        pending_orders = sum(1 for order in self.inventory_system.inventory_manager.order_history 
                           if order.status in ["pending", "approved"])
        self.pending_orders_var.set(f"Pending Orders: {pending_orders}")
        
        in_transit = sum(1 for order in self.inventory_system.inventory_manager.order_history 
                        if order.status == "in_transit")
        self.in_transit_var.set(f"In Transit: {in_transit}")
    
    def search_showroom(self):
        """Search showroom inventory"""
        self.refresh_showroom()
        
    def search_warehouse(self):
        """Search warehouse inventory"""
        self.refresh_warehouse()
    
    def order_from_warehouse(self):
        """Create a new order from showroom to warehouse"""
        # Get selected item
        selected_item = self.showroom_tree.selection()
        if not selected_item:
            messagebox.showwarning("Selection Required", "Please select a part to order.")
            return
            
        # Get item details
        item_values = self.showroom_tree.item(selected_item[0], "values")
        part_id = item_values[0]
        
        # Show order dialog
        quantity = self.show_order_dialog(part_id)
        if not quantity:
            return
            
        # Create the order
        showroom_item = self.inventory_system.inventory_manager.showroom_inventory.get(part_id)
        if showroom_item:
            order_id = self.inventory_system.inventory_manager.generate_order_id()
            order = Order(
                order_id=order_id,
                part_id=part_id,
                quantity=quantity,
                order_type="showroom_to_warehouse",
                status="pending",
                order_date=datetime.now().strftime("%Y-%m-%d"),
                expected_delivery=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            )
            self.inventory_system.inventory_manager.order_history.append(order)
            self.inventory_system.inventory_manager.save_inventory()
            
            messagebox.showinfo("Order Created", f"Order {order_id} created successfully.")
            self.add_activity(f"Created order {order_id} for {quantity} units of {part_id}")
            self.refresh_orders()
            self.notebook.select(self.orders_tab)
    
    def order_from_supplier(self):
        """Create a new order from warehouse to supplier"""
        # Get selected item
        selected_item = self.warehouse_tree.selection()
        if not selected_item:
            messagebox.showwarning("Selection Required", "Please select a part to order.")
            return
            
        # Get item details
        item_values = self.warehouse_tree.item(selected_item[0], "values")
        part_id = item_values[0]
        
        # Show order dialog
        quantity = self.show_order_dialog(part_id)
        if not quantity:
            return
            
        # Create the order
        warehouse_item = self.inventory_system.inventory_manager.warehouse_inventory.get(part_id)
        supplier_item = self.inventory_system.inventory_manager.supplier_inventory.get(part_id)
        
        if warehouse_item and supplier_item:
            order_id = self.inventory_system.inventory_manager.generate_order_id()
            order = Order(
                order_id=order_id,
                part_id=part_id,
                quantity=quantity,
                order_type="warehouse_to_supplier",
                status="pending",
                order_date=datetime.now().strftime("%Y-%m-%d"),
                expected_delivery=(datetime.now() + timedelta(days=supplier_item.lead_time_days)).strftime("%Y-%m-%d")
            )
            self.inventory_system.inventory_manager.order_history.append(order)
            self.inventory_system.inventory_manager.save_inventory()
            
            messagebox.showinfo("Order Created", f"Order {order_id} created successfully.")
            self.add_activity(f"Created order {order_id} for {quantity} units of {part_id} from supplier")
            self.refresh_orders()
            self.notebook.select(self.orders_tab)
    
    def show_order_dialog(self, part_id):
        """Show a dialog to input order quantity"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create Order")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create dialog contents
        ttk.Label(dialog, text=f"Order Part: {part_id}").pack(pady=5)
        
        ttk.Label(dialog, text="Quantity:").pack(pady=5)
        quantity_var = tk.StringVar(value="1")
        quantity_entry = ttk.Entry(dialog, textvariable=quantity_var, width=10)
        quantity_entry.pack(pady=5)
        quantity_entry.focus_set()
        
        result = [None]  # Use a list to store the result
        
        def on_ok():
            try:
                quantity = int(quantity_var.get())
                if quantity <= 0:
                    messagebox.showwarning("Invalid Input", "Quantity must be greater than zero.", parent=dialog)
                    return
                result[0] = quantity
                dialog.destroy()
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid number.", parent=dialog)
        
        def on_cancel():
            dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Wait for the dialog to close
        self.root.wait_window(dialog)
        return result[0]
    
    def view_order_details(self):
        """View details of the selected order"""
        # Get selected item
        selected_item = self.orders_tree.selection()
        if not selected_item:
            messagebox.showwarning("Selection Required", "Please select an order to view.")
            return
            
        # Get order details
        item_values = self.orders_tree.item(selected_item[0], "values")
        order_id = item_values[0]
        
        # Find the order
        order = next((o for o in self.inventory_system.inventory_manager.order_history if o.order_id == order_id), None)
        if not order:
            messagebox.showwarning("Error", "Order not found.")
            return
            
        # Find part information
        part = None
        if order.order_type == "showroom_to_warehouse":
            part = self.inventory_system.inventory_manager.showroom_inventory.get(order.part_id)
        elif order.order_type == "warehouse_to_supplier":
            part = self.inventory_system.inventory_manager.warehouse_inventory.get(order.part_id)
            
        # Show details dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Order Details")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create dialog contents
        ttk.Label(dialog, text="Order Details", font=("", 12, "bold")).pack(pady=10)
        
        details_frame = ttk.Frame(dialog)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        ttk.Label(details_frame, text="Order ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(details_frame, text=order.order_id).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(details_frame, text="Part ID:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(details_frame, text=order.part_id).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        if part:
            ttk.Label(details_frame, text="Part Name:").grid(row=2, column=0, sticky=tk.W, pady=2)
            ttk.Label(details_frame, text=part.part_name).grid(row=2, column=1, sticky=tk.W, pady=2)
            
            ttk.Label(details_frame, text="Category:").grid(row=3, column=0, sticky=tk.W, pady=2)
            ttk.Label(details_frame, text=part.category).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(details_frame, text="Quantity:").grid(row=4, column=0,  sticky=tk.W, pady=2)
        ttk.Label(details_frame, text=order.quantity).grid(row=4, column=1, sticky=tk.W, pady=2)

        ttk.Label(details_frame, text="Order Type:").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Label(details_frame, text=order.order_type).grid(row=5, column=1, sticky=tk.W, pady=2)

        ttk.Label(details_frame, text="Status:").grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Label(details_frame, text=order.status).grid(row=6, column=1, sticky=tk.W, pady=2)

        ttk.Label(details_frame, text="Order Date:").grid(row=7, column=0, sticky=tk.W, pady=2)
        ttk.Label(details_frame, text=order.order_date).grid(row=7, column=1, sticky=tk.W, pady=2)

        ttk.Label(details_frame, text="Expected Delivery:").grid(row=8, column=0, sticky=tk.W, pady=2)
        ttk.Label(details_frame, text=order.expected_delivery).grid(row=8, column=1, sticky=tk.W, pady=2)

        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

    def approve_order(self):
        """Approve the selected order"""
        selected_item = self.orders_tree.selection()
        if not selected_item:
            messagebox.showwarning("Selection Required", "Please select an order to approve.")
            return

        item_values = self.orders_tree.item(selected_item[0], "values")
        order_id = item_values[0]

        order = next((o for o in self.inventory_system.inventory_manager.order_history if o.order_id == order_id), None)
        if order:
            if order.status == "pending":
                order.status = "approved"
                messagebox.showinfo("Order Approved", f"Order {order_id} has been approved.")
                self.add_activity(f"Order {order_id} approved.")
                self.refresh_orders()
            else:
                messagebox.showwarning("Invalid Action", "Only pending orders can be approved.")
        else:
            messagebox.showwarning("Error", "Order not found.")

    def deliver_order(self):
        """Mark the selected order as delivered"""
        selected_item = self.orders_tree.selection()
        if not selected_item:
            messagebox.showwarning("Selection Required", "Please select an order to mark as delivered.")
            return

        item_values = self.orders_tree.item(selected_item[0], "values")
        order_id = item_values[0]

        order = next((o for o in self.inventory_system.inventory_manager.order_history if o.order_id == order_id), None)
        if order:
            if order.status == "in_transit":
                order.status = "delivered"
                messagebox.showinfo("Order Delivered", f"Order {order_id} has been marked as delivered.")
                self.add_activity(f"Order {order_id} delivered.")
                self.refresh_orders()
            else:
                messagebox.showwarning("Invalid Action", "Only orders in transit can be marked as delivered.")
        else:
            messagebox.showwarning("Error", "Order not found.")

    def add_activity(self, description):
        """Add an activity to the recent activities log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.activities_tree.insert("", "end", values=(timestamp, description))

    def start_update_timer(self):
        """Start a timer to refresh data periodically"""
        self.refresh_data()
        self.update_timer = self.root.after(5000, self.start_update_timer)  # Refresh every 5 seconds

    def on_closing(self):
        """Handle window close event"""
        if self.inventory_system.running:
            self.inventory_system.stop()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = InventoryUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()