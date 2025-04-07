```
AmbientInventorySystem/
|
|-- README.txt                     # Explains the overall project structure
|
|-- .env                           # Stores environment variables (like GEMINI_API_KEY) - *Not part of the script, but essential external config*
|
|-- data/                          # Directory holding persistent data files
|   |-- showroom1.csv              # Input/Output: Current inventory state for the showroom
|   |-- warehouse1.csv             # Input/Output: Current inventory state for the warehouse
|   |-- supplier1.csv              # Input/Output: Supplier's item availability, price, lead times
|   |-- order_history1.csv         # Input/Output: Log of all created orders and their statuses
|   |-- events_log.csv             # Input/Output: Log of significant system events (e.g., low stock)
|   `-- agent_decisions.csv        # Input/Output: Log of decisions made by agents (e.g., creating an order)
|
`-- ambient_inventory_system.py    # The main Python script containing all the logic
    |
    |-- section_Imports/           # Represents the import statements at the top
    |   `-- description.txt        # Purpose: Load necessary libraries (os, csv, datetime, uuid, etc.)
    |
    |-- section_Setup/             # Represents initial configuration steps
    |   |-- load_dotenv.config     # Action: Loads variables from .env
    |   |-- logging.config         # Action: Sets up logging format and level
    |   `-- constants.pyc          # Simulated: Defines constants (CSV filenames, statuses, types)
    |
    |-- section_AI_Config/         # Represents Gemini API configuration
    |   |-- api_key.credential     # Simulated: Holds the API key (loaded from .env)
    |   `-- genai_model.init       # Action: Configures and initializes the Gemini model
    |
    |-- module_DataModels/         # Represents the classes defining data structures
    |   |-- InventoryItem.class    # Definition: Structure for a single inventory part (attributes, to_dict)
    |   |-- Order.class            # Definition: Structure for an inventory transfer order (attributes, status update, to_dict)
    |   |-- Event.class            # Definition: Structure for logging system events (attributes, to_dict)
    |   `-- AgentDecision.class    # Definition: Structure for logging agent decisions (attributes, to_dict)
    |
    |-- module_CoreLogic/          # Represents the central management component
    |   `-- InventoryManager.class # Definition: Handles data loading/saving from/to CSVs, provides thread-safe data access, generates IDs
    |       |-- method_load_inventory.pyc  # Function: Reads inventory CSVs into memory
    |       |-- method_load_orders.pyc     # Function: Reads order history CSV into memory
    |       |-- method_load_events.pyc     # Function: Reads event log CSV into memory
    |       |-- method_load_decisions.pyc  # Function: Reads decision log CSV into memory
    |       |-- method_save_inventory.pyc  # Function: Writes all data back to respective CSVs
    |       |-- method_add_event.pyc       # Function: Adds a new event to the log
    |       |-- method_add_decision.pyc    # Function: Adds a new decision to the log
    |       `-- method_generate_id.pyc     # Function: Creates unique IDs
    |
    |-- module_Agents/             # Represents the autonomous agent components
    |   |-- AmbientAgent_Base.class # Definition: Abstract base class for all agents (start/stop logic, threading, AI interaction, logging decisions)
    |   |   `-- method_ask_ai.pyc     # Function: Sends prompt to Gemini API, gets response
    |   |
    |   |-- ShowroomAmbientAgent.class # Definition: Agent for showroom operations
    |   |   |-- method_run_agent.pyc  # Function: Main loop (checks inventory, processes orders)
    |   |   |-- method_check_inventory.pyc # Function: Monitors stock, creates orders to warehouse if low (uses AI optionally)
    |   |   `-- method_process_approved_orders.pyc # Function: Handles delivery simulation from warehouse
    |   |
    |   |-- WarehouseAgent.class    # Definition: Agent for warehouse operations
    |   |   |-- method_run_agent.pyc  # Function: Main loop (checks inventory, reviews showroom orders, processes orders)
    |   |   |-- method_check_inventory.pyc # Function: Monitors stock, creates orders to supplier if low
    |   |   |-- method_review_showroom_orders.pyc # Function: Approves/cancels orders from showroom based on stock
    |   |   `-- method_process_approved_orders.pyc # Function: Fulfills approved orders to showroom
    |   |
    |   `-- SupplierAgent.class     # Definition: Agent simulating supplier actions
    |       |-- method_run_agent.pyc  # Function: Main loop (reviews warehouse orders)
    |       `-- method_review_warehouse_orders.pyc # Function: Approves/cancels orders from warehouse based on its available stock
    |
    |-- module_SystemOrchestration/ # Represents the class to manage the whole system (*Note: Currently underutilized*)
    |   `-- InventorySystem.class   # Definition: Intended to hold manager/agents and provide central start/stop
    |       |-- method_start.pyc      # Function: Starts all agents
    |       `-- method_stop.pyc       # Function: Stops all agents, saves data
    |
    `-- section_Execution/         # Represents the main execution block
        |-- main_function.pyc      # Function: `main()` - Initializes manager, starts agent(s), handles user input loop
        `-- entry_point.trigger    # Simulated: `if __name__ == "__main__":` - Calls `main()` when script is run directly
```
