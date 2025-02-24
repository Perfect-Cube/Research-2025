import os
import sqlite3
from groq import Groq

# -----------------------------------------------------------------------------
# Local DB Connection using SQLite: Manage connections and query execution
# -----------------------------------------------------------------------------
class LocalDBConnection:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
    
    def close(self):
        self.conn.close()
    
    def query(self, query, parameters=()):
        cur = self.conn.cursor()
        cur.execute(query, parameters)
        rows = cur.fetchall()
        # Convert rows to a list of dictionaries using column names
        col_names = [description[0] for description in cur.description]
        results = [dict(zip(col_names, row)) for row in rows]
        return results

    def execute(self, query, parameters=()):
        cur = self.conn.cursor()
        cur.execute(query, parameters)
        self.conn.commit()

# -----------------------------------------------------------------------------
# Database Initialization: Create table and insert sample data if needed
# -----------------------------------------------------------------------------
def initialize_db(db_conn):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS material_sampling (
        material TEXT,
        test TEXT,
        result TEXT,
        vehicle TEXT
    )
    """
    db_conn.execute(create_table_query)

    # Check if table is empty; if so, insert sample data.
    check_query = "SELECT COUNT(*) as count FROM material_sampling"
    count = db_conn.query(check_query)[0]['count']
    if count == 0:
        sample_data = [
            ("aluminum", "Hardness Test", "Pass", "Vehicle X"),
            ("aluminum", "Tensile Strength", "Fail", "Vehicle X"),
            ("aluminum", "Corrosion Resistance", "Pass", "Vehicle Y"),
            ("steel", "Hardness Test", "Pass", "Vehicle Z"),
            ("steel", "Impact Test", "Pass", "Vehicle Z")
        ]
        insert_query = "INSERT INTO material_sampling (material, test, result, vehicle) VALUES (?, ?, ?, ?)"
        for record in sample_data:
            db_conn.execute(insert_query, record)
        print("Inserted sample data into the database.")

# -----------------------------------------------------------------------------
# Multi-Hop Retrieval: Retrieve material sampling data from the local DB
# -----------------------------------------------------------------------------
def multi_hop_retrieval(material_name):
    query = """
    SELECT material, test, result, vehicle
    FROM material_sampling
    WHERE material = ?
    """
    results = db_conn.query(query, (material_name,))
    return results

# -----------------------------------------------------------------------------
# Groq API Call using the Groq library: Generate text using the "llama-3.3-70b-versatile" model.
# -----------------------------------------------------------------------------
def call_groq_api(prompt):
    # Retrieve the API key from environment variable or use a default value.
    api_key = os.getenv("GROQ_API_KEY", "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")
    # Create the Groq client with the provided API key.
    client = Groq(api_key=api_key)
    # Prepare the message list for the chat completion.
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    response_text = ""
    for chunk in completion:
        response_text += chunk.choices[0].delta.content or ""
    return response_text

# -----------------------------------------------------------------------------
# GroqAgentModel: A simple wrapper to interface with the Groq API.
# -----------------------------------------------------------------------------
class GroqAgentModel:
    def generate(self, prompt):
        return call_groq_api(prompt)

# -----------------------------------------------------------------------------
# Agent Behavior: Process the user query and generate a response using multi-hop retrieval and Groq.
# -----------------------------------------------------------------------------
def agent_behavior(user_input, model):
    # Extract material name from the query (basic extraction logic).
    if "about" in user_input:
        material_name = user_input.split("about")[-1].strip().split()[0]
    else:
        material_name = user_input.strip().split()[0]
    
    # Retrieve data from the local database.
    retrieval_results = multi_hop_retrieval(material_name)
    if not retrieval_results:
        return f"No data found for material: {material_name}"
    
    # Format the retrieved data into a context string.
    context_lines = []
    for record in retrieval_results:
        line = (f"Material: {record['material']}, "
                f"Test: {record['test']}, "
                f"Result: {record['result']}, "
                f"Vehicle: {record['vehicle']}")
        context_lines.append(line)
    context = "\n".join(context_lines)
    
    # Build the prompt for the Groq API.
    prompt = (
        f"User asked about material '{material_name}'. Here is the relevant sampling data:\n"
        f"{context}\n"
        "Provide a concise summary of the material's performance and test outcomes."
    )
    
    # Generate the response using the Groq model.
    response_text = model.generate(prompt)
    return response_text

# -----------------------------------------------------------------------------
# Main: Execute user query processing using the local database and Groq API.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize local database connection.
    db_conn = LocalDBConnection("materials.db")
    
    # Initialize the database (create table and insert sample data if needed).
    initialize_db(db_conn)
    
    # Set up the Groq model.
    groq_model = GroqAgentModel()
    
    # Example query.
    user_query = "Tell me about aluminum used in Vehicle X."
    response = agent_behavior(user_query, groq_model)
    
    print("Response:")
    print(response)
    
    # Close the local database connection when done.
    db_conn.close()
