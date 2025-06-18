Of course. This is an excellent use case for Autogen's flexible multi-agent architecture. We will build the exact system you described, using Groq for high-speed LLM inference and avoiding the `UserProxyAgent` for a fully automated, programmatic workflow.

Here is a complete, runnable solution.

### **Prerequisites**

1.  **Install necessary libraries:**
    ```bash
    pip install "pyautogen>=0.2.20" groq python-dotenv duckduckgo-search
    ```

2.  **Set up Groq API Key:**
    Create a file named `.env` in the same directory as your script and add your Groq API key to it:
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ```

---

### **The Code: Multi-Agent System with Orchestration**

This script sets up the entire environment. I have added extensive comments to explain each part of the process, from agent definition to the conditional logic flow.

```python
import os
import autogen
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# 1. --- SETUP ---
# Load API keys from .env file
load_dotenv()

# Check if the API key is loaded
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

# Configuration for Groq LLM
config_list_groq = [
    {
        "model": "llama3-70b-8192", # Or "llama3-8b-8192", "mixtral-8x7b-32768"
        "api_key": os.getenv("GROQ_API_KEY"),
        "base_url": "https://api.groq.com/openai/v1"
    }
]

llm_config_groq = {
    "config_list": config_list_groq,
    "cache_seed": 42, # Use a seed for reproducibility
}


# 2. --- TOOL DEFINITION ---
# We define a simple Python function for the Search Agent to use.
# The docstring is crucial as it tells the Planner agent what the tool does.

def search_duckduckgo(query: str) -> str:
    """
    Searches the web using DuckDuckGo for a given query.

    Args:
        query (str): The search query.

    Returns:
        str: A string containing the search results, formatted.
    """
    print(f"\n🤖 Executing Search: '{query}'")
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return f"Search results for '{query}':\n\n" + "\n".join(str(res) for res in results)


# 3. --- AGENT DEFINITIONS ---

# Orchestrator Agent: The first point of contact. It decides if a query is simple or complex.
orchestrator = autogen.ConversableAgent(
    name="Orchestrator",
    system_message="""You are the Orchestrator. Your job is to analyze the user's query and determine its complexity.
    - If the query is simple, can be answered directly without needing external information, planning, or comparison (e.g., "Hello", "What is 2+2?", "Summarize this text: ..."), respond with ONLY the word "SIMPLE".
    - If the query is non-trivial and requires web searches, data comparison, multi-step reasoning, or involves multiple topics (e.g., "Compare Llama 3 and GPT-4", "What are the main attractions in Paris and how much do they cost?"), respond with ONLY the word "COMPLEX".
    Do not add any other words, explanations, or punctuation.""",
    llm_config=llm_config_groq,
    human_input_mode="NEVER",
)

# Planner Agent: For complex tasks. It creates a plan and uses tools/other agents.
planner = autogen.ConversableAgent(
    name="Planner",
    system_message="""You are the Planner. You receive complex user queries.
    Your goal is to create a step-by-step plan to resolve the query.
    You can use tools by calling their functions and you can delegate tasks to other agents.
    You have access to a 'search_duckduckgo' tool for fetching information.
    You also work with a 'Comparison_Agent' for comparing data.
    
    Execution Flow:
    1.  Break down the user's query into smaller, manageable steps.
    2.  For each step, decide if you need to use the `search_duckduckgo` tool.
    3.  Execute the search and gather all necessary information.
    4.  If the query requires comparison, formulate a clear request to the 'Comparison_Agent', providing it with the data you gathered.
    5.  Once all steps are complete and you have the final answer from yourself or the Comparison_Agent, compile a comprehensive final response.
    6.  Conclude your work by sending this final compiled response to the 'Output_Agent'. Start your message with "FINAL ANSWER:" to signify completion.
    """,
    llm_config=llm_config_groq,
    human_input_mode="NEVER",
)

# Search Agent: A specialized agent that only executes the search tool.
# This agent doesn't need an LLM, it just needs to be an executor for the function.
search_agent = autogen.ConversableAgent(
    name="Search_Agent",
    llm_config=False,  # No LLM needed, it's just an executor
    human_input_mode="NEVER",
)

# Comparison Agent: Specialized in comparing data.
comparison_agent = autogen.ConversableAgent(
    name="Comparison_Agent",
    system_message="""You are the Comparison Agent. You receive data and a request to compare items.
    Analyze the provided data and create a clear, structured comparison.
    This could be a markdown table, a pros-and-cons list, or a segmented analysis.
    Return only the structured comparison. Do not add conversational fluff.""",
    llm_config=llm_config_groq,
    human_input_mode="NEVER",
)

# Output Agent: The final agent that presents the response to the user.
output_agent = autogen.ConversableAgent(
    name="Output_Agent",
    system_message="""You are the Output Agent. You receive the final, complete answer.
    Your job is to present this information clearly and concisely to the end-user.
    Do not add any extra commentary, analysis, or conversational phrases.
    Just format and display the final answer provided to you.""",
    llm_config=llm_config_groq,
    human_input_mode="NEVER",
)


# 4. --- REGISTERING THE TOOL (FUNCTION CALLING) ---
# We register the search function with the Planner (caller) and the Search Agent (executor).
# This allows the Planner to "call" the function, which is then "executed" by the Search Agent.
autogen.register_function(
    search_duckduckgo,
    caller=planner,
    executor=search_agent,
    name="search_duckduckgo",
    description="A tool to search the web using DuckDuckGo for a given query.",
)


# 5. --- THE ORCHESTRATION FLOW ---

def run_conversation(query: str):
    """
    This function orchestrates the entire agent workflow based on the query.
    """
    print(f"*****\nQuery: {query}\n*****\n")

    # A temporary agent to initiate chats and capture responses without a UserProxy
    initiator = autogen.ConversableAgent("initiator", llm_config=False, human_input_mode="NEVER")

    # Step 1: Orchestrator decides the path
    print("--- Orchestrator is deciding the path ---")
    initiator.initiate_chat(orchestrator, message=query, max_turns=1, silent=False)
    decision = initiator.last_message(orchestrator)["content"]
    print(f"Orchestrator decision: {decision}")

    # Step 2: Conditional execution based on the decision
    if "SIMPLE" in decision.upper():
        # --- SIMPLE PATH ---
        print("\n--- Path: SIMPLE ---")
        print("--- Handing over to Output Agent directly ---")
        
        # We craft a message for the output agent to summarize the initial query
        simple_task = f"Please provide a direct answer to the following user query: '{query}'"
        
        initiator.initiate_chat(output_agent, message=simple_task, max_turns=1)
        final_response = initiator.last_message(output_agent)["content"]

    elif "COMPLEX" in decision.upper():
        # --- COMPLEX PATH ---
        print("\n--- Path: COMPLEX ---")
        print("--- Initiating multi-agent collaboration ---")

        # Set up the Group Chat for the complex workflow
        # The Output_Agent is NOT in this group. It's the final step.
        group_agents = [planner, search_agent, comparison_agent]
        group_chat = autogen.GroupChat(agents=group_agents, messages=[], max_round=15)
        
        # The manager for the group chat
        manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config_groq,
            # The manager's role is to facilitate the conversation between the other agents
            system_message="You are a helpful AI assistant. You manage the conversation between the Planner, Search_Agent, and Comparison_Agent. Let the Planner lead the conversation. Conclude when the Planner signals the 'FINAL ANSWER:'."
        )

        # Start the conversation with the Planner as the entry point
        initiator.initiate_chat(manager, message=query)
        
        # The final answer is expected to be the last message from the chat
        final_summary = initiator.last_message(manager)["content"]
        
        # Clean up the final summary if it contains termination signals
        if "FINAL ANSWER:" in final_summary:
            final_summary = final_summary.replace("FINAL ANSWER:", "").strip()

        print("\n--- Multi-agent collaboration complete ---")
        print("--- Handing over final summary to Output Agent ---")

        initiator.initiate_chat(output_agent, message=final_summary, max_turns=1)
        final_response = initiator.last_message(output_agent)["content"]
    
    else:
        final_response = "The orchestrator failed to classify the query. Please rephrase."

    # Step 3: Print the final, clean output
    print("\n\n" + "="*50)
    print("                FINAL RESPONSE")
    print("="*50)
    print(final_response)
    print("="*50 + "\n")


# --- EXAMPLES ---

# Example 1: A simple query that should go directly to the Output Agent
simple_query = "What is the capital of France?"
run_conversation(simple_query)

# Example 2: A complex query that requires the full multi-agent workflow
complex_query = "Compare the key features of the Llama 3 and Mistral Large language models. Present the comparison in a markdown table."
run_conversation(complex_query)

# Example 3: Another complex query
complex_query_2 = "What are the top 3 selling electric cars in 2023, and what are their battery ranges?"
run_conversation(complex_query_2)
```

### **How It Works: A Breakdown**

1.  **Setup & Config:** We load the Groq API key and create a standard `llm_config` that all LLM-powered agents will use.

2.  **Tool Definition (`search_duckduckgo`):** This is a standard Python function. The crucial part is its docstring, which Autogen's LLM agents read to understand what the function does, what arguments it takes, and what it returns.

3.  **Agent Definitions:**
    *   **Orchestrator:** Its `system_message` is very strict. It forces the LLM to output *only* "SIMPLE" or "COMPLEX", making it a reliable binary classifier for our workflow.
    *   **Planner:** This is the core of the complex workflow. Its prompt tells it to create plans, use the `search_duckduckgo` tool, delegate to the `Comparison_Agent`, and signal completion with "FINAL ANSWER:".
    *   **Search\_Agent:** A "dummy" agent. It has `llm_config=False` because it doesn't need to think; its only purpose is to be the `executor` for the `search_duckduckgo` function. This is a clean way to isolate tool execution.
    *   **Comparison\_Agent:** A specialist agent. Its prompt directs it to focus solely on creating structured comparisons from the data it's given.
    *   **Output\_Agent:** The final "voice." It takes the finished product and formats it for the user, ensuring a clean and consistent output style.

4.  **Registering the Tool:** The line `autogen.register_function(...)` is key. It links the `search_duckduckgo` function to the agents. We specify that the `planner` is the one who can *call* it, and the `search_agent` is the one who *executes* it.

5.  **The Orchestration Flow (`run_conversation` function):**
    *   This function replaces the need for an interactive `UserProxyAgent`.
    *   It first sends the query to the `Orchestrator`.
    *   It checks the `Orchestrator`'s one-word response.
    *   **Simple Path:** If the response is "SIMPLE", it constructs a new, simple message and sends it directly to the `Output_Agent` to get a final answer.
    *   **Complex Path:** If the response is "COMPLEX", it sets up a `GroupChat` containing the `Planner`, `Search_Agent`, and `Comparison_Agent`. The `Planner` is tasked with initiating and driving this chat. The `GroupChatManager` oversees their interaction. Once the `Planner` signals it's done, we extract the final summary.
    *   **Final Handoff:** In the complex path, the final summary from the group chat is then passed to the `Output_Agent` for clean presentation, just like in the simple path. This ensures the user always gets the final result from the same "voice."

This setup provides a robust, programmatic, and efficient multi-agent system that intelligently routes tasks based on their complexity, all powered by the speed of Groq.
