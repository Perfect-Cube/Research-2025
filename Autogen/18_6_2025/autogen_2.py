import os
import autogen
from duckduckgo_search import DDGS

# --- Configuration ---
# IMPORTANT: Replace the placeholder with your actual Groq API key.
GROQ_API_KEY = "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" # <--- REPLACE WITH YOUR KEY

# The core list of LLM configurations. This is still needed.
config_list_groq = [
    {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct", # Or "llama3-8b-8192", "mixtral-8x7b-32768"
        "api_key": "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",
        "api_type": "groq"
    }
]

# The llm_config_groq variable has been removed.
# The configuration will be passed directly to the agents.


# --- TOOL DEFINITION ---
# The tool for the Search Agent with added error handling.
def search_duckduckgo(query: str) -> str:
    """
    Searches the web using DuckDuckGo for a given query.

    Args:
        query (str): The search query.

    Returns:
        str: A string containing the search results, formatted.
    """
    print(f"\n🤖 Executing Search: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            if not results:
                return f"No results found for '{query}'."
            return f"Search results for '{query}':\n\n" + "\n".join(str(res) for res in results)
    except Exception as e:
        return f"An error occurred during the search: {e}"


# --- AGENT DEFINITIONS ---
# The llm_config dictionary is now defined directly inside each agent.

# Orchestrator Agent: Decides the workflow path.
orchestrator = autogen.ConversableAgent(
    name="Orchestrator",
    system_message="""You are the Orchestrator. Your job is to analyze the user's query and determine its complexity.
    - If the query is simple, respond with ONLY the word "SIMPLE".
    - If the query is non-trivial and requires web searches or multi-step reasoning, respond with ONLY the word "COMPLEX".
    Do not add any other words, explanations, or punctuation.""",
    llm_config={
        "config_list": config_list_groq,
        "cache_seed": 42,
    },
    human_input_mode="NEVER",
)

# Planner Agent: Creates plans and uses tools for complex tasks.
planner = autogen.ConversableAgent(
    name="Planner",
    system_message="""You are the Planner. You receive complex user queries.
    Your goal is to create a step-by-step plan. You have access to a 'search_duckduckgo' tool.
    Execution Flow:
    1. Break down the query into steps.
    2. Use `search_duckduckgo` for each step to gather information.
    3. If comparison is needed, ask the 'Comparison_Agent' with the data you found.
    4. Compile a final response and send it to the 'Output_Agent' starting with "FINAL ANSWER:".""",
    llm_config={
        "config_list": config_list_groq,
        "cache_seed": 42,
    },
    human_input_mode="NEVER",
)

# Search Agent: Executes the search tool.
search_agent = autogen.ConversableAgent(
    name="Search_Agent",
    llm_config=False,  # No LLM needed
    human_input_mode="NEVER",
)

# Comparison Agent: Specializes in comparing data.
comparison_agent = autogen.ConversableAgent(
    name="Comparison_Agent",
    system_message="""You are the Comparison Agent. You receive data and a request to compare items.
    Create a clear, structured comparison (e.g., markdown table).
    Return only the structured comparison.""",
    llm_config={
        "config_list": config_list_groq,
        "cache_seed": 42,
    },
    human_input_mode="NEVER",
)

# Output Agent: Presents the final response.
output_agent = autogen.ConversableAgent(
    name="Output_Agent",
    system_message="""You are the Output Agent. You receive the final answer.
    Present this information clearly to the end-user without extra commentary.""",
    llm_config={
        "config_list": config_list_groq,
        "cache_seed": 42,
    },
    human_input_mode="NEVER",
)


# --- REGISTERING THE TOOL (FUNCTION CALLING) ---
autogen.register_function(
    search_duckduckgo,
    caller=planner,
    executor=search_agent,
    name="search_duckduckgo",
    description="A tool to search the web using DuckDuckGo for a given query.",
)


# --- THE ORCHESTRATION FLOW ---
def run_conversation(query: str):
    """
    This function orchestrates the entire agent workflow based on the query.
    """
    print(f"*****\nQuery: {query}\n*****\n")
    initiator = autogen.ConversableAgent("initiator", llm_config=False, human_input_mode="NEVER")

    print("--- Orchestrator is deciding the path ---")
    initiator.initiate_chat(orchestrator, message=query, max_turns=1)

    last_msg = initiator.last_message(orchestrator)
    if not last_msg or "content" not in last_msg or not last_msg["content"]:
        print("Orchestrator failed to respond. Defaulting to COMPLEX path.")
        decision = "COMPLEX"
    else:
        decision = last_msg["content"].strip()
    
    print(f"Orchestrator decision: {decision}")

    if "SIMPLE" in decision.upper():
        # --- SIMPLE PATH ---
        print("\n--- Path: SIMPLE ---")
        simple_task = f"Please provide a direct answer to the following user query: '{query}'"
        initiator.initiate_chat(output_agent, message=simple_task, max_turns=1)
        final_response = initiator.last_message(output_agent)["content"]

    elif "COMPLEX" in decision.upper():
        # --- COMPLEX PATH ---
        print("\n--- Path: COMPLEX ---")
        group_agents = [planner, search_agent, comparison_agent]
        group_chat = autogen.GroupChat(agents=group_agents, messages=[], max_round=15)
        manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config={ # LLM config for manager defined directly
                "config_list": config_list_groq,
                "cache_seed": 42,
            },
            system_message="You are a helpful AI assistant. Manage the conversation. Let the Planner lead. Conclude when the Planner signals 'FINAL ANSWER:'."
        )
        initiator.initiate_chat(manager, message=query)
        final_summary = initiator.last_message(manager)["content"]
        if "FINAL ANSWER:" in final_summary:
            final_summary = final_summary.replace("FINAL ANSWER:", "").strip()

        print("\n--- Handing over final summary to Output Agent ---")
        initiator.initiate_chat(output_agent, message=final_summary, max_turns=1)
        final_response = initiator.last_message(output_agent)["content"]
    
    else:
        final_response = f"The orchestrator failed to classify the query, responding with: '{decision}'. Please rephrase your query."

    # --- Print the final, clean output ---
    print("\n\n" + "="*50)
    print("                FINAL RESPONSE")
    print("="*50)
    print(final_response)
    print("="*50 + "\n")


# --- EXAMPLES ---
if __name__ == "__main__":
    
        # Example 1: Simple query
      run_conversation("What is the capital of France?")

        # Example 2: Complex query
      run_conversation("Compare the key features of the Llama 3 and Mixtral language models. Present the comparison in a markdown table.")
        
        # Example 3: Another complex query
      run_conversation("What were the top 3 selling electric cars in 2023, and what are their battery ranges?")
