import os
import autogen
from duckduckgo_search import DDGS

# --- Configuration ---
# The API key is defined directly in the script for simplicity in Colab/local testing.
GROQ_API_KEY = "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" # Replace with your key if it changes

# FIX: Corrected configuration for Groq's OpenAI-compatible API
# - The 'api_key' is taken directly from the variable above.
# - 'api_type' is replaced with 'base_url' pointing to the Groq endpoint.
config_list_groq = [
    {
        "model": "llama3-70b-8192", # Or "llama3-8b-8192", "mixtral-8x7b-32768"
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1"
    }
]

# The rest of the llm_config remains the same.
llm_config_groq = {
    "config_list": config_list_groq,
    "cache_seed": 42, # Use a seed for reproducibility
}


# --- TOOL DEFINITION ---
# The tool for the Search Agent. No changes needed here.
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
        if not results:
            return f"No results found for '{query}'."
        return f"Search results for '{query}':\n\n" + "\n".join(str(res) for res in results)


# --- AGENT DEFINITIONS ---
# All agent definitions are correct and use the llm_config_groq defined above.

# Orchestrator Agent: Decides the workflow path.
orchestrator = autogen.ConversableAgent(
    name="Orchestrator",
    system_message="""You are the Orchestrator. Your job is to analyze the user's query and determine its complexity.
    - If the query is simple, can be answered directly without needing external information, planning, or comparison (e.g., "Hello", "What is 2+2?", "Summarize this text: ..."), respond with ONLY the word "SIMPLE".
    - If the query is non-trivial and requires web searches, data comparison, multi-step reasoning, or involves multiple topics (e.g., "Compare Llama 3 and GPT-4", "What are the main attractions in Paris and how much do they cost?"), respond with ONLY the word "COMPLEX".
    Do not add any other words, explanations, or punctuation.""",
    llm_config=llm_config_groq,
    human_input_mode="NEVER",
)

# Planner Agent: Creates plans and uses tools for complex tasks.
planner = autogen.ConversableAgent(
    name="Planner",
    system_message="""You are the Planner. You receive complex user queries.
    Your goal is to create a step-by-step plan to resolve the query.
    You can use tools by calling their functions and you can delegate tasks to other agents.
    You have access to a 'search_duckduckgo' tool for fetching information.
    You also work with a 'Comparison_Agent' for comparing data.
    Execution Flow:
    1. Break down the user's query into smaller, manageable steps.
    2. For each step, decide if you need to use the `search_duckduckgo` tool.
    3. Execute the search and gather all necessary information.
    4. If the query requires comparison, formulate a clear request to the 'Comparison_Agent', providing it with the data you gathered.
    5. Once all steps are complete, compile a comprehensive final response.
    6. Conclude your work by sending this final compiled response to the 'Output_Agent'. Start your message with "FINAL ANSWER:" to signify completion.
    """,
    llm_config=llm_config_groq,
    human_input_mode="NEVER",
)

# Search Agent: Executes the search tool.
search_agent = autogen.ConversableAgent(
    name="Search_Agent",
    llm_config=False,  # No LLM needed, it's just an executor
    human_input_mode="NEVER",
)

# Comparison Agent: Specializes in comparing data.
comparison_agent = autogen.ConversableAgent(
    name="Comparison_Agent",
    system_message="""You are the Comparison Agent. You receive data and a request to compare items.
    Analyze the provided data and create a clear, structured comparison (e.g., markdown table, pros-and-cons list).
    Return only the structured comparison.""",
    llm_config=llm_config_groq,
    human_input_mode="NEVER",
)

# Output Agent: Presents the final response.
output_agent = autogen.ConversableAgent(
    name="Output_Agent",
    system_message="""You are the Output Agent. You receive the final, complete answer.
    Your job is to present this information clearly and concisely to the end-user.
    Do not add any extra commentary or conversational phrases.""",
    llm_config=llm_config_groq,
    human_input_mode="NEVER",
)


# --- REGISTERING THE TOOL (FUNCTION CALLING) ---
# This part is correct.
autogen.register_function(
    search_duckduckgo,
    caller=planner,
    executor=search_agent,
    name="search_duckduckgo",
    description="A tool to search the web using DuckDuckGo for a given query.",
)


# --- THE ORCHESTRATION FLOW ---
# This orchestration logic is correct.

def run_conversation(query: str):
    """
    This function orchestrates the entire agent workflow based on the query.
    """
    print(f"*****\nQuery: {query}\n*****\n")
    initiator = autogen.ConversableAgent("initiator", llm_config=False, human_input_mode="NEVER")

    print("--- Orchestrator is deciding the path ---")
    initiator.initiate_chat(orchestrator, message=query, max_turns=1)
    decision = initiator.last_message(orchestrator)["content"]
    print(f"Orchestrator decision: {decision}")

    if "SIMPLE" in decision.upper():
        # --- SIMPLE PATH ---
        print("\n--- Path: SIMPLE ---")
        print("--- Handing over to Output Agent directly ---")
        simple_task = f"Please provide a direct answer to the following user query: '{query}'"
        initiator.initiate_chat(output_agent, message=simple_task, max_turns=1)
        final_response = initiator.last_message(output_agent)["content"]

    elif "COMPLEX" in decision.upper():
        # --- COMPLEX PATH ---
        print("\n--- Path: COMPLEX ---")
        print("--- Initiating multi-agent collaboration ---")
        group_agents = [planner, search_agent, comparison_agent]
        group_chat = autogen.GroupChat(agents=group_agents, messages=[], max_round=15)
        manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config_groq,
            system_message="You are a helpful AI assistant. You manage the conversation between the Planner, Search_Agent, and Comparison_Agent. Let the Planner lead the conversation. Conclude when the Planner signals the 'FINAL ANSWER:'."
        )
        initiator.initiate_chat(manager, message=query)
        final_summary = initiator.last_message(manager)["content"]
        if "FINAL ANSWER:" in final_summary:
            final_summary = final_summary.replace("FINAL ANSWER:", "").strip()

        print("\n--- Multi-agent collaboration complete ---")
        print("--- Handing over final summary to Output Agent ---")
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
    # Example 1: A simple query that should go directly to the Output Agent
    simple_query = "What is 10 + 10?"
    run_conversation(simple_query)

    # Example 2: A complex query that requires the full multi-agent workflow
    complex_query = "Compare the key features of the Llama 3 and Mixtral language models. Present the comparison in a markdown table."
    run_conversation(complex_query)

    # Example 3: Another complex query
    complex_query_2 = "What were the top 3 selling electric cars in 2024, and what are their battery ranges?"
    run_conversation(complex_query_2)
