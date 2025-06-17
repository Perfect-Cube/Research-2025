import os
import autogen
from typing import Annotated
from duckduckgo_search import DDGS
from groq import Groq

# --- Configuration ---
api_key ="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"
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it to your Groq API key.")

config_list = [{
    "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "api_key": api_key,
    "api_type": "groq"
}]

# --- 1. Define REAL Tool Functions (No changes needed) ---

def search_agent_function(
    query: Annotated[str, "The search query for a specific topic."]
) -> Annotated[str, "The summary of search results."]:
    """
    A real tool that uses DuckDuckGo to search for information.
    It returns the top 3 results.
    """
    print(f"\n🔎 [Tool Call] Searching DuckDuckGo for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return f"No information found for '{query}'."
            
            formatted_results = "\n\n".join(
                [f"Title: {r['title']}\nSnippet: {r['body']}" for r in results]
            )
            
            print(f"📊 [Search Results Found]\n---\n{formatted_results}\n---")
            
            return formatted_results
    except Exception as e:
        return f"An error occurred during search: {e}"


def compare_agent_function(
    item1_summary: Annotated[str, "The search result summary of the first item to compare."],
    item2_summary: Annotated[str, "The search result summary of the second item to compare."]
) -> Annotated[str, "The detailed comparison result from the LLM."]:
    """
    A real tool that uses the Groq API (LLaMA 3.1) to compare two pieces of information.
    """
    print(f"\n⚖️ [Tool Call] Comparing items using Groq LLaMA 3.1...")
    
    try:
        client = Groq(api_key=api_key)
        comparison_prompt = f"""
        You are a highly analytical AI assistant. Your task is to provide a detailed comparison between two topics based ONLY on the information provided below. Do not use any external knowledge.

        **Topic 1 Information:**
        ---
        {item1_summary}
        ---

        **Topic 2 Information:**
        ---
        {item2_summary}
        ---

        Please provide a clear, structured comparison. Start with a brief summary of each, then highlight the key differences and potential synergies. When you are finished, terminate the conversation by saying TERMINATE.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert comparison analyst."},
                {"role": "user", "content": comparison_prompt}
            ],
            model="llama-3.1-70b-versatile",
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        return f"An error occurred during comparison: {e}"


# --- 2. Define the AI Agents ---

# FIX: The Planner's prompt is updated to guide it to give a final answer.
planner_agent = autogen.AssistantAgent(
    name="Planner_Agent",
    system_message=(
        "You are a planner. Your job is to create and execute a plan to answer a user's query. "
        "First, create a plan that uses the available tools. "
        "The user proxy agent will execute the tools for you. "
        "Review the results of the tool execution. "
        "If more steps are needed, call more tools. "
        "Once all information is gathered, provide a comprehensive final answer to the user's original query. "
        "When you are finished, you must say TERMINATE."
    ),
    llm_config={"config_list": config_list}
)

# The Output_Agent is no longer needed in this simplified flow, but we leave it for the trivial case.
output_agent = autogen.AssistantAgent(
    name="Output_Agent",
    system_message="You are an output agent. Formulate a polite, friendly response to the user's greeting.",
    llm_config={"config_list": config_list}
)

# The Executor Agent is a User Proxy that can execute tool functions.
# It needs both the llm_config and the code_execution_config to work.
executor_agent = autogen.UserProxyAgent(
    name="Executor_Agent",
    human_input_mode="NEVER", # It should run automatically
    # FIX: Set up code_execution_config to use the registered functions
    code_execution_config={"executor": autogen.coding.LocalCommandLineCodeExecutor(work_dir="coding")},
)

# Register the functions with the agent that will execute them.
executor_agent.register_function(
    function_map={
        "search_agent_function": search_agent_function,
        "compare_agent_function": compare_agent_function,
    }
)


# --- 3. The Orchestrator Logic (Simplified) ---

def orchestrator(query: str):
    print(f"🎬 [Orchestrator] Received query: '{query}'")

    trivial_greetings = ["hi", "hello", "hey", "how are you"]
    if query.lower().strip() in trivial_greetings:
        print("✅ [Orchestrator] Query is TRIVIAL. Routing to Output Agent.")
        # We use a simple proxy to talk to the output agent for greetings
        user_proxy = autogen.UserProxyAgent(name="user_proxy", human_input_mode="NEVER")
        user_proxy.initiate_chat(output_agent, message=query)
    else:
        print("🤔 [Orchestrator] Query is COMPLEX. Starting Planner-Executor conversation.")
        # FIX: For complex queries, the Executor Agent initiates a chat with the Planner Agent.
        # The Executor will automatically run the tools suggested by the Planner.
        executor_agent.initiate_chat(planner_agent, message=query)

# --- 4. Main Interactive Loop ---
if __name__ == "__main__":
    print("🤖 Multi-Agent System is ready. Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        orchestrator(user_query)
