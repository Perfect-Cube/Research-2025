# --- 1. SETUP ---
# In a new Colab notebook, or your local environment, run this first:
# !pip install "pyautogen>=0.2.25" "duckduckgo-search>=5.3.1b1" "groq>=0.9.0"

import os
import ast
from typing import List
import concurrent.futures

from groq import Groq
from duckduckgo_search import DDGS
import autogen

# --- IMPORTANT ---
# Make sure to replace this with your actual Groq API key
GROQ_API_KEY = "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" # <--- REPLACE WITH YOUR KEY

if "gsk_" not in GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set correctly. Please replace the placeholder with your key.")

# --- 2. LLM CONFIGURATION ---
llm_config = {
    "config_list": [
        {
            "model": "llama3-70b-8192",
            "api_key": GROQ_API_KEY,
            "api_type": "groq",
        }
    ],
    "cache_seed": None,  # Use None for dynamic tasks
}

# --- 3. TOOL DEFINITION FOR SUB-AGENTS ---
def duckduckgo_search(query: str) -> str:
    """A tool to search the web with DuckDuckGo for a given query and return results."""
    # This print statement helps visualize which thread is running the tool
    print(f"\n🔎 [Sub-agent Tool] Searching for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            if not results:
                return f"No results found for '{query}'."
            return "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results)
    except Exception as e:
        return f"Error during search for '{query}': {e}"


# --- 4. Function to run a single sub-task (This will be executed in a thread) ---
def run_subtask(subtask: str, worker_agent: autogen.AssistantAgent) -> str:
    """
    Runs a single sub-task in isolation. This function is designed to be thread-safe.
    """
    print(f"  🧵 [Thread] Starting task for {worker_agent.name}: '{subtask}'")
    
    # CRITICAL: Create a new UserProxyAgent for each thread to ensure isolation.
    # This prevents conflicts in chat history and state.
    thread_local_proxy = autogen.UserProxyAgent(
        name=f"Thread_Proxy_{worker_agent.name}",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
        max_consecutive_auto_reply=4, # Limit turns for safety
    )
    
    # Register the tool for this specific, isolated interaction
    autogen.register_function(
        duckduckgo_search,
        caller=worker_agent,
        executor=thread_local_proxy,
        name="duckduckgo_search",
        description="Search the web for information."
    )
    
    # Initiate the chat synchronously within the thread
    thread_local_proxy.initiate_chat(
        worker_agent,
        message=subtask,
        max_turns=2, # The worker should search and then reply.
        silent=True,
    )
    
    # The last message from the worker agent is its result for the sub-task
    result = thread_local_proxy.last_message(worker_agent)["content"]
    print(f"  ✅ [Thread] Finished task for {worker_agent.name}.")
    return result


# --- 5. MAIN WORKFLOW IMPLEMENTATION ---

def run_parallel_workflow_threaded(user_query: str):
    """
    Implements the agent workflow from the flowchart using multi-threading.
    """
    print(f"\n\n{'='*50}\n🎬 STARTING WORKFLOW FOR QUERY: {user_query}\n{'='*50}")

    # --- AGENT DEFINITIONS ---
    # These agents are defined once and used in the main thread.
    coordinator = autogen.AssistantAgent(
        name="CoordinatorAgent",
        llm_config=llm_config,
        system_message=f"""You are a master coordinator. Break down the user's query into a series of smaller, independent sub-tasks.
Respond with ONLY a Python-parseable list of strings.
User Query: "{user_query}"
""",
    )
    
    synthesizer = autogen.AssistantAgent(
        name="SynthesizerAgent",
        llm_config=llm_config,
        system_message="""You are a master report writer. You will be given a user's original query and a collection of results from sub-agents.
Synthesize these results into a single, cohesive, well-formatted final answer. Provide only the final output.
""",
    )
    
    # This proxy is used only in the main thread for coordination and final synthesis.
    main_thread_proxy = autogen.UserProxyAgent(name="Main_Proxy", human_input_mode="NEVER")


    # --- STEP 1: COORDINATOR CREATES SUBTASK LIST (in main thread) ---
    print(f"\n--- [Phase 1/4] Coordinator is breaking down the task... ---")
    main_thread_proxy.initiate_chat(
        coordinator,
        message=f"Break down this query: {user_query}",
        max_turns=1,
    )
    subtasks_response = main_thread_proxy.last_message(coordinator)["content"]
    
    try:
        subtasks = ast.literal_eval(subtasks_response)
        if not isinstance(subtasks, list): raise ValueError("Response was not a list.")
        print(f"✅ Coordinator generated {len(subtasks)} subtasks: {subtasks}")
    except (ValueError, SyntaxError) as e:
        print(f"❌ Error parsing subtasks: {e}. Using the original query as a single task.")
        subtasks = [user_query]


    # --- STEP 2 & 3: CREATE SUB-AGENTS AND EXECUTE IN PARALLEL (using threads) ---
    print(f"\n--- [Phase 2 & 3] Creating Sub-agents and executing in parallel... ---")
    
    # Create the worker agent templates
    worker_agents = [
        autogen.AssistantAgent(
            name=f"Sub_Agent_{i+1}",
            llm_config=llm_config,
            system_message=f"""You are a specialized Sub-agent. Your goal is to fully complete a given subtask.
1. Receive Subtask.
2. Use the `duckduckgo_search` tool to find information.
3. Formulate a comprehensive answer to the subtask.
4. Your final message will be the result.""",
        )
        for i in range(len(subtasks))
    ]

    results_for_synthesis = []
    # Use ThreadPoolExecutor to run tasks concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as executor:
        # Map each worker and its subtask to the run_subtask function
        future_to_subtask = {executor.submit(run_subtask, task, agent): (task, agent.name) for agent, task in zip(worker_agents, subtasks)}
        
        for future in concurrent.futures.as_completed(future_to_subtask):
            subtask, agent_name = future_to_subtask[future]
            try:
                # Get the result from the completed thread
                worker_response = future.result()
                results_for_synthesis.append(f"--- Result from {agent_name} for Subtask: '{subtask}' ---\n{worker_response}\n")
            except Exception as exc:
                print(f"'{subtask}' generated an exception: {exc}")
                results_for_synthesis.append(f"--- Failed to get result for Subtask: '{subtask}' due to an error. ---\n")

    print("✅ All sub-agent threads have completed.")


    # --- STEP 4: COMBINE AND GENERATE FINAL OUTPUT (in main thread) ---
    print(f"\n--- [Phase 4/4] Synthesizer is generating the final output... ---")
    combined_results = "\n".join(results_for_synthesis)

    synthesis_message = f"""The user's original query was: "{user_query}"

Here are the results gathered by the sub-agents:
{combined_results}

Please synthesize these into the final answer."""

    main_thread_proxy.initiate_chat(
        synthesizer,
        message=synthesis_message,
        max_turns=1
    )

    # --- FINAL RESPONSE ---
    final_response = main_thread_proxy.last_message(synthesizer)["content"]
    print(f"\n\n{'='*50}\n✅ FINAL RESPONSE\n{'='*50}\n")
    print(final_response)


# --- RUN THE WORKFLOW ---
if __name__ == "__main__":
    query = "Compare the key features, performance, and community support for the Python web frameworks Django and FastAPI."
    run_parallel_workflow_threaded(query)
