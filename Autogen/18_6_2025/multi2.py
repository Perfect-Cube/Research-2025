# --- 1. SETUP ---
# In a new Colab notebook, or your local environment, run this first:
# !pip install "pyautogen>=0.2.25" "duckduckgo-search>=5.3.1b1" "groq>=0.9.0"

import os
import asyncio
import ast
from typing import List

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
    print(f"\n🔎 [Sub-agent Tool] Searching for: '{query}'")
    try:
        with DDGS() as ddgs:
            # We use text search to get snippets directly
            results = [r for r in ddgs.text(query, max_results=5)]
            if not results:
                return f"No results found for '{query}'."
            return "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results)
    except Exception as e:
        return f"Error during search for '{query}': {e}"

# --- 4. MAIN WORKFLOW IMPLEMENTATION ---

async def run_parallel_workflow(user_query: str):
    """
    Implements the agent workflow from the flowchart.
    """
    print(f"\n\n{'='*50}\n🎬 STARTING WORKFLOW FOR QUERY: {user_query}\n{'='*50}")

    # --- AGENT DEFINITIONS ---

    # A UserProxyAgent to act as the main entry point and orchestrator of chats.
    # It also serves as the tool executor for the other agents.
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy_And_Executor",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}, # For executing tool calls
        max_consecutive_auto_reply=10,
    )

    # Coordinator Agent: Receives the complex task and breaks it into a subtask list.
    coordinator = autogen.AssistantAgent(
        name="CoordinatorAgent",
        llm_config=llm_config,
        system_message=f"""You are a master coordinator. Your job is to receive a complex user query and break it down into a series of smaller, independent, and specific sub-tasks that can be executed in parallel.
Each sub-task should be a clear instruction for another agent.
The user's query is: "{user_query}"
Respond with ONLY a Python-parseable list of strings, where each string is a subtask.
For example:
["Find the current CEO of Microsoft.", "Search for the latest quarterly revenue of Microsoft.", "Summarize recent news about Microsoft's AI strategy."]
""",
    )

    # Synthesizer Agent: Collects, combines, and generates the final output.
    synthesizer = autogen.AssistantAgent(
        name="SynthesizerAgent",
        llm_config=llm_config,
        system_message="""You are a master report writer. You will be given a user's original query and a collection of results from several sub-agents.
Your sole job is to synthesize these results into a single, cohesive, well-formatted, and final answer that directly addresses the user's query.
Do not mention the sub-agents or the process. Just provide the final, polished output using Markdown for clarity.
""",
    )


    # --- STEP 1: COORDINATOR CREATES SUBTASK LIST ---
    print(f"\n--- [Phase 1/4] Coordinator is breaking down the task... ---")
    await user_proxy.a_initiate_chat(
        coordinator,
        message=f"Break down this query into subtasks: {user_query}",
        max_turns=1,
    )
    subtasks_response = user_proxy.last_message(coordinator)["content"]

    try:
        # Safely parse the string into a Python list
        subtasks = ast.literal_eval(subtasks_response)
        if not isinstance(subtasks, list): raise ValueError("Response was not a list.")
        print(f"✅ Coordinator generated {len(subtasks)} subtasks.")
    except (ValueError, SyntaxError) as e:
        print(f"❌ Error parsing subtasks: {e}. Using the original query as a single task.")
        subtasks = [user_query]


    # --- STEP 2 & 3: CREATE SUB-AGENTS AND EXECUTE IN PARALLEL ---
    print(f"\n--- [Phase 2/4] Creating Sub-agents and executing in parallel... ---")
    worker_agents = []
    for i, subtask in enumerate(subtasks):
        # This is the "Sub-agent Process" from the flowchart
        worker = autogen.AssistantAgent(
            name=f"Sub_Agent_{i+1}",
            llm_config=llm_config,
            system_message=f"""You are a specialized Sub-agent. Your goal is to fully complete the following subtask.
1.  **Receive Subtask**: Your task is: "{subtask}"
2.  **Prepare LLM Input / Use Tools**: Analyze the task. Use the `duckduckgo_search` tool to find the necessary information.
3.  **Call LLM for Processing**: Based on the search results, formulate a comprehensive answer to the subtask.
4.  **Return Result**: Your final response in this conversation will be considered the result for this subtask. Be thorough and clear.
            """,
        )
        # Register the search tool for each worker
        autogen.register_function(
            duckduckgo_search,
            caller=worker,
            executor=user_proxy,
            name="duckduckgo_search",
            description="A tool to search the web for information."
        )
        worker_agents.append(worker)

    # Use `initiate_chats` for parallel execution
    parallel_chat_requests = [
        {"recipient": agent, "message": task, "max_turns": 2, "silent": True}
        for agent, task in zip(worker_agents, subtasks)
    ]
    chat_results = await user_proxy.a_initiate_chats(parallel_chat_requests)
    print("✅ All sub-agents have completed their tasks.")


    # --- STEP 4: COLLECT, COMBINE, AND GENERATE FINAL OUTPUT ---
    print(f"\n--- [Phase 3/4] Collecting and combining results... ---")
    # Collect results from each parallel chat
    results_for_synthesis = []
    for i, result in enumerate(chat_results):
        subtask = subtasks[i]
        worker_response = result.chat_history[-1]['content']
        results_for_synthesis.append(f"--- Result for Subtask: '{subtask}' ---\n{worker_response}\n")

    combined_results = "\n".join(results_for_synthesis)
    print("✅ Results collected.")

    print(f"\n--- [Phase 4/4] Synthesizer is generating the final output... ---")
    synthesis_message = f"""The user's original query was: "{user_query}"

Here are the results gathered by the sub-agents:
{combined_results}

Please synthesize these into the final answer."""

    await user_proxy.a_initiate_chat(
        synthesizer,
        message=synthesis_message,
        max_turns=1
    )

    # --- FINAL RESPONSE ---
    final_response = user_proxy.last_message(synthesizer)["content"]
    print(f"\n\n{'='*50}\n✅ FINAL RESPONSE\n{'='*50}\n")
    print(final_response)


# --- RUN THE WORKFLOW ---
if __name__ == "__main__":
    # Use asyncio.run() to execute the async function
    query = "Compare the key features, performance, and community support for the Python web frameworks Django and FastAPI."
    asyncio.run(run_parallel_workflow(query))
