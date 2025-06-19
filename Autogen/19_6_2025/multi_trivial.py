# --- 1. SETUP ---
# !pip install "pyautogen>=0.2.25" "duckduckgo-search>=5.3.1b1" "groq>=0.9.0" "gliclass>=0.2.1" "transformers" "torch" "sentence-transformers"

import os
import ast
import re
import concurrent.futures
from typing import List, Dict

from groq import Groq
from duckduckgo_search import DDGS
import autogen

# --- GLiClass Imports ---
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

# --- IMPORTANT ---
# Make sure to replace this with your actual Groq API key
GROQ_API_KEY = "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" # <--- REPLACE WITH YOUR KEY

if "gsk_" not in GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set correctly. Please replace the placeholder with your key.")

# --- 2. LLM CONFIGURATION ---
llm_config = {
    "config_list": [
        {
            # <--- FIX: Using a more reliable model for structured tool use ---
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "api_key": GROQ_API_KEY,
            "api_type": "groq",
        }
    ],
    "cache_seed": None,
}

# --- 3. HELPER FUNCTION TO CLEAN LLM OUTPUT ---
def extract_python_list_from_string(text: str) -> str:
    """Finds and extracts the first Python list from a string."""
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        return match.group(0)
    return None

# --- 4. CLASSIFIER SETUP ---
def setup_classifier():
    """Loads and initializes the GLiClass zero-shot classification pipeline."""
    print("Setting up the classifier model...")
    try:
        model = GLiClassModel.from_pretrained("knowledgator/gliclass-large-v1.0")
        tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-large-v1.0")
        pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cpu')
        print("✅ Classifier setup complete.")
        return pipeline
    except Exception as e:
        print(f"❌ Failed to setup classifier: {e}")
        return None

def classify_query(pipeline: ZeroShotClassificationPipeline, query: str, labels: List[str]) -> Dict:
    """Classifies the query and returns the label with the highest score."""
    if not pipeline:
        print("⚠️ Classifier not available. Defaulting to complex research.")
        return {"label": "complex_research", "score": 1.0}
    results = pipeline(query, labels, threshold=0.0)[0]
    if not results:
        return {"label": "complex_research", "score": 0.0}
    return max(results, key=lambda x: x["score"])

# --- 5. TOOL DEFINITION ---
def duckduckgo_search(query: str) -> str:
    """A tool to search the web with DuckDuckGo for a given query and return results."""
    print(f"\n🔎 [Sub-agent Tool] Searching for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            if not results: return f"No results found for '{query}'."
            return "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results)
    except Exception as e:
        return f"Error during search for '{query}': {e}"

# --- 6. WORKFLOW IMPLEMENTATIONS ---

def run_simple_qa_workflow(user_query: str):
    """Handles simple queries with a single agent for a direct answer."""
    print(f"\n\n{'='*50}\n🚀 STARTING SIMPLE Q&A WORKFLOW\n{'='*50}")
    simple_responder = autogen.AssistantAgent(name="SimpleResponder", llm_config=llm_config, system_message="You are a helpful AI assistant. Answer the user's query directly and concisely. If the query is a greeting, respond politely.")
    user_proxy = autogen.UserProxyAgent(name="UserProxy", human_input_mode="NEVER")
    user_proxy.initiate_chat(simple_responder, message=user_query, max_turns=1)
    final_response = user_proxy.last_message(simple_responder)["content"]
    print(f"\n\n{'='*50}\n✅ FINAL RESPONSE (Simple Q&A)\n{'='*50}\n")
    print(final_response)

def run_subtask(subtask: str, worker_agent: autogen.AssistantAgent) -> str:
    """Runs a single sub-task in isolation (for the complex workflow)."""
    print(f"  🧵 [Thread] Starting task for {worker_agent.name}: '{subtask}'")
    thread_local_proxy = autogen.UserProxyAgent(name=f"Thread_Proxy_{worker_agent.name}", human_input_mode="NEVER", code_execution_config={"use_docker": False}, max_consecutive_auto_reply=5)
    autogen.register_function(duckduckgo_search, caller=worker_agent, executor=thread_local_proxy, name="duckduckgo_search", description="Search the web for information.")

    thread_local_proxy.initiate_chat(
        worker_agent,
        message=subtask,
        # <--- FIX 1: INCREASED MAX_TURNS TO ALLOW FOR FULL TOOL USE FLOW ---
        max_turns=5,
        silent=True,
    )

    # <--- FIX 2: ADDED ROBUSTNESS CHECK ---
    # Check if the conversation finished properly or was cut short.
    last_message = thread_local_proxy.last_message(worker_agent)
    if last_message.get("tool_calls") or "function_call" in last_message.get("content", ""):
        # The agent's last message was a tool call, meaning it didn't get to summarize.
        return f"Error: The sub-task for '{subtask}' ended prematurely without a final answer."
    else:
        # The conversation completed successfully. The agent's last message is the answer.
        result = last_message["content"]
        print(f"  ✅ [Thread] Finished task for {worker_agent.name}.")
        return result

def run_complex_research_workflow(user_query: str):
    """Implements the agent workflow with parallel sub-agents for research."""
    print(f"\n\n{'='*50}\n🎬 STARTING COMPLEX RESEARCH WORKFLOW\n{'='*50}")

    coordinator = autogen.AssistantAgent(name="CoordinatorAgent", llm_config=llm_config, system_message=f"You are a master coordinator. Break down the user's query into smaller, independent sub-tasks for web research. Respond with ONLY a Python-parseable list of strings.\nUser Query: \"{user_query}\"")
    synthesizer = autogen.AssistantAgent(name="SynthesizerAgent", llm_config=llm_config, system_message="You are a master report writer. Synthesize results from sub-agents into a single, cohesive, final answer.")
    main_thread_proxy = autogen.UserProxyAgent(name="Main_Proxy", human_input_mode="NEVER")

    print(f"\n--- [Phase 1/4] Coordinator is breaking down the task... ---")
    main_thread_proxy.initiate_chat(coordinator, message=f"Break down this query: {user_query}", max_turns=1)
    subtasks_response = main_thread_proxy.last_message(coordinator)["content"]

    subtasks = []
    try:
        list_string = extract_python_list_from_string(subtasks_response)
        if list_string:
            subtasks = ast.literal_eval(list_string)
            if not isinstance(subtasks, list): raise ValueError("Parsed object is not a list.")
            print(f"✅ Coordinator generated {len(subtasks)} subtasks: {subtasks}")
        else:
            raise ValueError("No Python list found in the coordinator's response.")
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"❌ Error parsing subtasks: {e}. Using the original query as a single task.")
        subtasks = [user_query]
    
    if not subtasks:
        print("⚠️ Subtask list is empty. Using the original query as a single task.")
        subtasks = [user_query]

    print(f"\n--- [Phase 2 & 3] Creating Sub-agents and executing in parallel... ---")
    worker_agents = [
        autogen.AssistantAgent(
            name=f"Sub_Agent_{i+1}",
            llm_config=llm_config,
            # <--- FIX 3: IMPROVED SYSTEM PROMPT FOR CLARITY ---
            system_message="You are a specialized research agent. Your goal is to complete a given sub-task. First, use the `duckduckgo_search` tool to find information. Second, based ONLY on the search results, formulate a comprehensive answer to the sub-task. Your final message should be this answer.",
        ) for i in range(len(subtasks))
    ]
    results_for_synthesis = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as executor:
        future_to_subtask = {executor.submit(run_subtask, task, agent): (task, agent.name) for agent, task in zip(worker_agents, subtasks)}
        for future in concurrent.futures.as_completed(future_to_subtask):
            subtask, agent_name = future_to_subtask[future]
            try:
                worker_response = future.result()
                results_for_synthesis.append(f"--- Result from {agent_name} for Subtask: '{subtask}' ---\n{worker_response}\n")
            except Exception as exc:
                results_for_synthesis.append(f"--- Failed to get result for Subtask: '{subtask}' due to an error: {exc} ---\n")
    print("✅ All sub-agent threads have completed.")

    print(f"\n--- [Phase 4/4] Synthesizer is generating the final output... ---")
    combined_results = "\n".join(results_for_synthesis)
    synthesis_message = f"The user's original query was: \"{user_query}\"\n\nHere are the results gathered by the sub-agents:\n{combined_results}\n\nPlease synthesize these into the final answer."
    main_thread_proxy.initiate_chat(synthesizer, message=synthesis_message, max_turns=1)
    final_response = main_thread_proxy.last_message(synthesizer)["content"]
    print(f"\n\n{'='*50}\n✅ FINAL RESPONSE (Complex Research)\n{'='*50}\n")
    print(final_response)

# --- 7. MAIN ROUTER AND EXECUTION ---
def main():
    """Main function to classify queries and route them to the appropriate workflow."""
    classifier_pipeline = setup_classifier()
    if classifier_pipeline is None:
        print("Fatal: Could not initialize the classifier. Exiting program.")
        return

    labels = ["trivial", "non-trivial","greetings", "complex_research","nonsensical","malformed","technology"]
    
    print("\n--- Interactive Agentic Workflow Router ---")
    print("The system will classify your query and route it to the appropriate workflow.")

    while True:
        query = input("\n> Enter your query (or type 'exit' or 'quit' to stop): ")
        if query.lower() in ["exit", "quit"]:
            print("\nExiting program. Goodbye!")
            break

        print(f"\n\n{'#'*70}\nProcessing new query: '{query}'\n{'#'*70}")

        best_label_info = classify_query(classifier_pipeline, query, labels)
        best_label = best_label_info['label']
        print(f"📊 Query classified as: '{best_label}' with a score of {best_label_info['score']:.4f}")

        if best_label in ["trivial", "greetings"]:
            run_simple_qa_workflow(query)
        else:
            run_complex_research_workflow(query)
        
        print(f"\n{'#'*70}\n✅ Task complete. Ready for next query.\n{'#'*70}")

if __name__ == "__main__":
    main()
