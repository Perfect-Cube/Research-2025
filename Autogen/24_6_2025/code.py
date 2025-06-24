# ==============================================================================
# CELL 1: SETUP AND INSTALLATIONS
# ==============================================================================
print("⏳ [1/5] Installing system packages for Graphviz...")
!apt-get update -qq
!apt-get install -y -qq graphviz
print("✅ [1/5] System packages installed.")

print("⏳ [2/5] Installing Python libraries (autogen, gradio, groq, etc.)...")
!pip install "pyautogen>=0.2.25" "duckduckgo-search>=5.3.1b1" "groq>=0.9.0" "gliclass>=0.2.1" "transformers" "torch" "sentence-transformers" "graphviz" "wikipedia" -q
print("✅ [2/5] Python libraries installed.")

# ==============================================================================
# CELL 2: IMPORTS AND API KEY CONFIGURATION
# ==============================================================================
print("⏳ [3/5] Importing libraries and setting up API Keys...")

# --- Standard & External Libs ---
import os
import ast
import re
import json
import time
import requests
import wikipedia
import concurrent.futures
from typing import List, Dict

# --- Service/API Libs ---
from groq import Groq
from duckduckgo_search import DDGS

# --- AutoGen Core ---
import autogen

# --- GLiClass for Classification ---
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

# --- Gradio & Graphviz for UI ---
import gradio as gr
import graphviz

# --- ⚠️ IMPORTANT: SET YOUR API KEYS HERE ---
# Replace the placeholders with your actual API keys
GROQ_API_KEY = "gsk_..."  # <--- REPLACE WITH YOUR GROQ KEY
LANGSEARCH_API_KEY = "sk-..." # <--- REPLACE WITH YOUR LANGSEARCH KEY

# --- Environment and validation ---
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if "gsk_" not in GROQ_API_KEY or "sk-" not in LANGSEARCH_API_KEY:
    raise ValueError("API Keys are not set correctly. Please replace the placeholders.")
else:
    print("✅ [3/5] API Keys set successfully.")

# ==============================================================================
# CELL 3: LLM CONFIG, GRAPHVIZ CONFIG, AND CLASSIFIER SETUP
# ==============================================================================
print("⏳ [4/5] Configuring LLM and setting up the classifier model (this may take a minute)...")

# --- LLM CONFIG (FOR AUTOGEN) ---
llm_config = {
    "config_list": [{"model": "llama3-8b-8192", "api_key": os.environ.get("GROQ_API_KEY"), "api_type": "groq"}],
    "cache_seed": None,
}

# --- GRAPHVIZ CONFIG (FOR UI) ---
NODE_IDS = {
    "human": "human_user", "classifier": "classifier", "coordinator_receives": "coord_receives",
    "subtasks_list": "subtasks_list", "create_subagents": "create_subagents",
    "execute_parallel": "execute_parallel", "collect_results": "collect_results",
    "combine_results": "combine_results", "final_output": "final_output",
    "simple_responder": "simple_responder",
}
STATUS_COLORS = {
    "pending": "#d3d3d3", "running": "#ffdd77", "completed": "#b2e5b2",
    "human": "#ffc0cb", "skipped": "#f0f0f0",
}

# --- Classifier setup ---
def setup_classifier():
    try:
        model = GLiClassModel.from_pretrained("knowledgator/gliclass-large-v1.0")
        tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-large-v1.0")
        return ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cpu')
    except Exception as e:
        print(f"❌ Failed to setup classifier: {e}")
        return None
CLASSIFIER_PIPELINE = setup_classifier()
print("✅ [4/5] Classifier setup complete.")


# ==============================================================================
# CELL 4: HELPER FUNCTIONS, TOOL DEFINITIONS, AND CORE WORKFLOW LOGIC
# ==============================================================================
print("⏳ [5/5] Defining all helper functions and workflow logic...")

def extract_python_list_from_string(text: str) -> str:
    match = re.search(r'\[[\s\S]*?\]', text)
    return match.group(0) if match else None

def classify_query(pipeline: ZeroShotClassificationPipeline, query: str, labels: List[str]) -> Dict:
    if not pipeline: return {"label": "complex_research", "score": 1.0}
    results = pipeline(query, labels, threshold=0.5)[0]
    return max(results, key=lambda x: x["score"]) if results else {"label": "complex_research", "score": 0.0}

# --- TOOL DEFINITIONS ---
def langsearch_web_search(query: str) -> str:
    """Performs a web search using the LangSearch API. Best for up-to-date information, products, or current events."""
    print(f"\n🔎 [Tool: LangSearch] Searching for: '{query}'")
    url = "https://api.langsearch.com/v1/web-search"
    payload = json.dumps({"query": query, "summary": True})
    headers = {'Authorization': LANGSEARCH_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json().get("data", {}).get("webPages", {}).get("value", [])
        if not data: return f"No results found for '{query}'."
        return "\n\n".join(f"Title: {r.get('name')}\nURL: {r.get('url')}\nSnippet: {r.get('snippet')}" for r in data[:5])
    except Exception as e:
        return f"Error during LangSearch for '{query}': {e}"

def wikipedia_search(query: str) -> str:
    """Fetches a summary from Wikipedia. Best for well-known entities, concepts, or historical events."""
    print(f"\n📚 [Tool: Wikipedia] Searching for: '{query}'")
    try:
        return wikipedia.summary(query, sentences=5, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous term. Possible options: {e.options[:5]}. Please be more specific."
    except wikipedia.exceptions.PageError:
        return f"Page '{query}' not found on Wikipedia."
    except Exception as e:
        return f"An unexpected error occurred with Wikipedia search: {e}"

def duckduckgo_search(query: str) -> str:
    """An alternative web search tool using DuckDuckGo. Use if other search tools fail."""
    print(f"\n🦆 [Tool: DuckDuckGo] Searching for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results) if results else f"No results found for '{query}'."
    except Exception as e:
        return f"Error during DuckDuckGo search for '{query}': {e}"

# --- VISUALIZATION FUNCTION ---
def create_graph_image(statuses, num_subtasks=0, temp_dir='gradio_temp'):
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    dot = graphviz.Digraph(comment='Live Agentic Process')
    dot.attr(rankdir='TB', splines='ortho')
    dot.node_attr['style'] = 'filled'
    dot.node(NODE_IDS["human"], '1. Human User', shape='box', fillcolor=statuses[NODE_IDS["human"]])
    dot.node(NODE_IDS["classifier"], '2. Classify Query', shape='ellipse', fillcolor=statuses[NODE_IDS["classifier"]])
    dot.node(NODE_IDS["coordinator_receives"], '3a. Coordinator Receives Task', shape='box', fillcolor=statuses[NODE_IDS["coordinator_receives"]])
    dot.node(NODE_IDS["subtasks_list"], '4a. Create Subtasks List', shape='note', fillcolor=statuses[NODE_IDS["subtasks_list"]])
    dot.node(NODE_IDS["create_subagents"], '5a. Create Sub-agents', shape='box', fillcolor=statuses[NODE_IDS["create_subagents"]])
    dot.node(NODE_IDS["execute_parallel"], '6a. Execute in Parallel', shape='diamond', fillcolor=statuses[NODE_IDS["execute_parallel"]])
    for i in range(1, num_subtasks + 1):
        dot.node(f"sub_agent_{i}", f'Sub-agent {i}', shape='box', fillcolor=statuses.get(f"sub_agent_{i}", STATUS_COLORS['pending']))
    dot.node(NODE_IDS["collect_results"], '7a. Collect Results', shape='box', fillcolor=statuses[NODE_IDS["collect_results"]])
    dot.node(NODE_IDS["combine_results"], '8a. Combine Results', shape='box', fillcolor=statuses[NODE_IDS["combine_results"]])
    dot.node(NODE_IDS["simple_responder"], '3b. Simple Responder', shape='box', fillcolor=statuses[NODE_IDS["simple_responder"]])
    dot.node(NODE_IDS["final_output"], '9. Generate Final Output', shape='box', fillcolor=statuses[NODE_IDS["final_output"]])
    dot.edge(NODE_IDS["human"], NODE_IDS["classifier"])
    dot.edge(NODE_IDS["classifier"], NODE_IDS["coordinator_receives"], xlabel='Complex')
    dot.edge(NODE_IDS["classifier"], NODE_IDS["simple_responder"], xlabel='Simple')
    dot.edge(NODE_IDS["coordinator_receives"], NODE_IDS["subtasks_list"])
    dot.edge(NODE_IDS["subtasks_list"], NODE_IDS["create_subagents"])
    dot.edge(NODE_IDS["create_subagents"], NODE_IDS["execute_parallel"])
    for i in range(1, num_subtasks + 1):
        dot.edge(NODE_IDS["execute_parallel"], f"sub_agent_{i}")
        dot.edge(f"sub_agent_{i}", NODE_IDS["collect_results"])
    dot.edge(NODE_IDS["collect_results"], NODE_IDS["combine_results"])
    dot.edge(NODE_IDS["combine_results"], NODE_IDS["final_output"])
    dot.edge(NODE_IDS["simple_responder"], NODE_IDS["final_output"])
    output_path = os.path.join(temp_dir, 'graph')
    dot.render(output_path, format='png', cleanup=True)
    return f"{output_path}.png"

# --- AGENT WORKFLOW FUNCTIONS ---
def run_subtask(subtask: str, worker_agent: autogen.AssistantAgent) -> str:
    """Helper to run a single sub-task with the definitive max_consecutive_auto_reply fix."""
    print(f"  🧵 [Thread] Starting task for {worker_agent.name}: '{subtask}'")
    thread_proxy = autogen.UserProxyAgent(
        name=f"Proxy_{worker_agent.name}",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=5 # THIS IS THE CRITICAL FIX
    )
    autogen.register_function(langsearch_web_search, caller=worker_agent, executor=thread_proxy)
    autogen.register_function(wikipedia_search, caller=worker_agent, executor=thread_proxy)
    autogen.register_function(duckduckgo_search, caller=worker_agent, executor=thread_proxy)
    thread_proxy.initiate_chat(worker_agent, message=subtask, silent=False)
    result = thread_proxy.last_message(worker_agent)["content"]
    print(f"  ✅ [Thread] Finished task for {worker_agent.name}.")
    return result

def run_full_process(user_query: str):
    """The main generator function that runs the whole process and yields UI updates."""
    log = ""
    statuses = {node_id: STATUS_COLORS['pending'] for node_id in NODE_IDS.values()}
    statuses[NODE_IDS["human"]] = STATUS_COLORS['human']
    log += "Process started...\n"
    yield create_graph_image(statuses), log

    # Phase 1: Classification
    statuses[NODE_IDS["classifier"]] = STATUS_COLORS['running']
    log += f"\n--- [Phase 1/2] Classifying Query ---\n"
    yield create_graph_image(statuses), log
    labels = ["trivial", "complex_research", "greetings"]
    best_label_info = classify_query(CLASSIFIER_PIPELINE, user_query, labels)
    best_label = best_label_info['label']
    statuses[NODE_IDS["classifier"]] = STATUS_COLORS['completed']
    log += f"📊 Query classified as: '{best_label}'.\n"
    yield create_graph_image(statuses), log

    if best_label in ["trivial", "greetings"]:
        # Simple Workflow
        log += "\n--- [Phase 2/2] Executing Simple Q&A Workflow ---\n"
        statuses[NODE_IDS["simple_responder"]] = STATUS_COLORS['running']
        yield create_graph_image(statuses), log
        simple_responder = autogen.AssistantAgent(name="SimpleResponder", llm_config=llm_config, system_message="Answer the user's query directly and concisely.")
        proxy = autogen.UserProxyAgent(name="UserProxy", human_input_mode="NEVER")
        proxy.initiate_chat(simple_responder, message=user_query, max_turns=1, silent=True)
        final_response = proxy.last_message(simple_responder)["content"]
        statuses[NODE_IDS["simple_responder"]] = STATUS_COLORS['completed']
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['running']
        yield create_graph_image(statuses), log
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['completed']
        log += f"\n✅ FINAL RESPONSE:\n\n{final_response}\n"
        yield create_graph_image(statuses), log
    else:
        # Complex Workflow
        num_subtasks = 0
        statuses[NODE_IDS["coordinator_receives"]] = STATUS_COLORS['running']
        log += "\n--- [Phase 2/6] Coordinator: Breaking down task... ---\n"
        yield create_graph_image(statuses, num_subtasks), log
        coordinator = autogen.AssistantAgent(name="Coordinator", llm_config=llm_config, system_message=f"You are a master coordinator. Break down the user's query into clear, independent research questions. The user's query is: \"{user_query}\"\n\nRespond with ONLY a Python-parseable list of strings.")
        proxy = autogen.UserProxyAgent(name="MainProxy", human_input_mode="NEVER")
        proxy.initiate_chat(coordinator, message=f"Break down this query: {user_query}", max_turns=1, silent=True)
        subtasks_response = proxy.last_message(coordinator)["content"]
        statuses[NODE_IDS["coordinator_receives"]] = STATUS_COLORS['completed']
        statuses[NODE_IDS["subtasks_list"]] = STATUS_COLORS['running']
        yield create_graph_image(statuses, num_subtasks), log
        subtasks = []
        try:
            list_string = extract_python_list_from_string(subtasks_response)
            subtasks = ast.literal_eval(list_string) if list_string else [user_query]
        except: subtasks = [user_query]
        num_subtasks = len(subtasks)
        log += f"✅ Coordinator generated {num_subtasks} subtasks: {subtasks}\n"
        statuses[NODE_IDS["subtasks_list"]] = STATUS_COLORS['completed']
        yield create_graph_image(statuses, num_subtasks), log
        statuses[NODE_IDS["create_subagents"]] = STATUS_COLORS['running']
        log += "\n--- [Phase 3/6] Creating & Executing Sub-agents... ---\n"
        yield create_graph_image(statuses, num_subtasks), log
        worker_agents = [autogen.AssistantAgent(name=f"Sub_Agent_{i+1}", llm_config=llm_config, system_message=("You are a research agent. You must complete your sub-task using the provided tools.\n\nTOOLS:\n1. `langsearch_web_search`: For up-to-date web searches.\n2. `wikipedia_search`: For encyclopedic summaries of well-known topics.\n3. `duckduckgo_search`: An alternative web search tool.\n\nYOUR PROCESS:\n1. Choose the single best tool for the task.\n2. Call the tool.\n3. Based ONLY on the tool's result, formulate a comprehensive answer.\n4. Your final message MUST be only this answer.")) for i in range(num_subtasks)]
        statuses[NODE_IDS["create_subagents"]] = STATUS_COLORS['completed']
        statuses[NODE_IDS["execute_parallel"]] = STATUS_COLORS['running']
        for i in range(1, num_subtasks + 1): statuses[f"sub_agent_{i}"] = STATUS_COLORS['running']
        yield create_graph_image(statuses, num_subtasks), log
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_subtasks) as executor:
            future_to_task = {executor.submit(run_subtask, task, agent): task for agent, task in zip(worker_agents, subtasks)}
            for future in concurrent.futures.as_completed(future_to_task):
                try: results.append(future.result())
                except Exception as e: results.append(f"Error processing subtask: {e}")
        log += "✅ All sub-agent threads have completed.\n"
        statuses[NODE_IDS["execute_parallel"]] = STATUS_COLORS['completed']
        for i in range(1, num_subtasks + 1): statuses[f"sub_agent_{i}"] = STATUS_COLORS['completed']
        yield create_graph_image(statuses, num_subtasks), log
        for node_id, phase_name in [(NODE_IDS["collect_results"], "Collecting"), (NODE_IDS["combine_results"], "Synthesizing")]:
            statuses[node_id] = STATUS_COLORS['running']
            log += f"\n--- [Phase {4 if node_id == NODE_IDS['collect_results'] else 5}/6] {phase_name} Results... ---\n"
            yield create_graph_image(statuses, num_subtasks), log
            time.sleep(1)
            statuses[node_id] = STATUS_COLORS['completed']
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['running']
        log += "\n--- [Phase 6/6] Generating Final Output... ---\n"
        yield create_graph_image(statuses, num_subtasks), log
        synthesizer = autogen.AssistantAgent(name="Synthesizer", llm_config=llm_config, system_message="Synthesize results from sub-agents into a single, cohesive, final answer.")
        combined_results = "\n\n".join(results)
        synthesis_message = f"Original query: \"{user_query}\"\n\nResults:\n{combined_results}\n\nSynthesize these into a final answer."
        proxy.initiate_chat(synthesizer, message=synthesis_message, max_turns=1, silent=True)
        final_response = proxy.last_message(synthesizer)["content"]
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['completed']
        log += f"\n✅ FINAL RESPONSE:\n\n{final_response}\n"
        yield create_graph_image(statuses, num_subtasks), log

print("✅ [5/5] All functions defined. Ready to launch Gradio App.")
# ==============================================================================
# CELL 5: GRADIO UI AND APP LAUNCH
# ==============================================================================

with gr.Blocks(theme=gr.themes.Soft(), title="Live Agentic Workflow") as demo:
    gr.Markdown("# Live Visualization of a Multi-Tool AutoGen Agentic Workflow")
    gr.Markdown("Enter a query below. The system will classify it, choose a workflow, and the sub-agents will **choose the best tool** to complete the task.")
    
    with gr.Row():
        query_input = gr.Textbox(label="Your Query", placeholder="e.g., What is the Hubble Space Telescope?", scale=3)
        start_button = gr.Button("▶️ Start Process", variant="primary", scale=1)

    with gr.Row():
        graph_output = gr.Image(label="Process Flowchart", type="filepath", height=700, interactive=False)
        log_output = gr.Textbox(label="Process Log & Final Answer", lines=25, interactive=False)

    start_button.click(fn=run_full_process, inputs=[query_input], outputs=[graph_output, log_output])
    
    gr.Examples(
        examples=[
            "Hello, how are you?", "What is the capital of France?",
            "What is the Large Hadron Collider?",
            "Compare the latest features of the iPhone 15 vs Samsung S24.",
        ],
        inputs=query_input
    )

# Launch the app - share=True is essential for Google Colab
demo.launch(share=True, debug=True)
