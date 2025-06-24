Excellent. This is a perfect evolution of the agentic system. Adding more specialized tools and allowing the agent to *choose* the best one for the job is a key step towards more robust and intelligent automation.

We will integrate `langsearch` and `wikipedia` as new tools for your research agents. The agents will be instructed to pick the most appropriate tool for their given sub-task.

Here is the updated, self-contained Google Colab notebook. I have clearly marked the changes in each cell.

---

### **Google Colab Notebook: Multi-Tool Agentic Workflow**

Run these cells in order in your Colab notebook.

### **Cell 1: Install Dependencies & Setup Environment (MODIFIED)**

We just need to add the `wikipedia` library to our installation list. `langsearch` uses the standard `requests` library, which is already available.

```python
# Cell 1: Install Dependencies and Setup Environment (MODIFIED)

# 1. Install system-level packages for graph rendering
print("⏳ Installing system packages for Graphviz...")
!apt-get update > /dev/null
!apt-get install -y graphviz > /dev/null
print("✅ System packages installed.")

# 2. Install all required Python libraries (MODIFIED: added 'wikipedia')
print("⏳ Installing Python libraries (autogen, gradio, groq, etc.)...")
!pip install "pyautogen>=0.2.25" "duckduckgo-search>=5.3.1b1" "groq>=0.9.0" "gliclass>=0.2.1" "transformers" "torch" "sentence-transformers" "graphviz" "wikipedia" > /dev/null
print("✅ Python libraries installed.")

# 3. IMPORTANT: SET YOUR API KEYS HERE
# Replace the placeholders with your actual API keys
GROQ_API_KEY = "gsk_..." # <--- REPLACE WITH YOUR GROQ KEY
LANGSEARCH_API_KEY = "sk-975b5afab002431a99a9dd24863f55da" # <--- REPLACE WITH YOUR LANGSEARCH KEY

# 4. Environment and validation
import os
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

if "gsk_" not in GROQ_API_KEY or "sk-" not in LANGSEARCH_API_KEY:
    raise ValueError("API Keys are not set correctly. Please replace the placeholders.")
else:
    print("✅ API Keys set successfully.")
```

---

### **Cell 2: Combined Imports and Configuration (MODIFIED)**

We add imports for `requests`, `json`, and `wikipedia`.

```python
# Cell 2: All Imports and Configurations (MODIFIED)

# --- Standard & External Libs ---
import ast
import re
import concurrent.futures
from typing import List, Dict
import json # <--- ADDED
import requests # <--- ADDED
import wikipedia # <--- ADDED

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
import time

# --- LLM CONFIG (FOR AUTOGEN) ---
llm_config = {
    "config_list": [
        {
            "model": "llama3-8b-8192",
            "api_key": os.environ.get("GROQ_API_KEY"),
            "api_type": "groq",
        }
    ],
    "cache_seed": None,
}

# --- GRAPHVIZ CONFIG (FOR UI) ---
# ... (This section remains unchanged)
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
```

---

### **Cell 3: Classifier & Tool Definitions (HEAVILY MODIFIED)**

This is where we define our new tool functions. We now have three distinct search tools. The docstrings are **extremely important** as they tell the agent what each tool is for.

```python
# Cell 3: Classifier Setup, Helper Functions, and Agent Tools (HEAVILY MODIFIED)

# --- Helper to clean LLM output ---
def extract_python_list_from_string(text: str) -> str:
    match = re.search(r'\[[\s\S]*?\]', text)
    return match.group(0) if match else None

# --- Classifier setup and execution (unchanged) ---
def setup_classifier():
    print("⏳ Setting up the zero-shot classifier model...")
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
    if not pipeline: return {"label": "complex_research", "score": 1.0}
    results = pipeline(query, labels, threshold=0.5)[0]
    return max(results, key=lambda x: x["score"]) if results else {"label": "complex_research", "score": 0.0}

# --- TOOL DEFINITIONS ---

def langsearch_web_search(query: str) -> str:
    """
    Performs a web search using the LangSearch API for a given query.
    This is a general-purpose tool for finding up-to-date information on any topic.
    Returns a formatted string of search results including snippets and URLs.
    """
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
    """
    Fetches a concise summary from Wikipedia for a given search term.
    This tool is best for well-known entities, concepts, or historical events where an encyclopedic summary is needed.
    """
    print(f"\n📚 [Tool: Wikipedia] Searching for: '{query}'")
    try:
        # Use auto_suggest=True for better matching, but handle disambiguation
        summary = wikipedia.summary(query, sentences=5, auto_suggest=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous term. Possible options: {e.options[:5]}. Please be more specific."
    except wikipedia.exceptions.PageError:
        return f"Page '{query}' not found on Wikipedia."
    except Exception as e:
        return f"An unexpected error occurred with Wikipedia search: {e}"

def duckduckgo_search(query: str) -> str:
    """
    An alternative web search tool using DuckDuckGo. Use if other search tools fail.
    Note: This tool may have rate limits.
    """
    print(f"\n🦆 [Tool: DuckDuckGo] Searching for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results) if results else f"No results found for '{query}'."
    except Exception as e:
        return f"Error during DuckDuckGo search for '{query}': {e}"


# --- Setup the classifier once globally ---
CLASSIFIER_PIPELINE = setup_classifier()
```

---

### **Cell 4: Integrated Workflow with Multi-Tool Agents (MODIFIED)**

This is the most important change. We update the sub-agent's `system_message` to make it aware of all three tools, and we register all three tools in the `run_subtask` function.

```python
# Cell 4: Integrated Agentic Workflow with Multi-Tool Agents (MODIFIED)

# The create_graph_image function is correct from the last fix, no changes needed here.
def create_graph_image(statuses, num_subtasks=0, temp_dir='gradio_temp'):
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    dot = graphviz.Digraph(comment='Live Agentic Process')
    dot.attr(rankdir='TB', splines='ortho')
    dot.node_attr['style'] = 'filled'
    # ... (rest of the function is the same, no need to copy it again)
    # Define all nodes first
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
    # Define Edges
    dot.edge(NODE_IDS["human"], NODE_IDS["classifier"])
    dot.edge(NODE_IDS["classifier"], NODE_IDS["coordinator_receives"], label='Complex')
    dot.edge(NODE_IDS["classifier"], NODE_IDS["simple_responder"], label='Simple')
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


def run_subtask(subtask: str, worker_agent: autogen.AssistantAgent) -> str:
    """Helper to run a single sub-task, now with multiple tools registered."""
    print(f"  🧵 [Thread] Starting task for {worker_agent.name}: '{subtask}'")
    thread_proxy = autogen.UserProxyAgent(name=f"Proxy_{worker_agent.name}", human_input_mode="NEVER", code_execution_config=False)
    
    # --- MODIFIED: Register all available tools for the worker agent ---
    autogen.register_function(langsearch_web_search, caller=worker_agent, executor=thread_proxy)
    autogen.register_function(wikipedia_search, caller=worker_agent, executor=thread_proxy)
    autogen.register_function(duckduckgo_search, caller=worker_agent, executor=thread_proxy)
    # --- END OF MODIFICATION ---
    
    thread_proxy.initiate_chat(worker_agent, message=subtask, max_turns=5, silent=False) # silent=False to see tool calls
    result = thread_proxy.last_message(worker_agent)["content"]
    print(f"  ✅ [Thread] Finished task for {worker_agent.name}.")
    return result

def run_full_process(user_query: str):
    """The main generator function, now with a multi-tool agent prompt."""
    log = ""
    statuses = {node_id: STATUS_COLORS['pending'] for node_id in NODE_IDS.values()}
    statuses[NODE_IDS["human"]] = STATUS_COLORS['human']

    log += "Process started. Waiting for user input...\n"
    yield create_graph_image(statuses), log
    # (The rest of the initial classification logic is unchanged)
    # ...
    # --- Phase 1: Classification ---
    statuses[NODE_IDS["classifier"]] = STATUS_COLORS['running']
    log += f"\n--- [Phase 1/2] Classifying Query ---\nQuery: '{user_query}'\n"
    yield create_graph_image(statuses), log
    time.sleep(1)

    labels = ["trivial", "complex_research", "greetings"]
    best_label_info = classify_query(CLASSIFIER_PIPELINE, user_query, labels)
    best_label = best_label_info['label']
    statuses[NODE_IDS["classifier"]] = STATUS_COLORS['completed']
    log += f"📊 Query classified as: '{best_label}' with score {best_label_info['score']:.2f}.\n"
    yield create_graph_image(statuses), log
    time.sleep(1)

    if best_label in ["trivial", "greetings"]:
        # --- SIMPLE WORKFLOW (Unchanged) ---
        # ... (This whole block is the same as before)
        log += "\n--- [Phase 2/2] Executing Simple Q&A Workflow ---\n"
        statuses[NODE_IDS["simple_responder"]] = STATUS_COLORS['running']
        yield create_graph_image(statuses), log
        
        simple_responder = autogen.AssistantAgent(name="SimpleResponder", llm_config=llm_config, system_message="You are a helpful AI assistant. Answer the user's query directly and concisely.")
        proxy = autogen.UserProxyAgent(name="UserProxy", human_input_mode="NEVER")
        proxy.initiate_chat(simple_responder, message=user_query, max_turns=1, silent=True)
        final_response = proxy.last_message(simple_responder)["content"]
        
        statuses[NODE_IDS["simple_responder"]] = STATUS_COLORS['completed']
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['running']
        yield create_graph_image(statuses), log
        time.sleep(1)
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['completed']
        log += f"\n✅ FINAL RESPONSE:\n\n{final_response}\n"
        yield create_graph_image(statuses), log
    else:
        # --- COMPLEX WORKFLOW (MODIFIED: New agent prompt) ---
        # ... (Coordinator and Subtask list creation is unchanged)
        # ...
        num_subtasks = 0
        statuses[NODE_IDS["coordinator_receives"]] = STATUS_COLORS['running']
        log += "\n--- [Phase 2/6] Coordinator: Breaking down task... ---\n"
        yield create_graph_image(statuses, num_subtasks), log
        coordinator = autogen.AssistantAgent(name="Coordinator", llm_config=llm_config, system_message=f"You are a master coordinator. Break down the user's query into smaller, independent sub-tasks. Respond with ONLY a Python-parseable list of strings.\nUser Query: \"{user_query}\"")
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
        time.sleep(1)
        
        # --- MODIFIED: The system message for worker agents is now multi-tool aware ---
        worker_agents = [
            autogen.AssistantAgent(
                name=f"Sub_Agent_{i+1}",
                llm_config=llm_config,
                system_message=(
                    "You are a specialized research agent. You have access to several tools to complete your sub-task. "
                    "You must decide which tool is best for the given sub-task.\n\n"
                    "TOOLS AVAILABLE:\n"
                    "1. `langsearch_web_search`: Use for general, up-to-date web searches on any topic.\n"
                    "2. `wikipedia_search`: Use for well-defined topics, people, or places to get a reliable, encyclopedic summary.\n"
                    "3. `duckduckgo_search`: An alternative general web search tool.\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Examine the sub-task.\n"
                    "2. Choose ONE best tool and use it.\n"
                    "3. Based ONLY on the information from the tool, formulate a comprehensive answer to the sub-task.\n"
                    "4. Your final message must be ONLY this answer."
                )
            ) for i in range(num_subtasks)
        ]
        # --- END OF MODIFICATION ---

        statuses[NODE_IDS["create_subagents"]] = STATUS_COLORS['completed']
        statuses[NODE_IDS["execute_parallel"]] = STATUS_COLORS['running']
        for i in range(1, num_subtasks + 1): statuses[f"sub_agent_{i}"] = STATUS_COLORS['running']
        yield create_graph_image(statuses, num_subtasks), log
        
        # ... (The rest of the workflow is unchanged)
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
        synthesizer = autogen.AssistantAgent(name="Synthesizer", llm_config=llm_config, system_message="You are a master report writer. Synthesize results from sub-agents into a single, cohesive, final answer.")
        combined_results = "\n\n".join(results)
        synthesis_message = f"Original query: \"{user_query}\"\n\nResults:\n{combined_results}\n\nSynthesize these into a final answer."
        proxy.initiate_chat(synthesizer, message=synthesis_message, max_turns=1, silent=True)
        final_response = proxy.last_message(synthesizer)["content"]
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['completed']
        log += f"\n✅ FINAL RESPONSE:\n\n{final_response}\n"
        yield create_graph_image(statuses, num_subtasks), log
```

---

### **Cell 5: Launch the Gradio Web Application (Unchanged)**

This cell remains the same, as the UI doesn't need to know about the internal tool changes.

```python
# Cell 5: Build and Launch the Gradio UI (Unchanged)

with gr.Blocks(theme=gr.themes.Soft(), title="Live Agentic Workflow") as demo:
    gr.Markdown("# Live Visualization of a Multi-Tool AutoGen Agentic Workflow")
    gr.Markdown("Enter a query below. The system will classify it, choose a workflow, and the sub-agents will **choose the best tool** (LangSearch, Wikipedia, etc.) to complete the task.")
    
    with gr.Row():
        query_input = gr.Textbox(label="Your Query", placeholder="e.g., What is the Hubble Space Telescope?", scale=3)
        start_button = gr.Button("▶️ Start Process", variant="primary", scale=1)

    with gr.Row():
        graph_output = gr.Image(label="Process Flowchart", type="filepath", height=700, interactive=False)
        log_output = gr.Textbox(label="Process Log & Final Answer", lines=25, interactive=False)

    start_button.click(
        fn=run_full_process,
        inputs=[query_input],
        outputs=[graph_output, log_output]
    )
    
    gr.Examples(
        examples=[
            "Hello, how are you?",
            "What is the capital of France?",
            "What is the Large Hadron Collider?", # Good for Wikipedia
            "Compare the latest features of the iPhone 15 vs Samsung S24.", # Good for LangSearch
        ],
        inputs=query_input
    )

demo.launch(share=True, debug=True)
```
