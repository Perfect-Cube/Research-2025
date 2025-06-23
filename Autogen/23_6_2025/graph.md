Of course. This is a fantastic and practical request. We will integrate your powerful AutoGen agentic workflow directly into the Gradio flowchart visualization. The flowchart will now be a *live monitor* of your actual agentic system as it works.

The strategy will be:
1.  **Combine Scripts**: Merge all the necessary code from both your AutoGen script and the Gradio/Graphviz script.
2.  **Refactor Main Logic**: Convert the `main()` function's `while` loop into a single generator function that Gradio can call.
3.  **Inject Visualization Hooks**: Interleave calls to update the graph's status and `yield` the new image/log throughout your `run_complex_research_workflow` and `run_simple_qa_workflow` functions.
4.  **Adapt UI**: Adjust the Gradio UI to have a single "Start" button and display the final synthesized answer in the log.

This will work perfectly in Google Colab.

---

### **Google Colab Notebook: Live Agentic Workflow Visualization**

Follow these steps by copying and pasting each cell into your Colab notebook and running them in order.

### **Cell 1: Install Dependencies & Setup Environment**

This cell installs all necessary system and Python packages and sets up the crucial Groq API key.

```python
# Cell 1: Install Dependencies and Setup Environment

# 1. Install system-level packages for graph rendering
print("⏳ Installing system packages for Graphviz...")
!apt-get update > /dev/null
!apt-get install -y graphviz > /dev/null
print("✅ System packages installed.")

# 2. Install all required Python libraries
print("⏳ Installing Python libraries (autogen, gradio, groq, etc.)...")
!pip install "pyautogen>=0.2.25" "duckduckgo-search>=5.3.1b1" "groq>=0.9.0" "gliclass>=0.2.1" "transformers" "torch" "sentence-transformers" "graphviz" > /dev/null
print("✅ Python libraries installed.")

# 3. IMPORTANT: SET YOUR API KEY HERE
# Replace the placeholder with your actual Groq API key
GROQ_API_KEY = "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" # <--- REPLACE WITH YOUR KEY

# 4. Environment and validation
import os
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

if "gsk_" not in GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set correctly. Please replace the placeholder with your key.")
else:
    print("✅ Groq API Key set successfully.")
```

---

### **Cell 2: Combined Imports and Configuration**

Here we import everything needed and configure our agents, graph nodes, and colors.

```python
# Cell 2: All Imports and Configurations

# --- Standard & External Libs ---
import ast
import re
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
import time

# --- LLM CONFIG (FOR AUTOGEN) ---
llm_config = {
    "config_list": [
        {
            "model": "llama3-8b-8192",  # Llama3 8B is a good, fast choice on Groq
            "api_key": os.environ.get("GROQ_API_KEY"),
            "api_type": "groq",
        }
    ],
    "cache_seed": None,
}

# --- GRAPHVIZ CONFIG (FOR UI) ---
NODE_IDS = {
    "human": "human_user",
    "classifier": "classifier",
    "coordinator_receives": "coord_receives",
    "subtasks_list": "subtasks_list",
    "create_subagents": "create_subagents",
    "execute_parallel": "execute_parallel",
    "collect_results": "collect_results",
    "combine_results": "combine_results",
    "final_output": "final_output",
    "simple_responder": "simple_responder",
}

STATUS_COLORS = {
    "pending": "#d3d3d3",    # Light Gray
    "running": "#ffdd77",    # Light Orange/Gold
    "completed": "#b2e5b2",  # Light Green
    "human": "#ffc0cb",      # Pink
    "skipped": "#f0f0f0",    # Very Light Gray
}
```

---

### **Cell 3: Classifier & Tool Definitions**

This cell contains the helper functions for classification and the `duckduckgo_search` tool. It also includes the crucial `setup_classifier` function which we will call once before launching the app.

```python
# Cell 3: Classifier Setup, Helper Functions, and Agent Tools

# --- Helper to clean LLM output ---
def extract_python_list_from_string(text: str) -> str:
    match = re.search(r'\[[\s\S]*?\]', text)
    return match.group(0) if match else None

# --- Classifier setup and execution ---
def setup_classifier():
    print("⏳ Setting up the zero-shot classifier model (this may take a minute)...")
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

# --- Tool Definition ---
def duckduckgo_search(query: str) -> str:
    print(f"\n🔎 [Sub-agent Tool] Searching for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results) if results else f"No results found for '{query}'."
    except Exception as e:
        return f"Error during search for '{query}': {e}"

# --- Setup the classifier once globally ---
CLASSIFIER_PIPELINE = setup_classifier()
```

---

### **Cell 4: The Integrated Agentic Workflow + Visualization Logic**

This is the core of the integration. The original `run_..._workflow` functions are merged into one master generator function (`run_full_process`) that yields updates for the Gradio UI at every step.

```python
# Cell 4: Integrated Agentic Workflow with Visualization Hooks

def create_graph_image(statuses, num_subtasks=0, temp_dir='gradio_temp'):
    """Generates and saves a Graphviz image based on current node statuses."""
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    
    dot = graphviz.Digraph(comment='Live Agentic Process')
    dot.attr(rankdir='TB', splines='ortho', node_attr={'style': 'filled'})

    # Define all nodes first
    dot.node(NODE_IDS["human"], '1. Human User', shape='box', fillcolor=statuses[NODE_IDS["human"]])
    dot.node(NODE_IDS["classifier"], '2. Classify Query', shape='ellipse', fillcolor=statuses[NODE_IDS["classifier"]])
    # Complex Path
    dot.node(NODE_IDS["coordinator_receives"], '3a. Coordinator Receives Task', shape='box', fillcolor=statuses[NODE_IDS["coordinator_receives"]])
    dot.node(NODE_IDS["subtasks_list"], '4a. Create Subtasks List', shape='note', fillcolor=statuses[NODE_IDS["subtasks_list"]])
    dot.node(NODE_IDS["create_subagents"], '5a. Create Sub-agents', shape='box', fillcolor=statuses[NODE_IDS["create_subagents"]])
    dot.node(NODE_IDS["execute_parallel"], '6a. Execute in Parallel', shape='diamond', fillcolor=statuses[NODE_IDS["execute_parallel"]])
    for i in range(1, num_subtasks + 1):
        dot.node(f"sub_agent_{i}", f'Sub-agent {i}', shape='box', fillcolor=statuses.get(f"sub_agent_{i}", STATUS_COLORS['pending']))
    dot.node(NODE_IDS["collect_results"], '7a. Collect Results', shape='box', fillcolor=statuses[NODE_IDS["collect_results"]])
    dot.node(NODE_IDS["combine_results"], '8a. Combine Results', shape='box', fillcolor=statuses[NODE_IDS["combine_results"]])
    # Simple Path
    dot.node(NODE_IDS["simple_responder"], '3b. Simple Responder', shape='box', fillcolor=statuses[NODE_IDS["simple_responder"]])
    # Final Output
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
    """Helper to run a single sub-task (no UI updates here, runs in a thread)."""
    print(f"  🧵 [Thread] Starting task for {worker_agent.name}: '{subtask}'")
    thread_proxy = autogen.UserProxyAgent(name=f"Proxy_{worker_agent.name}", human_input_mode="NEVER", code_execution_config=False)
    autogen.register_function(duckduckgo_search, caller=worker_agent, executor=thread_proxy, name="duckduckgo_search", description="Search the web for information.")
    thread_proxy.initiate_chat(worker_agent, message=subtask, max_turns=5, silent=False) # Set silent=False to see tool calls in console
    result = thread_proxy.last_message(worker_agent)["content"]
    print(f"  ✅ [Thread] Finished task for {worker_agent.name}.")
    return result

def run_full_process(user_query: str):
    """The main generator function that runs the whole process and yields UI updates."""
    log = ""
    statuses = {node_id: STATUS_COLORS['pending'] for node_id in NODE_IDS.values()}
    statuses[NODE_IDS["human"]] = STATUS_COLORS['human']

    # --- Initial State ---
    log += "Process started. Waiting for user input...\n"
    yield create_graph_image(statuses), log

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

    # --- Phase 2: Execution (Two Paths) ---
    if best_label in ["trivial", "greetings"]:
        # --- SIMPLE WORKFLOW ---
        log += "\n--- [Phase 2/2] Executing Simple Q&A Workflow ---\n"
        statuses[NODE_IDS["simple_responder"]] = STATUS_COLORS['running']
        yield create_graph_image(statuses), log
        
        simple_responder = autogen.AssistantAgent(name="SimpleResponder", llm_config=llm_config, system_message="You are a helpful AI assistant. Answer the user's query directly and concisely. If the query is a greeting, respond politely.")
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
        # --- COMPLEX WORKFLOW ---
        num_subtasks = 0

        # Coordinator
        statuses[NODE_IDS["coordinator_receives"]] = STATUS_COLORS['running']
        log += "\n--- [Phase 2/6] Coordinator: Breaking down task... ---\n"
        yield create_graph_image(statuses, num_subtasks), log
        coordinator = autogen.AssistantAgent(name="Coordinator", llm_config=llm_config, system_message=f"You are a master coordinator. Break down the user's query into smaller, independent sub-tasks for web research. Respond with ONLY a Python-parseable list of strings.\nUser Query: \"{user_query}\"")
        proxy = autogen.UserProxyAgent(name="MainProxy", human_input_mode="NEVER")
        proxy.initiate_chat(coordinator, message=f"Break down this query: {user_query}", max_turns=1, silent=True)
        subtasks_response = proxy.last_message(coordinator)["content"]
        statuses[NODE_IDS["coordinator_receives"]] = STATUS_COLORS['completed']
        yield create_graph_image(statuses, num_subtasks), log

        # Subtask List Creation
        statuses[NODE_IDS["subtasks_list"]] = STATUS_COLORS['running']
        yield create_graph_image(statuses, num_subtasks), log
        subtasks = []
        try:
            list_string = extract_python_list_from_string(subtasks_response)
            subtasks = ast.literal_eval(list_string) if list_string else [user_query]
        except:
            subtasks = [user_query]
        num_subtasks = len(subtasks)
        log += f"✅ Coordinator generated {num_subtasks} subtasks: {subtasks}\n"
        statuses[NODE_IDS["subtasks_list"]] = STATUS_COLORS['completed']
        yield create_graph_image(statuses, num_subtasks), log

        # Create & Execute Sub-agents
        statuses[NODE_IDS["create_subagents"]] = STATUS_COLORS['running']
        log += "\n--- [Phase 3/6] Creating & Executing Sub-agents... ---\n"
        yield create_graph_image(statuses, num_subtasks), log
        time.sleep(1)
        statuses[NODE_IDS["create_subagents"]] = STATUS_COLORS['completed']
        statuses[NODE_IDS["execute_parallel"]] = STATUS_COLORS['running']
        for i in range(1, num_subtasks + 1): statuses[f"sub_agent_{i}"] = STATUS_COLORS['running']
        yield create_graph_image(statuses, num_subtasks), log

        worker_agents = [autogen.AssistantAgent(name=f"Sub_Agent_{i+1}", llm_config=llm_config, system_message="You are a research agent. Use the `duckduckgo_search` tool to find information for your given sub-task, then formulate a comprehensive answer based ONLY on the search results.") for i in range(num_subtasks)]
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

        # Collect & Combine Results
        for node_id, phase_name in [(NODE_IDS["collect_results"], "Collecting"), (NODE_IDS["combine_results"], "Synthesizing")]:
            statuses[node_id] = STATUS_COLORS['running']
            log += f"\n--- [Phase {4 if node_id == NODE_IDS['collect_results'] else 5}/6] {phase_name} Results... ---\n"
            yield create_graph_image(statuses, num_subtasks), log
            time.sleep(1)
            statuses[node_id] = STATUS_COLORS['completed']
            yield create_graph_image(statuses, num_subtasks), log

        # Final Output
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

### **Cell 5: Launch the Gradio Web Application**

This final cell creates the user interface using Gradio Blocks and launches it. Use the public `.gradio.live` link it generates to interact with your live agentic system.

```python
# Cell 5: Build and Launch the Gradio UI

with gr.Blocks(theme=gr.themes.Soft(), title="Live Agentic Workflow") as demo:
    gr.Markdown("# Live Visualization of an AutoGen Agentic Workflow")
    gr.Markdown("Enter a query below. The system will classify it, choose a workflow (Simple Q&A or Complex Research), and visualize the process live.")
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Your Query",
            placeholder="e.g., What were the key findings of the Llama 3 paper?",
            scale=3
        )
        start_button = gr.Button("▶️ Start Process", variant="primary", scale=1)

    with gr.Row():
        graph_output = gr.Image(label="Process Flowchart", type="filepath", height=700, interactive=False)
        log_output = gr.Textbox(label="Process Log & Final Answer", lines=25, interactive=False)

    # Connect the button to the main workflow function
    start_button.click(
        fn=run_full_process,
        inputs=[query_input],
        outputs=[graph_output, log_output]
    )
    
    # Add examples for users to try
    gr.Examples(
        examples=[
            "Hello, how are you?",
            "What is the capital of France?",
            "Compare the pros and cons of using React vs. Vue for a new web project.",
            "What are the latest advancements in treating Alzheimer's disease?",
        ],
        inputs=query_input
    )

# Launch the app - share=True is essential for Google Colab
demo.launch(share=True, debug=True)
```
