# ==============================================================================
# FINAL, CORRECTED SINGLE-CELL SCRIPT
# ==============================================================================

# --- [1/5] Install all dependencies ---
print("⏳ [1/5] Installing system and Python packages...")
!apt-get update -qq && apt-get install -y -qq graphviz
!pip install "pyautogen>=0.2.25" "duckduckgo-search>=5.3.1b1" "groq>=0.9.0" "gliclass>=0.2.1" "transformers" "torch" "sentence-transformers" "graphviz" "wikipedia" -q
print("✅ [1/5] Installations complete.")

# --- [2/5] Imports and API Key Configuration ---
print("⏳ [2/5] Importing libraries and setting API Keys...")
import os, ast, re, json, time, requests, wikipedia, concurrent.futures
from typing import List, Dict, Any
import autogen
from groq import Groq
from duckduckgo_search import DDGS
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import gradio as gr
import graphviz

# --- ⚠️ IMPORTANT: SET YOUR API KEYS HERE ---
GROQ_API_KEY = "gsk_..."  # <--- REPLACE WITH YOUR GROQ KEY
LANGSEARCH_API_KEY = "sk-..." # <--- REPLACE WITH YOUR LANGSEARCH KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if "gsk_" not in GROQ_API_KEY or "sk-" not in LANGSEARCH_API_KEY:
    raise ValueError("API Keys are not set correctly.")
else:
    print("✅ [2/5] API Keys set successfully.")

# --- [3/5] LLM Config, Graphviz Config, Classifier Setup ---
print("⏳ [3/5] Configuring LLM and setting up classifier model...")
llm_config = {"config_list": [{"model": "llama3-8b-8192", "api_key": os.environ.get("GROQ_API_KEY"), "api_type": "groq"}], "cache_seed": None}
NODE_IDS = { "human": "human_user", "classifier": "classifier", "coordinator_receives": "coord_receives", "subtasks_list": "subtasks_list", "create_subagents": "create_subagents", "execute_parallel": "execute_parallel", "collect_results": "collect_results", "combine_results": "combine_results", "final_output": "final_output", "simple_responder": "simple_responder" }
STATUS_COLORS = { "pending": "#d3d3d3", "running": "#ffdd77", "completed": "#b2e5b2", "human": "#ffc0cb", "skipped": "#f0f0f0" }
def setup_classifier():
    try:
        model = GLiClassModel.from_pretrained("knowledgator/gliclass-large-v1.0")
        tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-large-v1.0")
        return ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cpu')
    except Exception as e: return None
CLASSIFIER_PIPELINE = setup_classifier()
print("✅ [3/5] Classifier setup complete.")

# --- [4/5] Helper Functions, Tool Definitions, and Core Workflow Logic ---
print("⏳ [4/5] Defining workflow functions...")
def extract_python_list_from_string(text: str) -> str:
    match = re.search(r'\[[\s\S]*?\]', text)
    return match.group(0) if match else None
def classify_query(pipeline: ZeroShotClassificationPipeline, query: str, labels: List[str]) -> Dict:
    if not pipeline: return {"label": "complex_research", "score": 1.0}
    results = pipeline(query, labels, threshold=0.5)[0]
    return max(results, key=lambda x: x["score"]) if results else {"label": "complex_research", "score": 0.0}
def langsearch_web_search(query: str) -> str:
    print(f"\n🔎 [Tool: LangSearch] Searching for: '{query}'")
    url = "https://api.langsearch.com/v1/web-search"
    payload = json.dumps({"query": query, "summary": True})
    headers = {'Authorization': LANGSEARCH_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json().get("data", {}).get("webPages", {}).get("value", [])
        return "\n\n".join(f"Title: {r.get('name')}\nURL: {r.get('url')}\nSnippet: {r.get('snippet')}" for r in data[:5]) if data else f"No results for '{query}'."
    except Exception as e: return f"Error during LangSearch for '{query}': {e}"
def wikipedia_search(query: str) -> str:
    print(f"\n📚 [Tool: Wikipedia] Searching for: '{query}'")
    try: return wikipedia.summary(query, sentences=5, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e: return f"Ambiguous term. Options: {e.options[:5]}."
    except wikipedia.exceptions.PageError: return f"Page '{query}' not found."
    except Exception as e: return f"Error with Wikipedia: {e}"
def duckduckgo_search(query: str) -> str:
    print(f"\n🦆 [Tool: DuckDuckGo] Searching for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results) if results else f"No results for '{query}'."
    except Exception as e: return f"Error during DuckDuckGo: {e}"
def create_graph_image(statuses, num_subtasks=0, temp_dir='gradio_temp'):
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', splines='ortho')
    dot.node_attr['style'] = 'filled'
    for node_id, label in {NODE_IDS["human"]: '1. Human User', NODE_IDS["classifier"]: '2. Classify Query', NODE_IDS["coordinator_receives"]: '3a. Coordinator', NODE_IDS["subtasks_list"]: '4a. Create Subtasks', NODE_IDS["create_subagents"]: '5a. Create Sub-agents', NODE_IDS["execute_parallel"]: '6a. Execute in Parallel', NODE_IDS["collect_results"]: '7a. Collect Results', NODE_IDS["combine_results"]: '8a. Combine Results', NODE_IDS["simple_responder"]: '3b. Simple Responder', NODE_IDS["final_output"]: '9. Generate Final Output'}.items():
        shape = 'diamond' if node_id == NODE_IDS["execute_parallel"] else 'ellipse' if node_id == NODE_IDS["classifier"] else 'note' if node_id == NODE_IDS["subtasks_list"] else 'box'
        dot.node(node_id, label, shape=shape, fillcolor=statuses[node_id])
    for i in range(1, num_subtasks + 1): dot.node(f"sub_agent_{i}", f'Sub-agent {i}', shape='box', fillcolor=statuses.get(f"sub_agent_{i}", STATUS_COLORS['pending']))
    for start, end, label in [(NODE_IDS["human"], NODE_IDS["classifier"], ''), (NODE_IDS["classifier"], NODE_IDS["coordinator_receives"], 'Complex'), (NODE_IDS["classifier"], NODE_IDS["simple_responder"], 'Simple'), (NODE_IDS["coordinator_receives"], NODE_IDS["subtasks_list"], ''), (NODE_IDS["subtasks_list"], NODE_IDS["create_subagents"], ''), (NODE_IDS["create_subagents"], NODE_IDS["execute_parallel"], ''), (NODE_IDS["collect_results"], NODE_IDS["combine_results"], ''), (NODE_IDS["combine_results"], NODE_IDS["final_output"], ''), (NODE_IDS["simple_responder"], NODE_IDS["final_output"], '')]: dot.edge(start, end, xlabel=label)
    for i in range(1, num_subtasks + 1): dot.edge(NODE_IDS["execute_parallel"], f"sub_agent_{i}"); dot.edge(f"sub_agent_{i}", NODE_IDS["collect_results"])
    output_path = os.path.join(temp_dir, 'graph')
    dot.render(output_path, format='png', cleanup=True)
    return f"{output_path}.png"

# ==============================================================================
# THIS IS THE DEFINITIVELY CORRECTED SUB-TASK FUNCTION
# ==============================================================================
def run_subtask(subtask: str, worker_agent: autogen.AssistantAgent) -> str:
    """Helper to run a single sub-task with the definitive termination fix."""
    print(f"  🧵 [Thread] Starting task for {worker_agent.name}: '{subtask}'")
    
    # 1. Define a termination message function. The chat ends when the last message
    #    is NOT a tool call, meaning the agent has provided its final text answer.
    def is_termination_msg(x: Dict[str, Any]) -> bool:
        return x.get("tool_calls") is None

    # 2. Create the proxy agent with the correct parameters
    thread_proxy = autogen.UserProxyAgent(
        name=f"Proxy_{worker_agent.name}",
        human_input_mode="NEVER",
        # Set how the proxy decides to terminate the conversation
        is_termination_msg=is_termination_msg,
        # Allow the proxy to reply automatically to tool calls
        max_consecutive_auto_reply=5,
        # The tool executor needs to know about the function definitions
        code_execution_config=False,
    )
    
    # 3. Register all tools with the agent and proxy
    autogen.register_function(langsearch_web_search, caller=worker_agent, executor=thread_proxy)
    autogen.register_function(wikipedia_search, caller=worker_agent, executor=thread_proxy)
    autogen.register_function(duckduckgo_search, caller=worker_agent, executor=thread_proxy)
    
    # 4. Initiate the chat with a safety-net turn limit
    thread_proxy.initiate_chat(
        worker_agent,
        message=subtask,
        max_turns=5, # A safety limit in case the termination message logic fails
        silent=False
    )
    
    # The conversation has now correctly terminated, and the last message is the result
    result = thread_proxy.last_message(worker_agent)["content"]
    print(f"  ✅ [Thread] Finished task for {worker_agent.name}.")
    return result

def run_full_process(user_query: str):
    """The main generator function that runs the whole process and yields UI updates."""
    log = ""
    statuses = {node_id: STATUS_COLORS['pending'] for node_id in NODE_IDS.values()}
    statuses[NODE_IDS["human"]] = STATUS_COLORS['human']
    log += "Process started...\n"; yield create_graph_image(statuses), log
    statuses[NODE_IDS["classifier"]] = STATUS_COLORS['running']
    log += f"\n--- [Phase 1/2] Classifying Query ---\n"; yield create_graph_image(statuses), log
    labels = ["trivial", "complex_research", "greetings"]
    best_label_info = classify_query(CLASSIFIER_PIPELINE, user_query, labels)
    best_label = best_label_info['label']
    statuses[NODE_IDS["classifier"]] = STATUS_COLORS['completed']
    log += f"📊 Query classified as: '{best_label}'.\n"; yield create_graph_image(statuses), log
    if best_label in ["trivial", "greetings"]:
        log += "\n--- [Phase 2/2] Executing Simple Q&A Workflow ---\n"; statuses[NODE_IDS["simple_responder"]] = STATUS_COLORS['running']; yield create_graph_image(statuses), log
        simple_responder = autogen.AssistantAgent(name="SimpleResponder", llm_config=llm_config, system_message="Answer the user's query directly and concisely.")
        proxy = autogen.UserProxyAgent(name="UserProxy", human_input_mode="NEVER")
        proxy.initiate_chat(simple_responder, message=user_query, max_turns=1, silent=True)
        final_response = proxy.last_message(simple_responder)["content"]
        statuses[NODE_IDS["simple_responder"]] = STATUS_COLORS['completed']; statuses[NODE_IDS["final_output"]] = STATUS_COLORS['running']; yield create_graph_image(statuses), log
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['completed']; log += f"\n✅ FINAL RESPONSE:\n\n{final_response}\n"; yield create_graph_image(statuses), log
    else:
        num_subtasks = 0
        statuses[NODE_IDS["coordinator_receives"]] = STATUS_COLORS['running']; log += "\n--- [Phase 2/6] Coordinator: Breaking down task... ---\n"; yield create_graph_image(statuses, num_subtasks), log
        coordinator = autogen.AssistantAgent(name="Coordinator", llm_config=llm_config, system_message=f"Break down the user's query into clear, independent research questions. Query: \"{user_query}\"\n\nRespond ONLY with a Python-parseable list of strings.")
        proxy = autogen.UserProxyAgent(name="MainProxy", human_input_mode="NEVER")
        proxy.initiate_chat(coordinator, message=f"Break down this query: {user_query}", max_turns=1, silent=True)
        subtasks_response = proxy.last_message(coordinator)["content"]
        statuses[NODE_IDS["coordinator_receives"]] = STATUS_COLORS['completed']; statuses[NODE_IDS["subtasks_list"]] = STATUS_COLORS['running']; yield create_graph_image(statuses, num_subtasks), log
        try: subtasks = ast.literal_eval(extract_python_list_from_string(subtasks_response)) if extract_python_list_from_string(subtasks_response) else [user_query]
        except: subtasks = [user_query]
        num_subtasks = len(subtasks)
        log += f"✅ Coordinator generated {num_subtasks} subtasks: {subtasks}\n"; statuses[NODE_IDS["subtasks_list"]] = STATUS_COLORS['completed']; yield create_graph_image(statuses, num_subtasks), log
        statuses[NODE_IDS["create_subagents"]] = STATUS_COLORS['running']; log += "\n--- [Phase 3/6] Creating & Executing Sub-agents... ---\n"; yield create_graph_image(statuses, num_subtasks), log
        worker_agents = [autogen.AssistantAgent(name=f"Sub_Agent_{i+1}", llm_config=llm_config, system_message=("You are a research agent. You must complete your sub-task using tools. Choose the single best tool, call it, then based ONLY on its result, formulate a comprehensive answer. Your final message MUST be this answer.")) for i in range(num_subtasks)]
        statuses[NODE_IDS["create_subagents"]] = STATUS_COLORS['completed']; statuses[NODE_IDS["execute_parallel"]] = STATUS_COLORS['running']
        for i in range(1, num_subtasks + 1): statuses[f"sub_agent_{i}"] = STATUS_COLORS['running']
        yield create_graph_-image(statuses, num_subtasks), log
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_subtasks) as executor:
            future_to_task = {executor.submit(run_subtask, task, agent): task for agent, task in zip(worker_agents, subtasks)}
            for future in concurrent.futures.as_completed(future_to_task):
                try: results.append(future.result())
                except Exception as e: results.append(f"Error processing subtask: {e}")
        log += "✅ All sub-agent threads have completed.\n"; statuses[NODE_IDS["execute_parallel"]] = STATUS_COLORS['completed']
        for i in range(1, num_subtasks + 1): statuses[f"sub_agent_{i}"] = STATUS_COLORS['completed']
        yield create_graph_image(statuses, num_subtasks), log
        for node_id, phase_name in [(NODE_IDS["collect_results"], "Collecting"), (NODE_IDS["combine_results"], "Synthesizing")]:
            statuses[node_id] = STATUS_COLORS['running']; log += f"\n--- [Phase {4 if node_id == NODE_IDS['collect_results'] else 5}/6] {phase_name} Results... ---\n"; yield create_graph_image(statuses, num_subtasks), log
            time.sleep(1); statuses[node_id] = STATUS_COLORS['completed']
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['running']; log += "\n--- [Phase 6/6] Generating Final Output... ---\n"; yield create_graph_image(statuses, num_subtasks), log
        synthesizer = autogen.AssistantAgent(name="Synthesizer", llm_config=llm_config, system_message="Synthesize results from sub-agents into a single, cohesive, final answer.")
        combined_results = "\n\n".join(results)
        synthesis_message = f"Original query: \"{user_query}\"\n\nResults:\n{combined_results}\n\nSynthesize these into a final answer."
        proxy.initiate_chat(synthesizer, message=synthesis_message, max_turns=1, silent=True)
        final_response = proxy.last_message(synthesizer)["content"]
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['completed']; log += f"\n✅ FINAL RESPONSE:\n\n{final_response}\n"; yield create_graph_image(statuses, num_subtasks), log
print("✅ [5/5] All functions defined. Ready to launch.")

# --- [LAUNCH] Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Live Agentic Workflow") as demo:
    gr.Markdown("## Live Visualization of a Multi-Tool AutoGen Agentic Workflow")
    gr.Markdown("Enter a query below. The system will classify it, choose a workflow, and the sub-agents will **choose the best tool** to complete the task.")
    with gr.Row():
        query_input = gr.Textbox(label="Your Query", placeholder="e.g., What is the Hubble Space Telescope?", scale=3)
        start_button = gr.Button("▶️ Start Process", variant="primary", scale=1)
    with gr.Row():
        graph_output = gr.Image(label="Process Flowchart", type="filepath", height=700, interactive=False)
        log_output = gr.Textbox(label="Process Log & Final Answer", lines=25, interactive=False)
    start_button.click(fn=run_full_process, inputs=[query_input], outputs=[graph_output, log_output])
    gr.Examples(examples=["What is the capital of France?", "What is the Large Hadron Collider?", "Compare the latest features of the iPhone 15 vs Samsung S24."], inputs=query_input)
demo.launch(share=True, debug=True)
