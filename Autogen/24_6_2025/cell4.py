# Cell 4: Integrated Agentic Workflow with Visualization Hooks (DEFINITIVE FIX)

# ==============================================================================
# FUNCTION 1: create_graph_image (with xlabel fix)
# ==============================================================================
def create_graph_image(statuses, num_subtasks=0, temp_dir='gradio_temp'):
    """Generates and saves a Graphviz image based on current node statuses."""
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    dot = graphviz.Digraph(comment='Live Agentic Process')
    dot.attr(rankdir='TB', splines='ortho')
    dot.node_attr['style'] = 'filled'

    # Define all nodes
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

    # Define Edges with 'xlabel' to prevent warnings
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

# ==============================================================================
# FUNCTION 2: run_subtask (with max_turns fix and debug print)
# ==============================================================================
def run_subtask(subtask: str, worker_agent: autogen.AssistantAgent) -> str:
    """Helper to run a single sub-task, now with multiple tools and MORE TURNS."""
    thread_proxy = autogen.UserProxyAgent(
        name=f"Proxy_{worker_agent.name}",
        human_input_mode="NEVER",
        code_execution_config=False,
        # Set max consecutive auto-reply to allow for tool use + final answer
        max_consecutive_auto_reply=5,
    )
    
    autogen.register_function(langsearch_web_search, caller=worker_agent, executor=thread_proxy)
    autogen.register_function(wikipedia_search, caller=worker_agent, executor=thread_proxy)
    autogen.register_function(duckduckgo_search, caller=worker_agent, executor=thread_proxy)
    
    # --- DEBUGGING PRINT STATEMENT ---
    print(f"\n\n>>>>> DEBUG: INITIATING CHAT FOR '{worker_agent.name}' with task: '{subtask}' and max_turns=5 <<<<<\n\n")
    
    # --- THE CRITICAL FIX: We are setting max_turns in initiate_chat ---
    thread_proxy.initiate_chat(
        worker_agent,
        message=subtask,
        max_turns=5,  # <--- THIS IS THE MOST IMPORTANT CHANGE
        silent=False
    )
    # --- END OF FIX ---

    result = thread_proxy.last_message(worker_agent)["content"]
    print(f"  ✅ [Thread] Finished task for {worker_agent.name}.")
    return result

# ==============================================================================
# FUNCTION 3: run_full_process (with improved prompts)
# ==============================================================================
def run_full_process(user_query: str):
    """The main generator function that runs the whole process and yields UI updates."""
    log = ""
    statuses = {node_id: STATUS_COLORS['pending'] for node_id in NODE_IDS.values()}
    statuses[NODE_IDS["human"]] = STATUS_COLORS['human']
    log += "Process started...\n"
    yield create_graph_image(statuses), log
    
    # --- Phase 1: Classification ---
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
        # --- SIMPLE WORKFLOW ---
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
        # --- COMPLEX WORKFLOW ---
        num_subtasks = 0
        statuses[NODE_IDS["coordinator_receives"]] = STATUS_COLORS['running']
        log += "\n--- [Phase 2/6] Coordinator: Breaking down task... ---\n"
        yield create_graph_image(statuses, num_subtasks), log
        
        coordinator = autogen.AssistantAgent(
            name="Coordinator", llm_config=llm_config,
            system_message=f"You are a master coordinator. Break down the user's query into clear, independent research questions. The user's query is: \"{user_query}\"\n\nRespond with ONLY a Python-parseable list of strings."
        )
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

        worker_agents = [
            autogen.AssistantAgent(
                name=f"Sub_Agent_{i+1}", llm_config=llm_config,
                system_message=(
                    "You are a specialized research agent. You must complete your assigned sub-task using the provided tools.\n\n"
                    "TOOLS AVAILABLE:\n1. `langsearch_web_search`: Best for general, up-to-date web searches, products, or current events.\n2. `wikipedia_search`: Best for well-defined topics, people, or places for an encyclopedic summary.\n3. `duckduckgo_search`: An alternative general web search tool.\n\n"
                    "YOUR PROCESS:\n1. Examine your assigned sub-task.\n2. Choose the single best tool for that task and call it.\n3. Based ONLY on the tool's information, formulate a comprehensive answer.\n4. Your final message MUST be this answer. Do not end on a tool call."
                )
            ) for i in range(num_subtasks)
        ]

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
        synthesizer = autogen.AssistantAgent(name="Synthesizer", llm_config=llm_config, system_message="You are a master report writer. Synthesize results from sub-agents into a single, cohesive, final answer.")
        combined_results = "\n\n".join(results)
        synthesis_message = f"Original query: \"{user_query}\"\n\nResults:\n{combined_results}\n\nSynthesize these into a final answer."
        proxy.initiate_chat(synthesizer, message=synthesis_message, max_turns=1, silent=True)
        final_response = proxy.last_message(synthesizer)["content"]
        statuses[NODE_IDS["final_output"]] = STATUS_COLORS['completed']
        log += f"\n✅ FINAL RESPONSE:\n\n{final_response}\n"
        yield create_graph_image(statuses, num_subtasks), log

print("--- All function definitions have been updated. Ready to launch the app. ---")
