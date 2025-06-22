# --- 1. SETUP ---
# !pip install "pyautogen>=0.2.25" "duckduckgo-search>=5.3.1b1" "groq>=0.9.0"

import os
import ast
import re
import concurrent.futures
import json
from typing import List, Dict

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
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "api_key": GROQ_API_KEY,
            "api_type": "groq",
        }
    ],
    "cache_seed": None,
}

# --- 3. HELPER FUNCTIONS ---
def extract_structured_data(text: str, data_type: type):
    """Extracts the first occurrence of a Python list or dictionary from a string."""
    if data_type == list:
        pattern = r'\[[\s\S]*?\]'
    elif data_type == dict:
        pattern = r'\{[\s\S]*?\}'
    else:
        raise ValueError("Unsupported data type. Use list or dict.")

    match = re.search(pattern, text)
    if match:
        try:
            return ast.literal_eval(match.group(0))
        except (ValueError, SyntaxError):
            return None
    return None

# --- 4. TOOL DEFINITION ---
def duckduckgo_search(query: str) -> str:
    """A tool to search the web with DuckDuckGo for a given query and return results."""
    print(f"\n\t🔎 [TOOL] Searching for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=4)]
            if not results: return f"No results found for '{query}'."
            return "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results)
    except Exception as e:
        return f"Error during search for '{query}': {e}"

# <--- CHANGE 1: REMOVED THE INCORRECT GLOBAL REGISTRATION ---
# The incorrect autogen.register_function call has been deleted from here.

# --- 5. DYNAMIC PROJECT TEAM EXECUTION ---
def run_project(project_description: str) -> str:
    """Runs a single, isolated project. It first plans a team, then executes the work in a GroupChat."""
    print(f"\n{'='*20} 🚀 LAUNCHING NEW PROJECT: {project_description} {'='*20}")

    # --- LEVEL 2: THE PLANNER ---
    planner_agent = autogen.AssistantAgent(name="Planner", llm_config=llm_config, system_message="""You are a project planner. Design a team of agents to accomplish the given project. Define roles and the number of agents for each role (e.g., Researcher, Synthesizer, Critic). Respond ONLY with a Python dictionary. Example: {"Researcher": 2, "Synthesizer": 1}""")
    planner_proxy = autogen.UserProxyAgent(name="Planner_Proxy", human_input_mode="NEVER")
    planner_proxy.initiate_chat(planner_agent, message=f"Design a team for this project: {project_description}", max_turns=1, silent=True)
    team_structure_response = planner_proxy.last_message(planner_agent)["content"]
    
    team_structure = extract_structured_data(team_structure_response, dict)
    if not team_structure or not isinstance(team_structure, dict):
        print(f"⚠️ Planner failed. Defaulting to a standard team.")
        team_structure = {"Researcher": 2, "Synthesizer": 1}
    
    print(f"\t📋 Project Team Plan: {team_structure}")

    # --- LEVEL 3: THE PROJECT TEAM ---
    
    # The Team_Manager proxy will act as the code executor for the group
    team_manager_proxy = autogen.UserProxyAgent(
        name="Team_Manager",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}
    )
    
    worker_agents = []
    for role, count in team_structure.items():
        for i in range(count):
            if role == "Researcher":
                system_message = "You are a researcher. Use the `duckduckgo_search` tool to find information. Report your findings clearly to the group."
            elif role == "Synthesizer":
                system_message = "You are a synthesizer. Gather findings from all researchers and consolidate them into a comprehensive report. Do not start until researchers have provided their data."
            else:
                system_message = f"You are a {role}. Fulfill your role's duties to contribute to the project."
                
            agent = autogen.AssistantAgent(
                name=f"{role}_{i+1}",
                llm_config=llm_config,
                system_message=system_message,
            )

            # <--- CHANGE 2: REGISTER THE TOOL DYNAMICALLY FOR EACH RESEARCHER ---
            if role == "Researcher":
                autogen.register_function(
                    duckduckgo_search,
                    caller=agent,  # The researcher can call the tool
                    executor=team_manager_proxy,  # The manager will execute the tool
                    name="duckduckgo_search",
                    description="Search the web for information about a given query."
                )

            worker_agents.append(agent)

    # Create the GroupChat
    groupchat = autogen.GroupChat(
        agents=[team_manager_proxy] + worker_agents,
        messages=[],
        max_round=15
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    print(f"\t💬 Team starting work on: '{project_description}'...")
    initial_chat_message = f"""Okay team, let's start our project: "{project_description}"
The final deliverable is a comprehensive report answering this query.
Researchers, please begin gathering information.
Synthesizer, please wait for the researchers' findings before creating the report."""

    team_manager_proxy.initiate_chat(manager, message=initial_chat_message)

    final_history = json.dumps(groupchat.messages, indent=2)
    print(f"\t✅ PROJECT COMPLETE: {project_description}")
    return f"### Project Report: {project_description}\n\n{final_history}"


# --- 6. MAIN ORCHESTRATOR WORKFLOW ---
def main():
    """The main orchestrator loop."""
    print("\n--- Dynamic Multi-Team Agent Orchestrator ---")

    orchestrator_agent = autogen.AssistantAgent(name="Orchestrator", llm_config=llm_config, system_message="""You are a master orchestrator. Decompose a user's complex query into distinct, parallelizable research projects. Respond ONLY with a Python list of strings.""")
    orchestrator_proxy = autogen.UserProxyAgent(name="Orchestrator_Proxy", human_input_mode="NEVER")
    final_synthesizer = autogen.AssistantAgent(name="Final_Synthesizer", llm_config=llm_config, system_message="You are a master report writer. You will be given the original user query and a collection of project reports (which are chat histories). Synthesize all this information into a single, cohesive, final answer for the user. Extract the key conclusions from the chat histories to build your report.")

    while True:
        query = input("\n> Enter your complex query (or type 'exit' or 'quit' to stop): ")
        if query.lower() in ["exit", "quit"]:
            print("\nExiting. Goodbye!")
            break

        print(f"\n{'#'*20} Decomposing Query {'#'*20}")
        orchestrator_proxy.initiate_chat(orchestrator_agent, message=query, max_turns=1, silent=True)
        project_list_response = orchestrator_proxy.last_message(orchestrator_agent)["content"]
        
        projects = extract_structured_data(project_list_response, list)
        if not projects or not isinstance(projects, list):
            print("⚠️ Orchestrator failed. Treating the whole query as one project.")
            projects = [query]
            
        print(f"✅ Orchestrator decomposed query into {len(projects)} project(s): {projects}")

        project_reports = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(projects)) as executor:
            future_to_project = {executor.submit(run_project, proj): proj for proj in projects}
            for future in concurrent.futures.as_completed(future_to_project):
                try:
                    report = future.result()
                    project_reports.append(report)
                except Exception as exc:
                    project_name = future_to_project[future]
                    project_reports.append(f"### FAILED Project Report: {project_name}\n\nError: {exc}")

        print(f"\n{'#'*20} Synthesizing Final Report {'#'*20}")
        
        combined_reports = "\n\n".join(project_reports)
        synthesis_message = f"""The original user query was: "{query}"

Here are the reports from the project teams:
{combined_reports}

Please synthesize these into a single, final answer for the user."""

        orchestrator_proxy.initiate_chat(final_synthesizer, message=synthesis_message, max_turns=1)
        
        final_answer = orchestrator_proxy.last_message(final_synthesizer)["content"]
        
        print(f"\n\n{'='*30} ✅ FINAL ANSWER {'='*30}\n")
        print(final_answer)
        print(f"\n{'#'*70}\n✅ Task complete. Ready for next query.\n{'#'*70}")

if __name__ == "__main__":
    main()
