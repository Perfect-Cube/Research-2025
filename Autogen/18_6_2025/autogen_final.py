# --- 1. SETUP ---

# In a new Colab notebook, run this cell first:
# !pip install "pyautogen>=0.2.25" "duckduckgo-search>=5.3.1b1" "groq>=0.9.0"
 
import os
import autogen
from duckduckgo_search import DDGS
 
# --- IMPORTANT ---
# Make sure to replace this with your actual Groq API key
GROQ_API_KEY = "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" # <--- REPLACE WITH YOUR KEY

if "gsk_" not in GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set correctly. Please replace the placeholder with your key.")

# --- 2. CONFIGURATION ---
 
llm_config = {
    "config_list": [
       {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct", # Other options: "llama3-8b-8192", "mixtral-8x7b-32768"
        "api_key": GROQ_API_KEY,
        "api_type": "groq"
    }
    ],
    "cache_seed": None,
}
 
# --- 3. TOOL DEFINITION ---

def search_duckduckgo(query: str) -> str:
    """A tool to search the web with DuckDuckGo for a query."""
    print(f"\n🔎 [TOOL] Searching for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            if not results:
                return f"No results found for '{query}'."
            return "\n".join(f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}\n---" for r in results)
    except Exception as e:
        return f"Error during search: {e}"
 
# --- 4. THE MAIN ORCHESTRATOR FUNCTION ---

def run_agent_workflow(query: str):
    """
    This function creates fresh agents for each query and runs the entire workflow.
    """
    print(f"\n\n{'='*40}\n🎬 EXECUTING NEW QUERY: {query}\n{'='*40}")
 
    # --- AGENT DEFINITIONS ---
    orchestrator = autogen.ConversableAgent(
        name="Orchestrator",
        system_message='''You are an Orchestrator. Your job is to analyze the query and respond with ONLY ONE WORD: "SIMPLE" or "COMPLEX".
- If the query is a greeting, a simple question (e.g., "what is 2+2?", "what is the capital of France?"), or basic math, it is "SIMPLE".
- If the query requires a web search, comparison of multiple items, or a detailed explanation, it is "COMPLEX".
Do not write anything else. Just the single word.''',
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
 
    planner = autogen.ConversableAgent(
        name="Planner",
        system_message=f'''You are a Planner. Your only job is to use the `search_duckduckgo` tool to find all information needed to answer the user's original query.
The user's query is: "{query}".
Think step-by-step. What information do you need?
1.  First, identify the key topics or questions in the user's query.
2.  Then, for each topic, perform a targeted search using the `search_duckduckgo` tool.
3.  Call the tool as many times as needed to gather comprehensive information.
4.  Once you believe you have all the necessary information from your searches, output all the gathered search results in a single, final message.
5.  After outputting the search results, on a new line, you MUST write the word TERMINATE to end your turn.''',
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
 
    search_agent = autogen.ConversableAgent(name="Search_Agent", llm_config=False, human_input_mode="NEVER")
 
    autogen.register_function(
        search_duckduckgo,
        caller=planner,
        executor=search_agent,
        name="search_duckduckgo",
        description="A tool to search the web with DuckDuckGo for a given query."
    )

    resolver = autogen.ConversableAgent(
        name="Resolver",
        system_message=f'''You are a Resolver. You will be given raw search data or a simple question.
Your job is to provide a complete, final answer to the user's original query: "{query}".
- If you are given search data, analyze it, synthesize the information, and format it clearly to answer the query.
- If you are asked to answer a simple question directly, do so.
- If the original query was a comparison, create a comparison table using Markdown.
- Do not mention the tools or the search process. Just provide the final, polished answer.
- Do not ask any questions. Your response is the final output.''',
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
 
    initiator = autogen.ConversableAgent("Initiator", llm_config=False, human_input_mode="NEVER")
 
    # --- WORKFLOW EXECUTION ---
 
    print("\n--- [Stage 1/3] Orchestrator deciding path... ---")
    initiator.initiate_chat(orchestrator, message=query, max_turns=1)
    decision = initiator.last_message(orchestrator)["content"].strip().upper()
    print(f"Path selected: {decision}")
    
    initiator.reset()

    if "SIMPLE" in decision:
        print("\n--- [Stage 2/2] SIMPLE path: Generating direct response... ---")
        # FIX: Added max_turns=2 to prevent the conversation from looping.
        # We only need one question from the initiator and one answer from the resolver.
        initiator.initiate_chat(
            resolver, 
            message=f"Please directly answer this simple query: {query}", 
            max_turns=2
        )
        final_response = initiator.last_message(resolver)["content"]
 
    elif "COMPLEX" in decision:
        print("\n--- [Stage 2/3] COMPLEX path: Planner gathering data... ---")
        group_chat = autogen.GroupChat(agents=[planner, search_agent], messages=[], max_round=10)
        manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper()
        )
        
        initiator.initiate_chat(manager, message=f"Gather all data needed for this query: {query}")
        gathered_data = initiator.last_message(manager)["content"].replace("TERMINATE", "").strip()
        print("\n--- Data gathering complete. ---")

        initiator.reset()
 
        print("\n--- [Stage 3/3] Resolver creating final answer... ---")
        # FIX: Added max_turns=2 here as well for the same reason.
        # We give the resolver the data and only need one final answer back.
        initiator.initiate_chat(
            resolver, 
            message=gathered_data, 
            max_turns=2
        )
        final_response = initiator.last_message(resolver)["content"]
 
    else:
        final_response = f"Sorry, the orchestrator gave an unclear response: '{decision}'. I'll try to answer directly.\n\n"
        # FIX: Also add max_turns=2 to the fallback case.
        initiator.initiate_chat(
            resolver, 
            message=f"Please directly answer this query: {query}",
            max_turns=2
        )
        final_response += initiator.last_message(resolver)["content"]

    print(f"\n\n{'='*40}\n✅ FINAL RESPONSE\n{'='*40}\n")
    print(final_response)
 
 
# --- EXAMPLES ---
if __name__ == "__main__":
    run_agent_workflow("hi, what is the capital of India?")
    run_agent_workflow("Compare the pros and cons of Llama 3 and GPT-4.")
    run_agent_workflow("What are the top 3 selling electric cars in 2024, and what are their battery ranges?")
