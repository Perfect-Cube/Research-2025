# --- 1. SETUP ---
# !pip install "autogen-core>=0.2.25" "duckduckgo-search>=5.3.1b1" "groq>=0.9.0"

import os
import asyncio
import ast
from dataclasses import dataclass
from typing import List

from groq import Groq
from duckduckgo_search import DDGS

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)

# --- IMPORTANT ---
# Make sure to replace this with your actual Groq API key
GROQ_API_KEY = "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" # <--- REPLACE WITH YOUR KEY

if "gsk_" not in GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set correctly. Please replace the placeholder with your key.")

# Initialize the Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


# --- 2. DEFINE MESSAGE TYPES (for direct request/response) ---

@dataclass
class UserQuery:
    """The initial user request sent to the Orchestrator."""
    query: str

@dataclass
class SearchTask:
    """A single search task sent from the Orchestrator to a SearchWorker."""
    query: str

@dataclass
class SearchResult:
    """The result of a search task returned from a SearchWorker."""
    query: str
    result: str

@dataclass
class FinalAnswer:
    """The final, synthesized answer returned by the Orchestrator."""
    content: str


# --- 3. DEFINE THE AGENTS (Hierarchical Structure) ---

class SearchWorkerAgent(RoutedAgent):
    """
    A worker agent that executes a single search task and returns the result.
    """
    def __init__(self) -> None:
        super().__init__(description="A worker agent that performs a web search.")

    @message_handler
    async def handle_search_task(self, message: SearchTask, ctx: MessageContext) -> SearchResult:
        # FIX: Use self.id.key instead of self.id.instance_id
        print(f"🔎 [WORKER-{self.id.key}] Starting search for: '{message.query}'")
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(message.query, max_results=3)]
            search_data = "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results)
            if not search_data:
                search_data = "No results found."
        except Exception as e:
            search_data = f"An error occurred during search: {e}"

        # FIX: Use self.id.key instead of self.id.instance_id
        print(f"👍 [WORKER-{self.id.key}] Finished search for: '{message.query}'")
        return SearchResult(query=message.query, result=search_data)


class OrchestratorAgent(RoutedAgent):
    """
    Controls the entire workflow:
    1. Decomposes the user query into sub-queries.
    2. Dispatches search tasks to worker agents.
    3. Collects the results.
    4. Synthesizes the final answer.
    """
    def __init__(
        self,
        groq_client: Groq,
        worker_agent_type: str,
        num_workers: int,
    ) -> None:
        super().__init__(description="Orchestrator for the search workflow.")
        self._groq_client = groq_client
        self._worker_agent_type = worker_agent_type
        self._num_workers = num_workers

    @message_handler
    async def handle_user_query(self, message: UserQuery, ctx: MessageContext) -> FinalAnswer:
        print(f"\n🧠 [ORCHESTRATOR] Received query: '{message.query}'. Starting workflow...")

        # --- Step 1: Decompose query into sub-tasks ---
        print("🧠 [ORCHESTRATOR] Decomposing query into search tasks...")
        decomposition_prompt = f"""
        Based on the user query, what are the distinct topics you need to search for to provide a comprehensive answer?
        Respond with ONLY a Python list of strings. Each string should be a focused DuckDuckGo search query.
        User Query: "{message.query}"
        """
        response = self._groq_client.chat.completions.create(
            messages=[{"role": "user", "content": decomposition_prompt}],
            model="llama3-8b-8192",
        )
        
        # IMPROVEMENT: More robust parsing for the LLM response
        sub_queries_str = response.choices[0].message.content
        try:
            # Find the start and end of the list in the response string
            list_start = sub_queries_str.find('[')
            list_end = sub_queries_str.rfind(']') + 1
            if list_start != -1 and list_end != 0:
                clean_list_str = sub_queries_str[list_start:list_end]
                sub_queries = ast.literal_eval(clean_list_str)
                if not isinstance(sub_queries, list) or not sub_queries:
                    sub_queries = [message.query]
            else:
                raise ValueError("List not found in response")
        except (ValueError, SyntaxError):
            print("⚠️ [ORCHESTRATOR] Could not parse LLM response, using original query.")
            sub_queries = [message.query]
        
        print(f"✅ [ORCHESTRATOR] Created {len(sub_queries)} search tasks: {sub_queries}")

        # --- Step 2: Dispatch tasks to workers and collect results ---
        print(f"🚀 [ORCHESTRATOR] Dispatching tasks to {self._num_workers} workers...")
        
        tasks_to_dispatch = []
        for i, sq in enumerate(sub_queries):
            worker_id = AgentId(self._worker_agent_type, f"worker_{i % self._num_workers}")
            task = SearchTask(query=sq)
            tasks_to_dispatch.append(self.send_message(task, worker_id))
        
        search_results: List[SearchResult] = await asyncio.gather(*tasks_to_dispatch)
        print(f"📥 [ORCHESTRATOR] Received all {len(search_results)} search results.")

        # --- Step 3: Synthesize final answer ---
        print("✍️ [ORCHESTRATOR] Synthesizing final answer...")
        results_str = "\n\n---\n\n".join(
            f"Results for query '{res.query}':\n{res.result}" for res in search_results
        )
        synthesis_prompt = f"""
        You are a helpful assistant. Your job is to synthesize the provided search results into a single, cohesive, and well-formatted answer to the user's original query.
        Use Markdown formatting, tables, and bullet points where appropriate.
        Do not mention the search process. Just provide the final answer.

        Original User Query: "{message.query}"

        Collected Search Data:
        ---
        {results_str}
        ---
        """
        final_response = self._groq_client.chat.completions.create(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model="llama3-70b-8192",
        )
        final_answer_content = final_response.choices[0].message.content
        
        print("✅ [ORCHESTRATOR] Final answer generated.")
        return FinalAnswer(content=final_answer_content)


async def main():
    """
    Sets up and runs the hierarchical agent workflow.
    """
    runtime = SingleThreadedAgentRuntime()
    num_workers = 3
    worker_type_name = "search_worker"

    await SearchWorkerAgent.register(runtime, worker_type_name, lambda: SearchWorkerAgent())

    await OrchestratorAgent.register(
        runtime,
        "orchestrator",
        lambda: OrchestratorAgent(
            groq_client=groq_client,
            worker_agent_type=worker_type_name,
            num_workers=num_workers,
        ),
    )

    queries = [
        "Compare the pros and cons of Llama 3 and GPT-4.",
        "What are the top 3 selling electric cars in 2024, and what are their battery ranges?"
    ]

    runtime.start()

    for task_query in queries:
        print(f"\n\n{'='*60}\n🎬 EXECUTING NEW QUERY: {task_query}\n{'='*60}")
        
        result = await runtime.send_message(UserQuery(query=task_query), AgentId("orchestrator", "default"))
        
        print(f"\n{'='*60}\n✅ FINAL RESPONSE\n{'='*60}\n")
        print(result.content)

    await runtime.stop_when_idle()


# In an environment with a running event loop (like a Jupyter notebook),
# you must `await` the async function directly instead of using `asyncio.run()`.
await main()
