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
    ClosureAgent,
    ClosureContext,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
    type_subscription,
)

# --- IMPORTANT ---
# Make sure to replace this with your actual Groq API key
GROQ_API_KEY = "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" # <--- REPLACE WITH YOUR KEY

if "gsk_" not in GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set correctly. Please replace the placeholder with your key.")
    
# Initialize the Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


# --- 2. DEFINE MESSAGE AND TOPIC TYPES ---

# Define the data structures (messages) that agents will pass around
@dataclass
class InitialUserQuery:
    content: str

@dataclass
class SearchTask:
    query: str
    original_user_query: str

@dataclass
class SearchResult:
    query: str
    result: str

@dataclass
class SynthesizeTask:
    original_user_query: str
    search_results: List[SearchResult]

@dataclass
class FinalAnswer:
    content: str

# Define topics to route messages between agents
INITIAL_QUERY_TOPIC = TopicId(type="initial_query")
SEARCH_TASKS_TOPIC = TopicId(type="search_tasks")
SEARCH_RESULTS_TOPIC = TopicId(type="search_results")
SYNTHESIZE_TASK_TOPIC = TopicId(type="synthesis_task")
FINAL_ANSWER_TOPIC = TopicId(type="final_answer")

# --- 3. DEFINE THE AGENTS ---

@type_subscription(topic_type=INITIAL_QUERY_TOPIC.type)
class DecomposerAgent(RoutedAgent):
    """
    Receives the initial user query and decomposes it into parallel search tasks.
    """
    @message_handler
    async def decompose_query(self, message: InitialUserQuery, ctx: MessageContext) -> None:
        print(f"🧠 [DECOMPOSER] Received query: '{message.content}'. Breaking it down...")
        
        # Use LLM to generate a list of search queries
        decomposition_prompt = f"""
        Based on the user query, what are the distinct topics you need to search for to provide a comprehensive answer?
        Respond with ONLY a Python list of strings. Each string should be a focused DuckDuckGo search query.
        For example, for "Compare Llama 3 and GPT-4o", respond with:
        ["Llama 3 model features and specs", "GPT-4o model features and specs", "Llama 3 vs GPT-4o comparison"]

        User Query: "{message.content}"
        """
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": decomposition_prompt}],
            model="llama3-8b-8192", # A smaller model is fine for this task
        )
        
        sub_queries_str = response.choices[0].message.content
        try:
            sub_queries = ast.literal_eval(sub_queries_str)
            if not isinstance(sub_queries, list): sub_queries = [message.content]
        except (ValueError, SyntaxError):
            print(f"⚠️ [DECOMPOSER] Could not parse LLM response, using original query.")
            sub_queries = [message.content]

        print(f"✅ [DECOMPOSER] Created {len(sub_queries)} search tasks.")
        
        # Publish a SearchTask for each sub-query
        for sq in sub_queries:
            task = SearchTask(query=sq, original_user_query=message.content)
            await self.publish_message(task, topic_id=SEARCH_TASKS_TOPIC)

@type_subscription(topic_type=SEARCH_TASKS_TOPIC.type)
class SearchWorkerAgent(RoutedAgent):
    """
    A worker agent that executes a single search task concurrently.
    """
    @message_handler
    async def process_search_task(self, message: SearchTask, ctx: MessageContext) -> None:
        print(f"🔎 [WORKER-{self.id.instance_id}] Starting search for: '{message.query}'")
        
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(message.query, max_results=3)]
            search_data = "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results)
            if not search_data: search_data = "No results found."
        except Exception as e:
            search_data = f"An error occurred: {e}"

        print(f"👍 [WORKER-{self.id.instance_id}] Finished search for: '{message.query}'")
        
        # Publish the result to the results topic
        await self.publish_message(SearchResult(query=message.query, result=search_data), topic_id=SEARCH_RESULTS_TOPIC)

@type_subscription(topic_type=SYNTHESIZE_TASK_TOPIC.type)
class SynthesizerAgent(RoutedAgent):
    """
    Receives all search results and synthesizes them into a final answer.
    """
    @message_handler
    async def synthesize_results(self, message: SynthesizeTask, ctx: MessageContext) -> None:
        print(f"✍️ [SYNTHESIZER] Received {len(message.search_results)} results. Synthesizing final answer...")
        
        results_str = "\n\n---\n\n".join(
            f"Results for query '{res.query}':\n{res.result}" for res in message.search_results
        )
        
        synthesis_prompt = f"""
        You are a helpful assistant. Your job is to synthesize the provided search results into a single, cohesive, and well-formatted answer to the user's original query.
        Use Markdown formatting, tables, and bullet points where appropriate.
        Do not mention the search process. Just provide the final answer.

        Original User Query: "{message.original_user_query}"

        Collected Search Data:
        ---
        {results_str}
        ---
        """

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model="llama3-70b-8192", # Use a powerful model for synthesis
        )
        final_answer_content = response.choices[0].message.content
        
        print("✅ [SYNTHESIZER] Final answer generated.")
        await self.publish_message(FinalAnswer(content=final_answer_content), topic_id=FINAL_ANSWER_TOPIC)
        

# --- 4. THE MAIN WORKFLOW ORCHESTRATOR ---

async def run_concurrent_workflow(query: str):
    """
    Sets up and runs the concurrent agent workflow.
    """
    print(f"\n\n{'='*50}\n🎬 EXECUTING NEW QUERY: {query}\n{'='*50}")
    
    runtime = SingleThreadedAgentRuntime()
    final_answer_queue = asyncio.Queue()
    
    # This list will be shared to collect results
    all_search_results: List[SearchResult] = []

    # -- Agent Registration --
    
    # 1. Register the Decomposer
    await DecomposerAgent.register(runtime, "decomposer", lambda: DecomposerAgent("Decomposer"))
    
    # 2. Register multiple Search Workers
    num_workers = 3
    for i in range(num_workers):
        await SearchWorkerAgent.register(runtime, "search_worker", lambda: SearchWorkerAgent(f"Search Worker {i+1}"))

    # 3. Register a ClosureAgent to collect search results
    async def collect_search_results(_agent: ClosureContext, message: SearchResult, _ctx: MessageContext) -> None:
        print(f"📥 [COLLECTOR] Received result for: '{message.query}'")
        all_search_results.append(message)
        # HACK/Simple Trigger: For this example, we assume if we get a result, the decomposer has finished.
        # A more robust solution might wait for a specific number of results.
        # We will trigger the synthesizer from here.
        
        # A simple way to check if all tasks are done (can be improved)
        # This part is tricky. Let's assume the Decomposer's job is done once the first result comes in.
        # We will then wait a bit for other results to arrive before synthesizing.
    
    COLLECTOR_AGENT_TYPE = "collector_agent"
    await ClosureAgent.register_closure(
        runtime,
        COLLECTOR_AGENT_TYPE,
        collect_search_results,
        subscriptions=lambda: [TypeSubscription(topic_type=SEARCH_RESULTS_TOPIC.type, agent_type=COLLECTOR_AGENT_TYPE)],
    )

    # 4. Register the Synthesizer
    await SynthesizerAgent.register(runtime, "synthesizer", lambda: SynthesizerAgent("Synthesizer"))

    # 5. Register a ClosureAgent to capture the final answer and put it in our queue
    async def capture_final_answer(_agent: ClosureContext, message: FinalAnswer, _ctx: MessageContext) -> None:
        await final_answer_queue.put(message)
        
    FINAL_ANSWER_AGENT_TYPE = "final_answer_agent"
    await ClosureAgent.register_closure(
        runtime,
        FINAL_ANSWER_AGENT_TYPE,
        capture_final_answer,
        subscriptions=lambda: [TypeSubscription(topic_type=FINAL_ANSWER_TOPIC.type, agent_type=FINAL_ANSWER_AGENT_TYPE)],
    )

    # -- Workflow Execution --
    runtime.start()

    # 1. Publish the initial query to kick things off
    await runtime.publish_message(InitialUserQuery(content=query))
    
    # 2. A simple mechanism to trigger synthesis
    # We will wait a few seconds to allow all concurrent searches to complete.
    # In a production system, you'd use a more robust counter or event-based trigger.
    await asyncio.sleep(10) # Wait for searches to complete
    
    # 3. Manually trigger the synthesis
    if all_search_results:
        await runtime.publish_message(
            SynthesizeTask(original_user_query=query, search_results=all_search_results),
            topic_id=SYNTHESIZE_TASK_TOPIC
        )
    else:
        # Handle case where no search results were found
        await final_answer_queue.put(FinalAnswer(content="Sorry, I could not find any information for your query."))

    # 4. Wait for the final answer to appear in the queue
    final_response = await final_answer_queue.get()
    
    print(f"\n\n{'='*50}\n✅ FINAL RESPONSE\n{'='*50}\n")
    print(final_response.content)
    
    await runtime.stop_when_idle()


# --- 5. RUN EXAMPLES ---
if __name__ == "__main__":
    # Use asyncio.run to execute the async main function
    asyncio.run(run_concurrent_workflow("Compare the pros and cons of Llama 3 and GPT-4."))
    asyncio.run(run_concurrent_workflow("What are the top 3 selling electric cars in 2024, and what are their battery ranges?"))
