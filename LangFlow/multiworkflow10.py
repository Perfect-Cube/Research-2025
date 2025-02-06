# Install necessary libraries
!pip install langchain langchain-groq requests beautifulsoup4 time json
 
# Import required libraries
import os
import time
import json
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
 
# Set Groq API Key
os.environ["GROQ_API_KEY"] = "key"
 
# Initialize Groq Chat Model
chat = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.7)
 
# Function to introduce sleep time for real-world simulation
def sleep_simulation(task_name, delay=3):
    print(f"\n⏳ Processing {task_name}... (Sleeping for {delay} seconds)")
    time.sleep(delay)
 
# Function to run AI workflows
# Function to run AI workflows
def run_workflow(task, user_input):
    task_prompts = {
        "news_summarization": "Summarize this news article:\n\n{text}",
        "sentiment_analysis": "Analyze the sentiment (Positive, Neutral, Negative) of this text:\n\n{text}",
        "translation": "Translate this product review into French:\n\n{text}",
        "code_gen": "Write a Python script for:\n\n{text}",
        "fake_news_detection": "Determine if this is fake news:\n\n{text}",
        "keyword_extraction": "Extract the main topics from this industry report:\n\n{text}",
        "entity_recognition": "Identify named entities (people, organizations, places) in the text:\n\n{text}",
        "grammar_correction": "Fix grammatical errors in this business email:\n\n{text}",
        "content_creation": "Write a blog post about:\n\n{text}",
    }

    if task not in task_prompts:
        return "❌ Invalid task."

    system_message = task_prompts[task]
    prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", "{text}")])
    chain = prompt | chat

    # Ensure we pass 'text' to match the prompt template
    response = chain.invoke({"text": user_input})
    return response.content

 
# 1️⃣ Fetch real-world news headlines
def get_live_news():
    print("\n🌍 Fetching real-time news headlines...")
    url = "https://news.ycombinator.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [item.text for item in soup.select(".storylink")][:3]  # Get top 3 headlines
    return headlines
 
# 2️⃣ Define real-world inputs for other tasks
real_world_inputs = {
    "sentiment_analysis": "I just received my order, and I am extremely disappointed with the quality!",
    "translation": "This smartphone has an amazing battery life, but the camera quality is not up to the mark.",
    "code_gen": "Write a Python script to monitor stock prices every hour and send alerts.",
    "fake_news_detection": "A new study claims that drinking coffee will make you live up to 200 years.",
    "keyword_extraction": "This quarterly industry report highlights trends in renewable energy, focusing on solar, wind, and battery technology.",
    "entity_recognition": "Apple Inc. announced a major event in California where Tim Cook presented the new iPhone.",
    "grammar_correction": "I has send the email yesterday but didn't got any reply.",
    "content_creation": "The impact of AI in transforming healthcare and medical research.",
}
 
# 🚀 Simulating the Workflow Execution
print("\n📌 Starting AI Workflows with Real-World Data...")
 
# News Summarization
news_headlines = get_live_news()
for i, headline in enumerate(news_headlines):
    sleep_simulation(f"News Summarization {i+1}")
    print(f"\n📰 Headline: {headline}")
    print(run_workflow("news_summarization", headline))
 
# Process other AI tasks with real-world inputs
for task, input_text in real_world_inputs.items():
    sleep_simulation(task)
    print(f"\n🔍 {task.replace('_', ' ').title()}:")
    print(f"📝 Input: {input_text}")
    print(f"💡 Output: {run_workflow(task, input_text)}")
 
print("\n✅ Simulation Complete!")
