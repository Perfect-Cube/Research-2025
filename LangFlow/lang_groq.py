# Install necessary libraries
!pip install langchain langchain-groq
 
# Import required libraries
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
 
import os
 
# Set Groq API Key
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
 
# Initialize the Groq Chat Model
chat = ChatGroq(model="mixtral-8x7b-32768", temperature=0.7)
 
# Define the system and user prompts
system_message = "You are a helpful assistant."
user_message = "Explain the significance of quantum computing in modern technology."
 
# Create the prompt template
prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", "{text}")])
 
# Define a function to run the workflow
def run_workflow(user_input):
    chain = prompt | chat
    response = chain.invoke({"text": user_input})
    return response.content
 
# Example usage
response = run_workflow(user_message)
print("AI Response:", response)
