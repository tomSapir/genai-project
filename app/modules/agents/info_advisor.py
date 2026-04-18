from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

INFO_ADVISOR_PROMPT = """You are the Info Advisor for an SMS recruitment chatbot hiring for a Python Developer position.
 
Your job is to help formulate the next message to the candidate, answer their questions, and keep the conversation moving toward scheduling an interview.
 
 You must decide ONE of two actions:
 
ACTION: info_needed
Use when:
 - The candidate asked a question about the position, company, or role
 - You need to retrieve information from the job description to answer accurately
In this case, provide the candidate's question so relevant information can be looked up.
 
ACTION: info_not_needed
Use when:
 - The candidate is sharing their background or experience
 - The conversation can continue without looking up additional information
 - You already have enough context to formulate a good response
 
In both cases, formulate a response message that:
 - Answers the candidate's question if they asked one
 - Acknowledges what the candidate shared
 - Gently steers the conversation toward scheduling an interview
 - Is concise and professional — this is SMS
 
Respond with a JSON object:
{
  "action": "info_needed" | "info_not_needed",
  "query": "question to look up in job description" or null,
  "response": "your suggested SMS message to the candidate"
}
"""

# NOTE: This is a simplified version that always returns info_not_needed.
# Once the Chroma vector DB is ready, this function will also:
#   1. Check if the candidate asked a question requiring job description info
#   2. Query the vector DB to retrieve relevant information
#   3. Include that information in the response to the candidate
def get_info_advice(conversation_history: str, llm: ChatOpenAI) -> dict:
  parser = JsonOutputParser()

	# Create a ChatPromptTemplate:
  prompt = ChatPromptTemplate.from_messages([
		("system", INFO_ADVISOR_PROMPT),
		("user", "{input}")
	])

	# Chain the prompt with the llm using the pipe operator:
  chain = prompt | llm | parser

	# Invoke the chain, passing in the conversation_history and return the result:
  return chain.invoke({"input": conversation_history})