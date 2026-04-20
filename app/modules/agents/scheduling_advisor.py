from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

SCHEDULING_ADVISOR_PROMPT = """You are the Scheduling Advisor for an SMS recruitment chatbot hiring for a Python Developer position.
 
Your job is to evaluate the conversation history and determine whether it is the right time to schedule an  interview.
 
You must decide ONE of two actions:
 
ACTION: schedule
Use when:
 - The candidate has expressed interest in the position
 - Enough background information has been gathered
 - The candidate is discussing availability or preferred times
 - A previously proposed time was rejected and a new one should be offered
 
ACTION: dont_schedule
Use when:
 - The conversation is still in its early stages
 - The candidate has not yet shown clear interest
 - There are still important questions to address before scheduling
 - The candidate seems hesitant and needs more information first
 
When action is "schedule", you will also receive available time slots from the recruiter's calendar. Suggest the three nearest available slots to the candidate.
 
Respond with a JSON object:
{{
  "action": "schedule" | "dont_schedule",
  "reason": "brief explanation for your decision",
  "suggested_slots": ["slot1", "slot2", "slot3"] or null
}}
"""

# NOTE: This is a simplified version that only makes the schedule/dont_schedule decision.
# Once the SQL function-calling tool is ready, this function will also:
#   1. Query the database for available time slots (filtered by position and date)
#   2. Pass the available slots to the LLM so it can suggest the 3 nearest ones
#   3. Return the suggested_slots in the response instead of null
def get_scheduling_advice(conversation_history: str, llm: ChatOpenAI) -> dict:
	parser = JsonOutputParser()

	# Create a ChatPromptTemplate:
	prompt = ChatPromptTemplate.from_messages([
		("system", SCHEDULING_ADVISOR_PROMPT),
		("user", "{input}")
	])

	# Chain the prompt with the llm using the pipe operator:
	chain = prompt | llm | parser

	# Invoke the chain, passing in the conversation_history and return the result:
	return chain.invoke({"input": conversation_history})