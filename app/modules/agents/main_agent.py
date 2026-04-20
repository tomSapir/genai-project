from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.modules.agents.exit_advisor import get_exit_advice
from app.modules.agents.scheduling_advisor import get_scheduling_advice
from app.modules.agents.info_advisor import get_info_advice

MAIN_AGENT_PROMPT = """You are an SMS recruitment chatbot for a Python Developer position at Tech company.
Your role is to interact with job candidates via SMS - gather information, answer their questions, and ultimately schedule an interview with a human recruiter or politely end the conversation.

For each candidate message, you must decide ONE of three actions:

ACTION: continue
Use when:
 - The candidate is sharing information about their background or experience
 - The candidate is asking questions about the position
 - You need to gather more information before scheduling
 - The conversation is still in its early stages

ACTION: schedule
 Use when:
 - The candidate has shown interest and shared enough background
 - It is appropriate to propose interview time slots
 - The candidate is discussing availability or dates
 - You are confirming a scheduled interview
 
 ACTION: end
 Use when:
 - The candidate explicitly says they are not interested
 - The candidate has already found another job
 - The interview has been confirmed and there is nothing left to discuss
 - The candidate asks to stop the conversation

RULES:
 - Always be professional, warm, and concise — this is SMS, keep messages short
 - Drive the conversation toward scheduling an interview when possible
 - Do not be pushy — if the candidate is not interested, respect that and end politely
 - When scheduling, propose specific time slots
 - After confirming an interview, end the conversation

You will receive the conversation history. Respond with a JSON object:
{{
  "action": "continue" | "schedule" | "end",
  "response": "your SMS message to the candidate"
}}
"""

def get_main_agent_response(conversation_history: str, llm: ChatOpenAI) -> dict:
	"""
	Main orchestrator - decides an action (continue/schedule/end) and delegates
	to the appropriate advisor for validation before returning the final response.
	"""
	parser = JsonOutputParser()
	prompt = ChatPromptTemplate.from_messages([
		("system", MAIN_AGENT_PROMPT),
		("user", "{input}")
	])
	chain = prompt | llm | parser

	# Get the main agent's initial decision
	response = chain.invoke({"input": conversation_history})
	action = response["action"]

	# Validate "end" decisions with the Exit Advisor
	if action == "end":
		exit_advice = get_exit_advice(conversation_history, llm)

		if exit_advice["action"] == "end":
			return response
		# Exit advisor disagrees — override back to continue
		action = "continue"
		response["action"] = "continue"

	# Validate "schedule" decisions with the Scheduling Advisor
	if action == "schedule":
		scheduling_advice = get_scheduling_advice(conversation_history, llm)

		if scheduling_advice["action"] == "schedule":
			return response
		# Scheduling advisor disagrees — override back to continue
		action = "continue"
		response["action"] = "continue"

	# For "continue" actions, consult the Info Advisor
	if action == "continue":
		info_advice = get_info_advice(conversation_history, llm)
		# NOTE: once Chroma is integrated, if info_advice["action"] == "info_needed",
		# we can enrich the response with relevant job information.

	return response