from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser

from app.modules.agents.exit_advisor import get_exit_advice
from app.modules.agents.scheduling_advisor import get_scheduling_advice
from app.modules.agents.info_advisor import get_info_advice

MAIN_AGENT_PROMPT = """You are an SMS recruitment chatbot for a Python Developer position at Tech company.
Your role is to interact with job candidates via SMS - gather information, answer their questions, and ultimately schedule an interview with a human recruiter or politely end the conversation.

For each candidate message, you must decide ONE of three actions:

ACTION: continue
Use when:
 - The candidate is asking a question about the position, the company, or the role
 - The candidate has not yet shared any background about their experience and the conversation is still in its opening exchange

ACTION: schedule
Use when ANY of the following is true:
 - The candidate has just shared ANY non-trivial background about themselves (years of experience, frameworks used, types of projects, technologies) AND no interview slot has been proposed yet. Propose a slot NOW — do not keep asking background questions.
 - The candidate is discussing availability or proposing a specific time
 - A previously proposed time was rejected and a new alternative should be offered

ACTION: end
Use when:
 - The candidate explicitly says they are not interested
 - The candidate has already found another job
 - The candidate has just named or accepted a specific time slot (e.g. "Monday at 3 PM is good", "Yes, Thursday works") — your next message should confirm the booking and close the conversation
 - The interview has been confirmed and there is nothing left to discuss
 - The candidate asks to stop the conversation

RULES:
 - Always be professional, warm, and concise — this is SMS, keep messages short
 - Drive the conversation toward scheduling — once the candidate has answered a background question, move to scheduling rather than asking another one
 - Do not be pushy — if the candidate is not interested, respect that and end politely
 - When scheduling, propose specific time slots
 - After confirming an interview, end the conversation

EXAMPLES:

Example 1 — candidate has shared experience, no slot proposed yet → schedule
Conversation:
Recruiter: Hi, thanks for applying to our Python Developer opening. Could you tell me about your Python experience?
Candidate: I've been writing Python professionally for about four years, mostly building backend APIs.
Output: {{"action": "schedule", "response": "Sounds great — could you do Tuesday at 10 AM or Wednesday at 2 PM for a short interview?"}}

Example 2 — candidate is asking about the role → continue
Conversation:
Recruiter: Hi, could you tell me a bit about your Python background?
Candidate: I've worked with Django and Flask for a couple of years. Is the position remote or hybrid?
Output: {{"action": "continue", "response": "Happy to share — let me get you those details."}}

Example 3 — candidate accepted a specific time → end
Conversation:
Recruiter: Could we do Wednesday at 11 AM or Friday at 3 PM?
Candidate: Wednesday at 11 AM works for me.
Output: {{"action": "end", "response": "Perfect — your interview is confirmed. You'll receive a calendar invite shortly."}}

Example 4 — candidate accepted another offer → end
Conversation:
Recruiter: Could you tell me about your recent Python work?
Candidate: I just accepted another offer, please take me off your list.
Output: {{"action": "end", "response": "Thanks for letting me know — wishing you the best in the new role!"}}

You will receive the conversation history. Respond with a JSON object:
{{
  "action": "continue" | "schedule" | "end",
  "response": "your SMS message to the candidate"
}}
"""

def get_main_agent_response(
	conversation_history: str,
	llm: ChatOpenAI,
	reference_date: date | None = None,
) -> dict:
	"""
	Main orchestrator - decides an action (continue/schedule/end) and delegates
	to the appropriate advisor for validation before returning the final response.

	reference_date is forwarded to the Scheduling Advisor as the conversation's
	"current date" for slot lookup (see get_scheduling_advice for details).
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
		scheduling_advice = get_scheduling_advice(conversation_history, llm, reference_date)

		if scheduling_advice["action"] == "schedule":
			# Use the scheduler's slot-grounded SMS when it produced one
			# (i.e. the DB returned available slots).
			if "response" in scheduling_advice:
				response["response"] = scheduling_advice["response"]
			return response
		# Scheduling advisor disagrees — override back to continue
		action = "continue"
		response["action"] = "continue"

	# For "continue" actions, consult the Info Advisor for JD-grounded answers.
	# When it has nothing to add (info_not_needed, or info_needed but the LLM
	# returned a null query), keep the Main Agent's own draft — it has the
	# full system prompt and knows when to ask background questions.
	if action == "continue":
		info_advice = get_info_advice(conversation_history, llm)
		if info_advice.get("response") is not None:
			response["response"] = info_advice["response"]

	return response
