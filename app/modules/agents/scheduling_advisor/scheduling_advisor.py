from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from app.modules.agents.scheduling_advisor.schedule_db import get_nearest_slots

SCHEDULING_ADVISOR_PROMPT = """You are the Scheduling Advisor for an SMS recruitment chatbot hiring for a Python Developer position.

Your job is to evaluate the conversation history and determine whether it is the right time to schedule an interview.

You must decide ONE of two actions:

ACTION: schedule
Use when ALL of the following are true:
 - The candidate has expressed clear interest in the position
 - The recruiter has already asked at least one background question (years of
   Python experience, current role, relevant projects) AND the candidate has
   answered it
 - The candidate is discussing availability, preferred times, or has explicitly
   agreed to schedule an interview
 - OR: a previously proposed time was rejected and a new one should be offered

ACTION: dont_schedule
Use when ANY of the following are true:
 - The conversation is still in its opening exchanges
 - The candidate has only said they are "interested" or asked general questions
   about the role — interest alone is NOT enough
 - The recruiter has not yet gathered any background about the candidate
 - There are still important questions to address before scheduling
 - The candidate seems hesitant and needs more information first

IMPORTANT: Interest alone is not sufficient. Before scheduling, the recruiter
must have gathered some concrete background about the candidate (experience,
current role, or skills). If that has not happened yet, choose dont_schedule
even if the candidate says "yes" to a scheduling question.

Respond with a JSON object:
{{
  "action": "schedule" | "dont_schedule",
  "reason": "brief explanation for your decision"
}}
"""

SCHEDULING_ADVISOR_ANSWER_PROMPT = """You are the Scheduling Advisor for an SMS recruitment chatbot hiring for a Python Developer position.

The candidate is ready to schedule an interview. Below are the three nearest available time slots from the recruiter's calendar.

AVAILABLE SLOTS:
 {slots}

CONVERSATION HISTORY:
 {conversation_history}

Your task: write the next SMS reply to the candidate, proposing these time slots and asking which one works best.

RULES:
  - Propose all the slots clearly, in a friendly conversational tone
  - Use ONLY the slots provided above — do not invent dates or times
  - Keep it short — this is SMS
  - Ask the candidate to pick one or suggest an alternative
  - Do not include any preamble — just write the SMS

Respond with the SMS message text only. No JSON, no quotes, no formatting."""


def get_scheduling_advice(
	conversation_history: str,
	llm: ChatOpenAI,
	reference_date: date | None = None,
) -> dict:
	"""
	Two-stage flow:
	  1. Ask the LLM to classify whether it is the right time to schedule.
	  2. If it is, query the slot DB for the nearest available slots and ask
	     the LLM again to draft an SMS proposing them.
	Returns the classification dict; when slots were retrieved, "suggested_slots"
	holds the DB rows and "response" holds the grounded SMS.

	reference_date is the date the conversation took place. The spec asks the
	advisor to anchor slot lookup to the conversation timestamp (so eval runs
	against 2024 conversations hit the 2024-seeded DB). For the live Streamlit
	demo no timestamp is available, so we fall back to date.today().
	"""
	decision_prompt = ChatPromptTemplate.from_messages([
		("system", SCHEDULING_ADVISOR_PROMPT),
		("user", "{input}")
	])
	decision_chain = decision_prompt | llm | JsonOutputParser()
	result = decision_chain.invoke({"input": conversation_history})

	if result["action"] == "schedule":
		ref = reference_date or date.today()
		slots = get_nearest_slots(ref, "Python Dev", 3)
		result["suggested_slots"] = slots

		if slots:
			slots_text = "\n".join(
				f"- {s['date']} at {s['time']}" for s in slots
			)
			answer_prompt = ChatPromptTemplate.from_messages([
				("system", SCHEDULING_ADVISOR_ANSWER_PROMPT),
			])
			answer_chain = answer_prompt | llm | StrOutputParser()
			result["response"] = answer_chain.invoke({
				"slots": slots_text,
				"conversation_history": conversation_history,
			})

	return result
