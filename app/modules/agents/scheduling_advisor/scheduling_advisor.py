from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from app.modules.agents.scheduling_advisor.schedule_db import get_nearest_slots

SCHEDULING_ADVISOR_PROMPT = """You are the Scheduling Advisor for an SMS recruitment chatbot hiring for a Python Developer position.

Your job is to evaluate the conversation history and determine whether it is the right time to schedule an interview.

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


def get_scheduling_advice(conversation_history: str, llm: ChatOpenAI) -> dict:
	"""
	Two-stage flow:
	  1. Ask the LLM to classify whether it is the right time to schedule.
	  2. If it is, query the slot DB for the nearest available slots and ask
	     the LLM again to draft an SMS proposing them.
	Returns the classification dict; when slots were retrieved, "suggested_slots"
	holds the DB rows and "response" holds the grounded SMS.
	"""
	decision_prompt = ChatPromptTemplate.from_messages([
		("system", SCHEDULING_ADVISOR_PROMPT),
		("user", "{input}")
	])
	decision_chain = decision_prompt | llm | JsonOutputParser()
	result = decision_chain.invoke({"input": conversation_history})

	if result["action"] == "schedule":
		slots = get_nearest_slots(date.today(), "Python Dev", 3)
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
