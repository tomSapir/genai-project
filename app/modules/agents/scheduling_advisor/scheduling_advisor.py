from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from app.modules.agents.scheduling_advisor.schedule_db import get_slots_by_day

DATE_EXTRACTION_PROMPT = """Today's date is {today}.

Read the conversation below and extract the most recent date or day reference made by either speaker
(e.g. "next Friday", "Monday", "June 15", "this Thursday").

If a specific date is mentioned, resolve it to an exact calendar date based on today's date.
If no date is mentioned, return null.

Respond with a JSON object:
{{
  "date": "YYYY-MM-DD" or null
}}

CONVERSATION:
{conversation_history}
"""

SCHEDULING_ADVISOR_PROMPT = """You are the Scheduling Advisor for an SMS recruitment chatbot hiring for a Python Developer position.

Your job is to evaluate the conversation history and determine whether it is the right time to schedule an interview.

You must decide ONE of two actions:

ACTION: schedule
Use when ANY of the following is true:
 - The candidate has just shared ANY non-trivial background about themselves (years of experience, frameworks used, types of projects, technologies) AND no interview slot has been proposed yet. This is enough on its own — the candidate does NOT need to explicitly ask for a meeting.
 - The candidate is discussing availability, proposing or accepting a specific time, or explicitly agreeing to schedule an interview
 - A previously proposed time was rejected and a new alternative should be offered

ACTION: dont_schedule
Use when ANY of the following is true:
 - The candidate has not yet shared anything about their experience and the conversation is in its opening exchange
 - The candidate is asking a question about the role (location, stack, compensation, etc.) and that question is still unanswered
 - The candidate has declined the position or asked to stop being contacted

EXAMPLES:

Example 1 — candidate shared background, no slot proposed yet → schedule
Conversation:
Recruiter: How long have you been working with Python?
Candidate: About four years, mostly on backend services.
Output: {{"action": "schedule", "reason": "Candidate has shared experience; time to propose interview slots."}}

Example 2 — candidate asking an unanswered question about the role → dont_schedule
Conversation:
Recruiter: Hi, could you tell me about your Python experience?
Candidate: I've used Flask for a couple of years. Is the role remote or hybrid?
Output: {{"action": "dont_schedule", "reason": "Candidate has an open question about the role; answer it before scheduling."}}

Example 3 — previously proposed slot rejected → schedule
Conversation:
Recruiter: Could we do Tuesday at 10 AM or Wednesday at 2 PM?
Candidate: Those don't work for me. Any other times?
Output: {{"action": "schedule", "reason": "Candidate rejected the proposed slots and is open to alternatives."}}

Example 4 — candidate declined → dont_schedule
Conversation:
Recruiter: How about Wednesday at 10 AM?
Candidate: Please remove me from your list, I'm not interested anymore.
Output: {{"action": "dont_schedule", "reason": "Candidate has asked to stop; do not propose further slots."}}

Respond with a JSON object:
{{
  "action": "schedule" | "dont_schedule",
  "reason": "brief explanation for your decision"
}}
"""

SCHEDULING_ADVISOR_ANSWER_PROMPT = """You are the Scheduling Advisor for an SMS recruitment chatbot hiring for a Python Developer position.

The candidate is ready to schedule an interview. Below are the three nearest available time slots from the recruiter's calendar.

Today's date is {today}.

AVAILABLE SLOTS:
 {slots}

CANDIDATE'S REQUESTED DATE: {requested_date}

CONVERSATION HISTORY:
 {conversation_history}

Your task: write the next SMS reply to the candidate, proposing these time slots and asking which one works best.

RULES:
  - You MUST list ALL THREE slots — do not drop or merge any of them
  - Use ONLY the slots provided above — do not invent dates or times
  - Keep it friendly and conversational
  - Ask the candidate to pick one or suggest an alternative
  - Do not include any preamble — just write the SMS
  - If CANDIDATE'S REQUESTED DATE is not "none" and none of the available slots fall on that date,
    start with a brief apology (e.g. "Unfortunately we don't have availability on [date],")
    then offer the three nearest slots
  - If the candidate asks what year the slots are in, use the year shown in the slot dates above — do not guess

Respond with the SMS message text only. No JSON, no quotes, no formatting."""


def _extract_date(conversation_history: str, llm: ChatOpenAI) -> date | None:
	"""
	Use the LLM to extract and resolve any date mentioned in the conversation
	(e.g. 'next Friday', 'Monday at 3 PM') into an exact calendar date.
	Returns None if no date is found or parsing fails.
	"""
	prompt = ChatPromptTemplate.from_messages([
		("system", DATE_EXTRACTION_PROMPT),
	])
	chain = prompt | llm | JsonOutputParser()
	try:
		result = chain.invoke({
			"today": date.today().isoformat(),
			"conversation_history": conversation_history,
		})
		raw = result.get("date")
		return date.fromisoformat(raw) if raw else None
	except Exception:
		return None


def get_scheduling_advice(
	conversation_history: str,
	llm: ChatOpenAI,
	reference_date: date | None = None,
) -> dict:
	"""
	Two-stage flow:
	  1. Ask the LLM to classify whether it is the right time to schedule.
	  2. If it is, query the slot DB for the nearest available  3 slots and ask
	     the LLM again to draft an SMS proposing them.
	Returns the classification dict; when slots were retrieved, "suggested_slots"
	holds the DB rows and "response" holds the grounded SMS.

	reference_date is the date the conversation took place. The spec asks the
	advisor to anchor slot lookup to the conversation timestamp. For the live
	Streamlit demo no timestamp is passed, so we fall back to a date mentioned
	in the conversation and finally to date.today() (the slot DB is seeded for
	the current year, 2026).
	"""
	decision_prompt = ChatPromptTemplate.from_messages([
		("system", SCHEDULING_ADVISOR_PROMPT),
		("user", "{input}")
	])
	decision_chain = decision_prompt | llm | JsonOutputParser()
	result = decision_chain.invoke({"input": conversation_history})

	if result["action"] == "schedule":
		# Prefer: explicit reference_date → date mentioned in conversation → today
		mentioned_date = reference_date or _extract_date(conversation_history, llm)
		ref = mentioned_date or date.today()
		slots = get_slots_by_day(ref, "Python Dev", 3)
		result["suggested_slots"] = slots

		if slots:
			slots_text = "\n".join(
				f"- {s['date']} at {s['time']}" for s in slots
			)
			# Tell the prompt what date the candidate requested (if any),
			# so it can apologise when that date isn't available.
			requested_date_str = f"{mentioned_date:%A, %B} {mentioned_date.day}" if mentioned_date else "none"
			answer_prompt = ChatPromptTemplate.from_messages([
				("system", SCHEDULING_ADVISOR_ANSWER_PROMPT),
			])
			answer_chain = answer_prompt | llm | StrOutputParser()
			result["response"] = answer_chain.invoke({
				"slots": slots_text,
				"conversation_history": conversation_history,
				"requested_date": requested_date_str,
				"today": date.today().isoformat(),
			})

	return result
