import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from openai import AuthenticationError, NotFoundError, PermissionDeniedError

EXIT_ADVISOR_PROMPT = """You are the Exit Advisor for an SMS recruitment chatbot hiring for a Python Developer position.

Your job is to evaluate the conversation history and determine whether the conversation should end.

You must decide ONE of two actions:

ACTION: end
Use when:
- The candidate explicitly states they are not interested in the position
- The candidate mentions they have already found or accepted another job
- The candidate asks to stop receiving messages
- The candidate has just named or accepted a specific time slot
  (e.g. "Monday at 3 PM is good", "Yes, Thursday works") — the recruiter
  is about to confirm the booking, so ending the conversation is appropriate
- The interview has been successfully confirmed and there is nothing left to discuss
- The candidate is unresponsive or clearly disengaged after multiple attempts

ACTION: dont_end
Use when:
- The candidate is still engaged in the conversation
- The candidate is asking questions or sharing information
- The candidate has shown interest but no interview has been scheduled yet
- There is still useful information to gather or share

Respond with a JSON object:
{{
  "action": "end" | "dont_end",
  "reason": "brief explanation for your decision"
}}
"""

def _resolve_llm(default_llm: ChatOpenAI) -> ChatOpenAI:
    # When EXIT_ADVISOR_MODEL is set (typically a fine-tuned `ft:...` id),
    # the Exit Advisor runs on its own dedicated model rather than the
    # shared LLM the Main Agent uses for everything else.
    model_id = os.getenv("EXIT_ADVISOR_MODEL")
    if not model_id:
        return default_llm
    return ChatOpenAI(model=model_id, temperature=0)


def get_exit_advice(conversation_history: str, llm: ChatOpenAI) -> dict:
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXIT_ADVISOR_PROMPT),
        ("user", "{input}")
    ])

    resolved = _resolve_llm(llm)
    payload = {"input": conversation_history}

    try:
        return (prompt | resolved | parser).invoke(payload)
    except (NotFoundError, AuthenticationError, PermissionDeniedError):
        # EXIT_ADVISOR_MODEL is set but unreachable with this API key — most
        # commonly because the fine-tuned model belongs to a different OpenAI
        # org. Fall back to the base llm so the pipeline keeps working.
        if resolved is llm:
            raise
        return (prompt | llm | parser).invoke(payload)
