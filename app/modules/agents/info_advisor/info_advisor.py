from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from app.modules.agents.info_advisor.pdf_embedder import get_retriever

INFO_ADVISOR_PROMPT = """You are the Info Advisor for an SMS recruitment chatbot hiring for a Python Developer position.

Your job is to help formulate the next message to the candidate, answer their questions, and keep the conversation moving toward scheduling an interview.

You must decide ONE of two actions:

ACTION: info_needed
Use when:
 - The candidate asked a question about the position, company, or role (technologies used, work model, requirements, benefits, etc.)
 - You need to retrieve information from the job description to answer accurately
In this case, set "query" to the candidate's question so relevant information
can be looked up. Leave "response" as null — it will be filled in after retrieval.

ACTION: info_not_needed
Use when:
 - The candidate is sharing their background or experience
 - The candidate is discussing availability, agreeing to a slot, declining, or making small talk
 - The conversation can continue without looking up information from the JD
In this case, set both "query" and "response" to null. The Main Agent's
original draft will be used as the reply.

EXAMPLES:

Example 1 — candidate asks about the tech stack → info_needed
Conversation:
Recruiter: Tell me about your Python experience.
Candidate: I've used Flask for a couple of years. What frameworks does your team work with?
Output: {{"action": "info_needed", "query": "What frameworks does the team use?", "response": null}}

Example 2 — candidate shares background → info_not_needed
Conversation:
Recruiter: Hi, what's your Python background like?
Candidate: I've been writing Python for four years on backend services.
Output: {{"action": "info_not_needed", "query": null, "response": null}}

Example 3 — candidate accepting a slot → info_not_needed
Conversation:
Recruiter: Could you do Wednesday at 11 AM?
Candidate: Wednesday at 11 AM works.
Output: {{"action": "info_not_needed", "query": null, "response": null}}

Example 4 — candidate asks about the work model → info_needed
Conversation:
Recruiter: Could you tell me about your SQL background?
Candidate: I've used Postgres heavily. Is the role remote, hybrid, or onsite?
Output: {{"action": "info_needed", "query": "Is the role remote, hybrid, or onsite?", "response": null}}

Respond with a JSON object:
{{
  "action": "info_needed" | "info_not_needed",
  "query": "question to look up in job description" or null,
  "response": null
}}
"""

INFO_ADVISOR_ANSWER_PROMPT = """You are the Info Advisor for an SMS recruitment chatbot hiring for a Python Developer position.

The candidate asked a question about the role. Below are the most relevant excerpts from the official Python
Developer job description, retrieved from our knowledge base.

JOB DESCRIPTION EXCERPTS:
 {context}

CONVERSATION HISTORY:
 {conversation_history}

Your task: write the next SMS reply to the candidate.

RULES:
  - Answer the candidate's question using ONLY the information in the excerpts above
  - If the excerpts do not contain the answer, say so honestly (e.g. "I'm not sure on that one — I can check with the recruiter")
  - Do NOT invent details about salary, benefits, location, or requirements that are not in the excerpts
  - Keep it short and conversational — this is SMS, not an email
  - After answering, gently steer the conversation toward scheduling an interview
  - Do not include any preamble like "Here is the answer:" — just write the SMS

Respond with the SMS message text only. No JSON, no quotes, no formatting."""

# Module-level singleton — loading Chroma is expensive, do it once at import.
_RETRIEVER = get_retriever()


def get_info_advice(conversation_history: str, llm: ChatOpenAI) -> dict:
    """
    Two-stage flow:
      1. Ask the LLM to classify whether the candidate's last turn needs JD lookup.
      2. If it does, retrieve relevant JD chunks from Chroma and ask the LLM
         again to rewrite the SMS reply grounded in those chunks.
    Returns the classification dict; when retrieval ran, "response" is the
    grounded SMS instead of the LLM's first-pass guess.
    """
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", INFO_ADVISOR_PROMPT),
        ("user", "{input}")
    ])
    decision_chain = decision_prompt | llm | JsonOutputParser()
    result = decision_chain.invoke({"input": conversation_history})

    # Defensive: the prompt asks the LLM to leave "response" null on the
    # decision pass, but enforce it here so the Main Agent can rely on
    # "response is None" meaning "no grounded reply from me".
    result["response"] = None

    # Guard both fields — the LLM occasionally returns info_needed with a null query.
    if result["action"] == "info_needed" and result["query"] is not None:
        documents = _RETRIEVER.invoke(result["query"])
        context = "\n".join(doc.page_content for doc in documents)

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", INFO_ADVISOR_ANSWER_PROMPT),
        ])
        answer_chain = answer_prompt | llm | StrOutputParser()
        result["response"] = answer_chain.invoke({
            "context": context,
            "conversation_history": conversation_history,
        })

    return result

