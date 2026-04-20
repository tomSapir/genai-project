from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

EXIT_ADVISOR_PROMPT = """You are the Exit Advisor for an SMS recruitment chatbot hiring for a Python Developer position.

Your job is to evaluate the conversation history and determine whether the conversation should end.

You must decide ONE of two actions:

ACTION: end
Use when:
- The candidate explicitly states they are not interested in the position
- The candidate mentions they have already found or accepted another job
- The candidate asks to stop receiving messages
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

def get_exit_advice(conversation_history: str, llm: ChatOpenAI) -> dict:
    parser = JsonOutputParser()

	# Create a ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages([
		("system", EXIT_ADVISOR_PROMPT),
		("user", "{input}")
	])

	# Chain the prompt with the llm using the pipe operator:
    chain = prompt | llm | parser

    # Invoke the chain, passing in the conversation_history and return the result:
    return chain.invoke({"input": conversation_history})





