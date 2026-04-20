from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from app.modules.agents.main_agent import get_main_agent_response

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_conversation_history(messages: list) -> str:
    role_map = {
        "user": "Candidate",
        "assistant": "Recruiter"
    }

    lines = []

    for msg in messages:
        role = role_map.get(msg.get("role"), msg.get("role", "Unknown"))
        content = msg.get("content", "")

        lines.append(f"{role}: {content}")
    
    return "\n".join(lines)


def get_bot_response(messages: list) -> dict:
    return get_main_agent_response(format_conversation_history(messages), llm)


