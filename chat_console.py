import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import get_bot_response

def main():
	messages = []
	print("SMS Recruitment Chatbot - Console Mode")
	print("Type 'quit' to exit\n")

	while True:
		user_input = input("You: ")
		if user_input.lower() == "quit":
			break

		messages.append({"role": "user", "content": user_input})

		try:
			result = get_bot_response(messages)
			action = result["action"]
			response = result["response"]

			print(f"Bot [{action}]: {response}\n")
			messages.append({"role": "assistant", "content": response})

			if action == "end":
				print("--- Conversation ended ---")
				break
		except Exception as e:
			print(f"Error: {e}\n")

if __name__ == "__main__":
	main()
