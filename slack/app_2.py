import sys
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
# Flask is a web application framework written in Python
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        # Initialize the Slack client with your bot token
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")

import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Initialize the OpenAI-based language model with conversational capabilities
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Create a memory buffer for maintaining context in conversations
memory = ConversationBufferMemory()

# System prompt template to define AI assistant's behavior
system_template = """
You are Alphie, a conversational AI assistant at New Mexico Tax & Rev. 
Your role is to engage in meaningful conversations and help with a variety of tasks, including answering general questions, solving coding issues, and more.
Provide thoughtful and friendly responses to all user questions. 


One of your primary tasks is to analyze the Current New Mexico Statutes Annotated 1978. Make sure to do a detailed analysis. 

"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Human message prompt to integrate user input into the conversation
human_template = "User says: {user_input}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Create a conversational prompt with system and human messages
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Create a conversational chain with memory for maintaining context
chain = LLMChain(llm=chat, prompt=chat_prompt, memory=memory)

# Define the event listener for Slack messages
@app.event("message")
def handle_message_events(event, say):
    user_text = event.get("text", "")

    # If the user text is a greeting
    if user_text.lower() in ["hi", "hello", "hey"]:
        say("Hello! I'm Alphie, your friendly AI assistant at New Mexico Tax & Rev. How can I help you today?")
        return

    # Generate the response with the conversational chain
    response = chain.run(user_input=user_text)

    # Respond with the generated response
    say(response)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)

# Initialize the Flask app
if __name__ == "__main__":
    flask_app.run()